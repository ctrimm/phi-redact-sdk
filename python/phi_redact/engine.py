"""phi-redact serving engine.

Two layers, merged into one entity list:
  1. rules   - redact/rules.py: checksum-gated deterministic scans
               (cards, SSN, IBAN, phone, email, URL, IP, ZIP, date, MRN).
               Always runs. Wins every overlap dispute.
  2. neural  - ONNX token-classification model (BIOES), for what rules
               cannot do: names, addresses, employers, PHI in free prose.
               Optional; falls back to rules-only when no model is given.

Placeholder policy: numbered per label in forward document order,
e.g. [NAME_PATIENT_1], [PHONE_2]. The mapping {placeholder: original}
is returned for pseudonymized round-trips; delete it and the copy is
anonymized. restore_text() in redact/rules.py reverses it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from redact import rules

DEFAULT_MAX_LEN = 384          # tokens per ONNX chunk (incl. specials)
DEFAULT_THRESHOLD = 0.5        # min mean-token confidence for neural spans
OVERLAP_CHARS = 96             # char overlap when a text is split for chunking

BOS, EOS = 0, 2  # fallback for BPE vocabs without [CLS]/[SEP]
# (the project's legacy smoke vocab: <s>=0, </s>=2)


def _special_ids(tok) -> tuple[int, int]:
    """Start/end marker ids for this tokenizer's vocabulary.

    BERT-style WordPiece vocabs mark sequences with [CLS]/[SEP] (101/102
    in the v7 model); the legacy BPE smoke vocab uses <s>/</s> (0/2).
    Hardcoding either pair silently feeds the wrong markers to the other
    model family and depresses neural scores, so read them from the
    tokenizer and fall back to BOS/EOS only when [CLS]/[SEP] are absent.
    """
    cls_id = tok.token_to_id("[CLS]")
    sep_id = tok.token_to_id("[SEP]")
    if cls_id is not None and sep_id is not None:
        return cls_id, sep_id
    return BOS, EOS


@dataclass
class RedactionResult:
    redacted: str
    entities: list[dict] = field(default_factory=list)  # start,end,label,text,source,score,placeholder
    mapping: dict = field(default_factory=dict)          # placeholder -> original
    counts: dict = field(default_factory=dict)           # label -> n
    engine: str = "rules"
    model: str | None = None

    def to_dict(self) -> dict:
        """JSON-serializable view of the result."""
        return {
            "redacted": self.redacted,
            "entities": self.entities,
            "mapping": self.mapping,
            "counts": self.counts,
            "engine": self.engine,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Neural engine (ONNX)
# ---------------------------------------------------------------------------

class OnnxEngine:
    """Token-classification inference over ONNX, chunked for long texts."""

    def __init__(self, model_path: str, tokenizer_path: str,
                 labels_path: str | None = None,
                 max_len: int = DEFAULT_MAX_LEN,
                 threshold: float = DEFAULT_THRESHOLD):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_path = model_path
        self.max_len = max_len
        self.threshold = threshold
        self.tok = Tokenizer.from_file(tokenizer_path)
        # Marker ids come from the tokenizer's own vocabulary: v7's
        # WordPiece vocab resolves [CLS]/[SEP] (101/102); the legacy BPE
        # smoke vocab falls back to <s>/</s> (0/2).
        self.bos_id, self.eos_id = _special_ids(self.tok)
        self.sess = ort.InferenceSession(model_path,
                                         providers=["CPUExecutionProvider"])
        # v7-class graphs require token_type_ids; older graphs don't have
        # that input. Feed exactly what the graph declares, so the stock
        # engine works for both without a subclass workaround.
        self._input_names = {i.name for i in self.sess.get_inputs()}
        labels_path = (labels_path
                       or os.environ.get("PHI_REDACT_LABELS")
                       or os.path.join(os.path.dirname(model_path),
                                       "config.json"))
        with open(labels_path) as f:
            cfg = json.load(f)
        id2label = cfg.get("id2label") or {
            v: k for k, v in cfg["label2id"].items()}
        self.id2label = {int(k): v for k, v in id2label.items()}

    # -- chunking ---------------------------------------------------------
    def _split(self, text: str, off: int, out: list[tuple[str, int]]) -> None:
        if len(self.tok.encode(text).ids) <= self.max_len - 2:
            out.append((text, off))
            return
        mid = len(text) // 2
        cut = mid
        for i in range(1, len(text) // 2):
            for j in (mid + i, mid - i):
                if 0 < j < len(text) and text[j] in " \t\n":
                    cut = j
                    break
            else:
                continue
            break
        self._split(text[:cut], off, out)
        back = max(0, cut - OVERLAP_CHARS)
        self._split(text[back:], off + back, out)

    # -- inference --------------------------------------------------------
    def predict(self, text: str) -> list[dict]:
        import numpy as np

        chunks: list[tuple[str, int]] = []
        self._split(text, 0, chunks)
        entities: list[dict] = []
        for chunk, coff in chunks:
            enc = self.tok.encode(chunk)
            ids = [self.bos_id] + enc.ids + [self.eos_id]
            mask = [1] * len(ids)
            offs = [(0, 0)] + [(s + coff, e + coff) for s, e in enc.offsets] \
                + [(0, 0)]
            feed = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
                "token_type_ids": np.zeros((1, len(ids)), dtype=np.int64),
            }
            logits = self.sess.run(
                ["logits"],
                {k: v for k, v in feed.items() if k in self._input_names},
            )[0][0]
            probs = _softmax(logits)
            pred = probs.argmax(axis=-1)
            entities.extend(self._decode(text, offs, pred, probs))
        # dedupe identical spans produced by chunk overlap, keep best score
        best: dict[tuple, dict] = {}
        for e in entities:
            key = (e["start"], e["end"], e["label"])
            if key not in best or e["score"] > best[key]["score"]:
                best[key] = e
        return sorted(best.values(), key=lambda d: d["start"])

    def _decode(self, text: str, offs, pred, probs) -> list[dict]:
        entities: list[dict] = []
        cur: list | None = None  # [label, start, end, confidences]

        def flush():
            nonlocal cur
            if cur is None:
                return
            label, s, e, confs = cur
            cur = None
            s, e = _trim(text, s, e)
            if s < e:
                entities.append({
                    "start": s, "end": e, "label": label,
                    "text": text[s:e], "source": "neural",
                    "score": round(sum(confs) / len(confs), 4),
                })

        for i, ((s, e), lid) in enumerate(zip(offs, pred)):
            if s == e == 0:                      # special token
                flush()
                continue
            tag = self.id2label.get(int(lid), "O")
            if tag == "O":
                flush()
                continue
            prefix, _, typ = tag.partition("-")
            conf = float(probs[i].max())
            if prefix == "S":
                flush()
                s2, e2 = _trim(text, s, e)
                if s2 < e2:
                    entities.append({
                        "start": s2, "end": e2, "label": typ,
                        "text": text[s2:e2], "source": "neural",
                        "score": round(conf, 4),
                    })
            elif prefix == "B":
                flush()
                cur = [typ, s, e, [conf]]
            else:  # I / E, lenient on malformed transitions
                if cur is not None and cur[0] == typ:
                    cur[2] = e
                    cur[3].append(conf)
                else:
                    cur = [typ, s, e, [conf]]
                if prefix == "E":
                    flush()
        flush()
        return [e for e in entities if e["score"] >= self.threshold]


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _softmax(x):
    import numpy as np
    m = x.max(axis=-1, keepdims=True)
    ex = np.exp(x - m)
    return ex / ex.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Merge + placeholders
# ---------------------------------------------------------------------------

def merge_entities(rule_ents: list[dict], neural_ents: list[dict]) -> list[dict]:
    """Rules win every overlap. Neural-neural overlaps keep the best score."""
    kept: list[dict] = []
    for n in sorted(neural_ents, key=lambda d: -d.get("score", 0.0)):
        if any(_overlaps(n, r) for r in rule_ents):
            continue
        if any(_overlaps(n, k) for k in kept):
            continue
        kept.append(n)
    return sorted(rule_ents + kept, key=lambda d: d["start"])


def _overlaps(a: dict, b: dict) -> bool:
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])


def apply_placeholders(text: str, entities: list[dict]):
    """Replace back-to-front, then renumber per label in document order."""
    mapping: dict[str, str] = {}
    out = text
    tmp: dict[int, str] = {}  # entity index -> temp placeholder
    for i, e in enumerate(sorted(entities, key=lambda d: d["start"],
                                 reverse=True)):
        ph = f"[\x00{i}]"
        tmp[id(e)] = ph
        mapping[ph] = e["text"]
        out = out[:e["start"]] + ph + out[e["end"]:]
    # renumber forward per label
    counters: dict[str, int] = {}
    final: dict[str, str] = {}
    ordered = sorted(entities, key=lambda d: d["start"])
    for e in ordered:
        counters[e["label"]] = counters.get(e["label"], 0) + 1
        new_ph = f"[{e['label']}_{counters[e['label']]}]"
        e["placeholder"] = new_ph
        final[new_ph] = mapping[tmp[id(e)]]
        out = out.replace(tmp[id(e)], new_ph)
    return out, final


def redact(text: str, neural: OnnxEngine | None = None) -> RedactionResult:
    rule_ents = rules.scan(text)
    neural_ents = neural.predict(text) if neural is not None else []
    entities = merge_entities(rule_ents, neural_ents)
    redacted, mapping = apply_placeholders(text, entities)
    counts: dict[str, int] = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return RedactionResult(
        redacted=redacted, entities=entities, mapping=mapping,
        counts=counts,
        engine="rules+onnx" if neural is not None else "rules",
        model=getattr(neural, "model_path", None),
    )
