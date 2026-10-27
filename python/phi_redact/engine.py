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

Anonymize operators (0.4.0, Presidio parity): the placeholder style is
an engine option — "numbered" (default, reversible via the mapping),
"label" (<PHONE>, Presidio-style, irreversible), "mask" (partial
reveal), "hash" (salted pseudonym token, join-preserving), "remove"
(delete the span). Only "numbered" returns a mapping; the other
operators return an empty one by design.

Allow/deny lists (0.4.0): allowlisted terms are never redacted;
denylisted terms are always redacted with a chosen label. Deny wins
over allow when a term sits in both lists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

from redact import rules

DEFAULT_MAX_LEN = 384          # tokens per ONNX chunk (incl. specials)
DEFAULT_THRESHOLD = 0.6        # min mean-token confidence for neural spans
# 0.6 since v9 (2026-10-21): on the golden set it dominates 0.5 —
# precision 0.746 vs 0.733, strict F1 0.677 vs 0.670, public retention
# 0.133 vs 0.033, at identical recall and clean-FP rate.
OVERLAP_CHARS = 96             # char overlap when a text is split for chunking

OPERATORS = ("numbered", "label", "mask", "hash", "remove")
DEFAULT_DENY_LABEL = "CUSTOM"  # label given to denylist-injected spans

BOS, EOS = 0, 2  # fallback for BPE vocabs without [CLS]/[SEP]
# (the project's legacy smoke vocab: <s>=0, </s>=2)


def _load_tokenizer(tokenizer_path: str):
    """Load a tokenizer with its serialized truncation cap removed.

    The published tokenizer.json carries a training-side truncation
    setting (max_length 256). This engine does its own chunking at
    max_len (see _split), so the tokenizer must report true lengths —
    with the cap in place, encode() silently returns at most 256 ids,
    _split's length check reads the truncated length, chunking never
    engages, and everything past the cap is never scanned. The cap is
    stripped from the serialized config before loading because the
    tokenizers library's runtime truncation toggles have changed shape
    across releases (attribute assignment pre-0.22, no_truncation()
    removed by 0.23); editing the config dict works on all of them.
    Tokenizers without a truncation block load unchanged.
    """
    from tokenizers import Tokenizer

    try:
        with open(tokenizer_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return Tokenizer.from_file(tokenizer_path)
    if isinstance(cfg, dict) and cfg.get("truncation"):
        cfg.pop("truncation")
        return Tokenizer.from_str(json.dumps(cfg))
    return Tokenizer.from_file(tokenizer_path)


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
        self.tok = _load_tokenizer(tokenizer_path)
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
        if back == 0:
            # Degenerate geometry: the overlap window covers the whole
            # left piece, so the right call would re-split the identical
            # text forever. Unreachable at the default max_len (a chunk
            # over budget is far longer than OVERLAP_CHARS), but possible
            # with a small custom max_len. Drop the overlap for this
            # split — it is a recall nicety at boundaries, not a
            # correctness requirement.
            back = cut
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


def apply_placeholders(text: str, entities: list[dict],
                       operator: str = "numbered", salt: str = ""):
    """Render entities into text under the chosen anonymize operator.

    Replacements are computed in forward document order (so the
    "numbered" operator numbers per label in reading order), then
    spliced back-to-front so offsets stay valid. Returns
    (redacted_text, mapping); the mapping is populated only for
    "numbered" — the other operators are irreversible by design.
    """
    if operator not in OPERATORS:
        raise ValueError(
            f"unknown operator {operator!r}; expected one of {OPERATORS}")
    counters: dict[str, int] = {}
    mapping: dict[str, str] = {}
    replacements: dict[int, str] = {}  # id(entity) -> rendered replacement
    for e in sorted(entities, key=lambda d: d["start"]):
        counters[e["label"]] = counters.get(e["label"], 0) + 1
        rep = _replacement(e, counters[e["label"]], operator, salt)
        e["placeholder"] = rep
        replacements[id(e)] = rep
        if operator == "numbered":
            mapping[rep] = e["text"]
    out = text
    for e in sorted(entities, key=lambda d: d["start"], reverse=True):
        out = out[:e["start"]] + replacements[id(e)] + out[e["end"]:]
    return out, mapping


def _replacement(e: dict, n: int, operator: str, salt: str) -> str:
    label = e["label"]
    if operator == "numbered":
        return f"[{label}_{n}]"
    if operator == "label":
        return f"<{label}>"
    if operator == "hash":
        return hash_token(label, e["text"], salt)
    if operator == "mask":
        return mask_value(label, e["text"])
    return ""  # remove


def hash_token(label: str, value: str, salt: str = "") -> str:
    """Consistent pseudonym token for the "hash" operator.

    [LABEL_<8 hex>] from SHA-256(salt \\x00 label \\x00 value). The same
    value under the same label and salt yields the same token in every
    document, so pseudonymized datasets still join. The TypeScript SDK
    uses the identical construction (shared test vectors pin it).

    The salt defaults to "" but should be a secret in practice: SSNs and
    other low-entropy values are dictionary-attackable from an unsalted
    truncated hash.
    """
    digest = hashlib.sha256(
        f"{salt}\x00{label}\x00{value}".encode("utf-8")).hexdigest()[:8]
    return f"[{label}_{digest}]"


# Labels whose values are structured IDs: masking keeps the last 4
# characters (the part support workflows quote) when the value is long
# enough for that to be safe.
_MASK_KEEP_LAST = {
    "SSN", "CREDITCARD", "PHONE", "TAX_ID", "ROUTING_NUMBER",
    "BANK_ACCOUNT", "ACCOUNT", "IBAN", "PASSPORT", "DRIVERS_LICENSE",
    "LICENSE", "GOVERNMENT_ID", "MRN",
}


def mask_value(label: str, value: str) -> str:
    """Masked rendering for the "mask" operator.

    Structured IDs (see _MASK_KEEP_LAST) of length >= 8 keep their last
    4 characters; preceding alphanumeric characters become "*" and
    separators are preserved ("234-56-7890" -> "***-**-7890"). Every
    other value is masked to its own length in asterisks, so a masked
    name leaks nothing but its length.
    """
    if label in _MASK_KEEP_LAST and len(value) >= 8:
        head, tail = value[:-4], value[-4:]
        return "".join(
            "*" if (c.isascii() and c.isalnum()) else c for c in head) + tail
    return "*" * len(value)


# ---------------------------------------------------------------------------
# Allow / deny lists
# ---------------------------------------------------------------------------

def apply_allowlist(entities: list[dict],
                    allowlist: list[str] | None) -> list[dict]:
    """Drop spans whose text matches an allowlist term (case-insensitive,
    exact string — a term protects exactly the value it names)."""
    if not allowlist:
        return entities
    allowed = {t.casefold() for t in allowlist}
    return [e for e in entities if e["text"].casefold() not in allowed]


def denylist_spans(text: str, denylist: list[str],
                   label: str = DEFAULT_DENY_LABEL) -> list[dict]:
    """Exact-match spans for denylist terms: case-insensitive,
    word-boundary aware (a term never matches inside a larger word)."""
    spans: list[dict] = []
    for term in denylist:
        if not term:
            continue
        rx = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                        re.IGNORECASE)
        for m in rx.finditer(text):
            spans.append({"start": m.start(), "end": m.end(),
                          "label": label, "text": m.group(),
                          "source": "denylist"})
    return spans


def inject_denylist(text: str, entities: list[dict],
                    denylist: list[str] | None,
                    label: str = DEFAULT_DENY_LABEL) -> list[dict]:
    """Add denylist spans to a detected entity list.

    A deny span wins any overlap: a detected span it touches is dropped
    (the rest of that span is left in the clear — the deny match is
    what the operator asked to hide). Overlapping deny matches keep the
    first in document order.
    """
    if not denylist:
        return entities
    deny = denylist_spans(text, denylist, label)
    if not deny:
        return entities
    kept = [e for e in entities
            if not any(_overlaps(e, d) for d in deny)]
    merged: list[dict] = []
    for d in sorted(deny, key=lambda x: x["start"]):
        if any(_overlaps(d, k) for k in merged):
            continue
        merged.append(d)
    return sorted(kept + merged, key=lambda d: d["start"])


def process_entities(text: str, entities: list[dict],
                     allowlist: list[str] | None = None,
                     denylist: list[str] | None = None,
                     deny_label: str = DEFAULT_DENY_LABEL) -> list[dict]:
    """Allowlist filter, then denylist injection. Deny wins over allow:
    a term in both lists is redacted (as a deny span)."""
    entities = apply_allowlist(entities, allowlist)
    entities = inject_denylist(text, entities, denylist, deny_label)
    return entities


def redact(text: str, neural: OnnxEngine | None = None, *,
           operator: str = "numbered", salt: str = "",
           allowlist: list[str] | None = None,
           denylist: list[str] | None = None,
           deny_label: str = DEFAULT_DENY_LABEL) -> RedactionResult:
    rule_ents = rules.scan(text)
    neural_ents = neural.predict(text) if neural is not None else []
    entities = merge_entities(rule_ents, neural_ents)
    entities = process_entities(text, entities, allowlist, denylist,
                                deny_label)
    redacted, mapping = apply_placeholders(text, entities,
                                           operator=operator, salt=salt)
    counts: dict[str, int] = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return RedactionResult(
        redacted=redacted, entities=entities, mapping=mapping,
        counts=counts,
        engine="rules+onnx" if neural is not None else "rules",
        model=getattr(neural, "model_path", None),
    )


# ---------------------------------------------------------------------------
# Streaming restore (LLM round-trips)
# ---------------------------------------------------------------------------

def _partial_suffix_len(buf: str, keys) -> int:
    """Length of the longest buffer suffix that is a proper prefix of
    some mapping key — text that might still grow into a placeholder
    when the next chunk arrives, so it must not be emitted yet."""
    if not keys:
        return 0
    maxk = max(len(k) for k in keys)
    best = 0
    for n in range(1, min(len(buf), maxk - 1) + 1):
        suffix = buf[len(buf) - n:]
        if any(len(k) > n and k.startswith(suffix) for k in keys):
            best = n
    return best


class ChunkedRestorer:
    """Restore placeholders from text that arrives in chunks (e.g. a
    streaming LLM response built on scrubbed input).

    feed() returns the restored text that is safe to emit now; a
    trailing partial placeholder is buffered until it completes or
    flush() proves it was never a placeholder. Placeholders split
    across chunks are restored whole; unknown or partial placeholders
    pass through untouched — feed/flush never raise and never invent
    text.

        cr = ChunkedRestorer(mapping)
        out = cr.feed(chunk1) + cr.feed(chunk2) + cr.flush()
    """

    def __init__(self, mapping: dict):
        self.mapping = dict(mapping)
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        hold = _partial_suffix_len(self._buf, self.mapping)
        if hold:
            emit, self._buf = self._buf[:-hold], self._buf[-hold:]
        else:
            emit, self._buf = self._buf, ""
        return rules.restore_text(emit, self.mapping)

    def flush(self) -> str:
        out = rules.restore_text(self._buf, self.mapping)
        self._buf = ""
        return out
