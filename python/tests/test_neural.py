"""Tests for neural-engine special token handling.

The ONNX engine must feed the model the start/end markers from its own
tokenizer vocabulary: [CLS]/[SEP] (101/102) for the v7 WordPiece model,
<s>/</s> (0/2) for the legacy BPE smoke vocab. Hardcoding either pair
silently degrades the other model family (issue 2026-10-18).
"""

import json

import pytest

from phi_redact.engine import BOS, EOS, OnnxEngine, _special_ids

tokenizers = pytest.importorskip("tokenizers")
onnxruntime = pytest.importorskip("onnxruntime")


class _StubTok:
    def __init__(self, ids):
        self._ids = ids

    def token_to_id(self, token):
        return self._ids.get(token)


class TestSpecialIds:
    def test_bert_vocab_uses_cls_sep(self):
        tok = _StubTok({"[CLS]": 101, "[SEP]": 102})
        assert _special_ids(tok) == (101, 102)

    def test_legacy_bpe_falls_back(self):
        tok = _StubTok({"<s>": 0, "</s>": 2})
        assert _special_ids(tok) == (BOS, EOS) == (0, 2)

    def test_partial_markers_fall_back(self):
        # Only one of the pair present: not a BERT-style vocab.
        tok = _StubTok({"[CLS]": 101})
        assert _special_ids(tok) == (BOS, EOS)


def _wordpiece_tok(tmp_path, vocab):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece

    tok = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    return path


class _FakeSession:
    def __init__(self, *a, **k):
        pass

    def get_inputs(self):
        return [type("I", (), {"name": "input_ids"})(),
                type("I", (), {"name": "attention_mask"})()]


def _labels_file(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"id2label": {"0": "O", "1": "B-NAME"}}))
    return str(p)


def _truncating_tok(tmp_path, vocab, cap):
    """A tokenizer file that mimics the published artifact's shape:
    WordPiece + [CLS]/[SEP] template processing + a truncation cap
    smaller than the engine's chunk budget (production: cap 256,
    engine max_len 384)."""
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordPiece
    try:
        from tokenizers.processors import TemplateProcessing
    except ImportError:  # tokenizers < 0.20 name
        from tokenizers.post_processors import TemplateProcessing

    tok = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tok.enable_truncation(max_length=cap)
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    return path


class _ScriptedSession:
    """Fake ONNX session: labels one chosen token id as B-NAME_PATIENT,
    everything else O. Logits are extreme so softmax confidence ~1."""

    TARGET_ID = 113  # "ellison" in the test vocab

    def __init__(self, *a, **k):
        pass

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in
                ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _outputs, feed):
        import numpy as np

        ids = feed["input_ids"][0]
        logits = np.zeros((1, len(ids), 2), dtype=np.float64)
        logits[0, :, 0] = 10.0                      # O everywhere
        hit = ids == self.TARGET_ID
        logits[0, hit, 0] = 0.0
        logits[0, hit, 1] = 10.0                    # B-NAME_PATIENT on target
        return [logits]


_VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 101, "[SEP]": 102,
          "the": 110, "patient": 111, "improves": 112, "ellison": 113}


class TestTokenizerTruncation:
    """Regression (2026-10-25): the published tokenizer.json truncates at
    256 tokens. The engine chunks at max_len=384, but its length check
    read the truncated length, so chunking never engaged and text past
    token ~256 was silently never scanned."""

    def _engine(self, tmp_path, monkeypatch, cap=24, max_len=32):
        pytest.importorskip("numpy")
        tok_path = _truncating_tok(tmp_path, _VOCAB, cap)
        monkeypatch.setattr(onnxruntime, "InferenceSession", _ScriptedSession)
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps(
            {"id2label": {"0": "O", "1": "B-NAME_PATIENT"}}))
        return OnnxEngine(model_path="fake.onnx", tokenizer_path=tok_path,
                          labels_path=str(labels), max_len=max_len)

    def test_engine_load_strips_tokenizer_truncation(self, tmp_path, monkeypatch):
        eng = self._engine(tmp_path, monkeypatch)
        ids = eng.tok.encode("the patient improves " * 40).ids
        assert len(ids) > 24  # cap would have held this at 24

    def test_long_text_is_fully_scanned(self, tmp_path, monkeypatch):
        eng = self._engine(tmp_path, monkeypatch)
        text = "the patient improves " * 30 + "ellison " + "the patient improves " * 5
        ents = eng.predict(text)
        hits = [e for e in ents if e["text"] == "ellison"]
        assert len(hits) == 1
        assert hits[0]["start"] == text.index("ellison")
        assert hits[0]["label"] == "NAME_PATIENT"

    def test_short_text_still_scanned(self, tmp_path, monkeypatch):
        eng = self._engine(tmp_path, monkeypatch)
        ents = eng.predict("the patient ellison improves")
        assert [e["text"] for e in ents] == ["ellison"]


class TestOnnxEngineWiring:
    def test_engine_reads_ids_from_tokenizer(self, tmp_path, monkeypatch):
        tok_path = _wordpiece_tok(tmp_path, {
            "[PAD]": 0, "[UNK]": 1, "[CLS]": 101,
            "[SEP]": 102, "hello": 103,
        })
        monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
        eng = OnnxEngine(model_path="fake.onnx", tokenizer_path=tok_path,
                         labels_path=_labels_file(tmp_path))
        assert (eng.bos_id, eng.eos_id) == (101, 102)

    def test_engine_falls_back_without_cls_sep(self, tmp_path, monkeypatch):
        tok_path = _wordpiece_tok(tmp_path, {
            "<s>": 0, "<pad>": 1, "</s>": 2, "hello": 3, "[UNK]": 4,
        })
        monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
        eng = OnnxEngine(model_path="fake.onnx", tokenizer_path=tok_path,
                         labels_path=_labels_file(tmp_path))
        assert (eng.bos_id, eng.eos_id) == (0, 2)
