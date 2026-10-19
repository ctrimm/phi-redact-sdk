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
