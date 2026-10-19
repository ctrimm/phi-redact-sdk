"""Integration tests for the phi-redact Python SDK.

These exercise the installed package as a whole: the public API surface,
an end-to-end redact/restore pass over a realistic synthetic note, and the
shape of the payload the `serve` HTTP API forwards to clients.
"""

import json

import pytest

import phi_redact

CLINICAL_NOTE = (
    "Patient intake 03/14/1978. Maria Santos, DOB 03/14/1978, SSN 234-56-7890. "
    "Contact: 415-555-0132, msantos@example.com. "
    "Card on file 4111111111111111 (ignore 1234567812345678). "
    "MRN 48291375, ZIP 90210. Follow up at https://portal.example.com."
)


class TestPublicSurface:
    def test_all_exports_importable(self):
        for name in phi_redact.__all__:
            assert getattr(phi_redact, name) is not None, name

    def test_module_functions_are_rules_only(self):
        r = phi_redact.redact("Call 415-555-0132")
        assert r.engine == "rules"
        assert r.model is None


class TestEndToEnd:
    def test_clinical_note_redact_restore(self):
        r = phi_redact.redact(CLINICAL_NOTE)
        # every detected label got a placeholder
        assert r.counts.get("PHONE") == 1
        assert r.counts.get("SSN") == 1
        assert r.counts.get("EMAIL") == 1
        assert r.counts.get("CREDITCARD") == 1
        assert r.counts.get("DATE") == 2
        # the Luhn-invalid card is not redacted
        assert "1234567812345678" in r.redacted
        # no raw PII from the mapping leaks into the redacted text,
        # except the deliberately ignored invalid card
        for original in r.mapping.values():
            if original != "1234567812345678":
                assert original not in r.redacted
        # full round-trip
        assert phi_redact.restore(r.redacted, r.mapping) == CLINICAL_NOTE

    def test_result_survives_json_round_trip(self):
        r = phi_redact.redact(CLINICAL_NOTE)
        d = json.loads(json.dumps(r.to_dict()))
        assert phi_redact.restore(d["redacted"], d["mapping"]) == CLINICAL_NOTE

    def test_serve_api_contract_keys(self):
        """The serve HTTP API forwards to_dict(); clients depend on these."""
        d = phi_redact.redact("Call 415-555-0132").to_dict()
        assert set(d) >= {
            "redacted", "entities", "mapping", "counts", "engine", "model",
        }
        for e in d["entities"]:
            assert set(e) >= {"start", "end", "label", "text", "source"}


class TestRedactorIntegration:
    def test_redactor_end_to_end_rules_only(self):
        r = phi_redact.Redactor()
        out = r.redact(CLINICAL_NOTE)
        assert out.engine == "rules"
        assert phi_redact.Redactor.restore(
            out.redacted, out.mapping) == CLINICAL_NOTE

    def test_scan_matches_module_scan(self):
        assert (phi_redact.Redactor().scan(CLINICAL_NOTE)
                == phi_redact.scan(CLINICAL_NOTE))

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("onnxruntime") is None,
        reason="onnx extra not installed",
    )
    def test_neural_engine_reports_missing_file(self):
        # onnxruntime present but no weights: a clear load error, not a
        # silent rules-only fallback
        with pytest.raises(Exception):
            phi_redact.Redactor(model="/nonexistent/model.onnx",
                                tokenizer="/nonexistent/tokenizer.json")
