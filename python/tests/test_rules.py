"""Unit tests for the phi-redact Python SDK (rules layer).

Covers validators, scan spans, placeholder numbering, merge policy,
restore round-trips, and the Redactor class in rules-only mode.
Synthetic fixtures only — no real PII anywhere in this file.
"""

import json

import pytest

import phi_redact
from phi_redact import (
    DEFAULT_MAX_LEN,
    DEFAULT_THRESHOLD,
    RedactionResult,
    Redactor,
    apply_placeholders,
    merge_entities,
)
from redact import rules

NOTE = (
    "Pt Maria Santos DOB 03/14/1978 SSN 234-56-7890, ph 415-555-0132, "
    "jane@example.com, card 4111111111111111, MRN 48291375, ZIP 90210."
)


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

class TestValidators:
    def test_luhn_ok(self):
        assert rules.luhn_ok("4111111111111111") is True
        assert rules.luhn_ok("1234567812345678") is False
        assert rules.luhn_ok("123") is False

    def test_ssn_valid(self):
        assert rules.ssn_valid("234-56-7890") is True
        assert rules.ssn_valid("000-12-3456") is False   # bad area
        assert rules.ssn_valid("666-12-3456") is False   # bad area
        assert rules.ssn_valid("900-12-3456") is False   # bad area
        assert rules.ssn_valid("234-00-7890") is False   # bad group
        assert rules.ssn_valid("234-56-0000") is False   # bad serial

    def test_iban_valid(self):
        assert rules.iban_valid("GB29NWBK60161331926819") is True
        assert rules.iban_valid("GB29NWBK60161331926818") is False


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

class TestScan:
    def test_finds_all_entity_types(self):
        labels = sorted(e["label"] for e in phi_redact.scan(NOTE))
        assert labels == [
            "CREDITCARD", "DATE", "EMAIL", "MRN",
            "PHONE", "SSN", "ZIP",
        ]

    def test_skips_luhn_invalid_card(self):
        ents = phi_redact.scan("card 1234567812345678 charged")
        assert not any(e["text"] == "1234567812345678" for e in ents)
        assert not any(e["label"] == "CREDITCARD" for e in ents)

    def test_spans_exact_and_sorted(self):
        text = "call 415-555-0132 or jane@example.com"
        ents = phi_redact.scan(text)
        starts = [e["start"] for e in ents]
        assert starts == sorted(starts)
        for e in ents:
            assert text[e["start"]:e["end"]] == e["text"]

    def test_golden_spans(self):
        """Pinned spans for a fixed note — catches regex regressions."""
        got = [
            {"start": e["start"], "end": e["end"], "label": e["label"]}
            for e in phi_redact.scan(NOTE)
        ]
        assert got == [
            {"start": 20, "end": 30, "label": "DATE"},
            {"start": 35, "end": 46, "label": "SSN"},
            {"start": 51, "end": 63, "label": "PHONE"},
            {"start": 65, "end": 81, "label": "EMAIL"},
            {"start": 88, "end": 104, "label": "CREDITCARD"},
            {"start": 106, "end": 118, "label": "MRN"},
            {"start": 124, "end": 129, "label": "ZIP"},
        ]

    def test_empty_and_clean_text(self):
        assert phi_redact.scan("") == []
        assert phi_redact.scan("nothing to see here, just words") == []

    def test_url_ip_iban(self):
        ents = phi_redact.scan(
            "see https://example.com/x from 10.0.0.1, "
            "IBAN GB29NWBK60161331926819"
        )
        by_label = {e["label"]: e["text"] for e in ents}
        assert by_label["URL"] == "https://example.com/x"
        assert by_label["IP"] == "10.0.0.1"
        assert by_label["IBAN"] == "GB29NWBK60161331926819"


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------

class TestRedact:
    def test_placeholders_numbered_per_label_in_order(self):
        r = phi_redact.redact(
            "Call 415-555-0132 then 212-555-0199 about SSN 234-56-7890.")
        assert r.redacted == \
            "Call [PHONE_1] then [PHONE_2] about SSN [SSN_1]."
        assert r.mapping == {
            "[PHONE_1]": "415-555-0132",
            "[PHONE_2]": "212-555-0199",
            "[SSN_1]": "234-56-7890",
        }
        assert r.counts == {"PHONE": 2, "SSN": 1}
        assert r.engine == "rules"
        assert r.model is None

    def test_no_entities_leaves_text_untouched(self):
        r = phi_redact.redact("nothing to see here")
        assert r.redacted == "nothing to see here"
        assert r.mapping == {}
        assert r.counts == {}
        assert r.entities == []

    def test_original_values_gone_from_redacted(self):
        r = phi_redact.redact("Call 415-555-0132")
        assert "415-555-0132" not in r.redacted

    def test_restore_round_trip(self):
        r = phi_redact.redact(NOTE)
        assert phi_redact.restore(r.redacted, r.mapping) == NOTE

    def test_to_dict_is_json_serializable(self):
        r = phi_redact.redact(NOTE)
        d = json.loads(json.dumps(r.to_dict()))
        assert set(d) == {
            "redacted", "entities", "mapping",
            "counts", "engine", "model",
        }
        assert d["engine"] == "rules"

    def test_result_type(self):
        assert isinstance(phi_redact.redact("hi"), RedactionResult)


# ---------------------------------------------------------------------------
# merge policy
# ---------------------------------------------------------------------------

def _ent(start, end, label, source, score=0.0):
    return {"start": start, "end": end, "label": label,
            "text": "x", "source": source, "score": score}


class TestMerge:
    def test_rules_win_every_overlap(self):
        rule = _ent(5, 17, "PHONE", "rule")
        neural = _ent(5, 10, "NAME_PATIENT", "neural", score=0.99)
        merged = merge_entities([rule], [neural])
        assert [e["label"] for e in merged] == ["PHONE"]

    def test_neural_neural_keeps_best_score(self):
        a = _ent(0, 4, "NAME_PATIENT", "neural", score=0.6)
        b = _ent(0, 4, "NAME_FAMILY", "neural", score=0.9)
        merged = merge_entities([], [a, b])
        assert [e["label"] for e in merged] == ["NAME_FAMILY"]

    def test_disjoint_union_sorted(self):
        rule = _ent(10, 15, "PHONE", "rule")
        neural = _ent(0, 4, "NAME_PATIENT", "neural", score=0.8)
        merged = merge_entities([rule], [neural])
        assert [e["label"] for e in merged] == ["NAME_PATIENT", "PHONE"]

    def test_touching_spans_do_not_overlap(self):
        # end == start is adjacency, not overlap
        rule = _ent(0, 4, "PHONE", "rule")
        neural = _ent(4, 8, "NAME", "neural", score=0.8)
        assert len(merge_entities([rule], [neural])) == 2


class TestPlaceholders:
    def test_interleaved_labels_numbered_independently(self):
        text = "a 415-555-0132 b 234-56-7890 c 212-555-0199"
        ents = rules.scan(text)
        out, mapping = apply_placeholders(text, ents)
        assert out == "a [PHONE_1] b [SSN_1] c [PHONE_2]"
        assert mapping["[PHONE_2]"] == "212-555-0199"

    def test_every_entity_gets_a_placeholder(self):
        ents = rules.scan(NOTE)
        out, mapping = apply_placeholders(NOTE, ents)
        assert len(mapping) == len(ents)
        for e in ents:
            assert "placeholder" in e


# ---------------------------------------------------------------------------
# Redactor (rules-only mode)
# ---------------------------------------------------------------------------

class TestRedactor:
    def test_rules_only_matches_module_functions(self):
        r = Redactor()
        assert r.engine_name == "rules"
        assert r.redact(NOTE).redacted == phi_redact.redact(NOTE).redacted
        assert r.scan(NOTE) == phi_redact.scan(NOTE)

    def test_per_call_threshold_without_neural_is_harmless(self):
        r = Redactor().redact("Call 415-555-0132", threshold=0.9)
        assert r.redacted == "Call [PHONE_1]"

    def test_missing_model_raises(self):
        with pytest.raises(Exception):
            Redactor(model="/nonexistent/model.onnx",
                     tokenizer="/nonexistent/tokenizer.json")

    def test_defaults(self):
        assert DEFAULT_THRESHOLD == 0.6
        assert DEFAULT_MAX_LEN == 384

    def test_version(self):
        assert phi_redact.__version__ == "0.2.0"


class TestPhoneBoundaries:
    """A leading "(" or "+" belongs to the number (issue 2026-10-18)."""

    def test_paren_phone_fully_consumed(self):
        ents = phi_redact.scan("Reach her at (415) 555-0132.")
        assert len(ents) == 1
        assert ents[0]["label"] == "PHONE"
        assert ents[0]["text"] == "(415) 555-0132"

    def test_country_code_fully_consumed(self):
        ents = phi_redact.scan("Call +1 415-555-0132.")
        assert len(ents) == 1
        assert ents[0]["text"] == "+1 415-555-0132"

    def test_redact_leaves_no_stray_chars(self):
        r = phi_redact.redact("Call (415) 555-0132 or +1 212-555-0199.")
        assert r.redacted == "Call [PHONE_1] or [PHONE_2]."
        assert r.mapping == {
            "[PHONE_1]": "(415) 555-0132",
            "[PHONE_2]": "+1 212-555-0199",
        }

    def test_phone_glued_to_word_not_matched(self):
        ents = phi_redact.scan("abc415-555-0132")
        assert not any(e["label"] == "PHONE" for e in ents)

    def test_plain_phone_still_matches(self):
        ents = phi_redact.scan("Call 415-555-0132.")
        assert [e["text"] for e in ents] == ["415-555-0132"]

    def test_paren_glued_to_letter_fully_consumed(self):
        ents = phi_redact.scan("x(312) 555-0167")
        assert len(ents) == 1
        assert ents[0]["text"] == "(312) 555-0167"
        r = phi_redact.redact("x(312) 555-0167")
        assert r.redacted == "x[PHONE_1]"

    def test_paren_glued_to_letter_without_phone_no_match(self):
        assert phi_redact.scan("x(abc)") == []


# ---------------------------------------------------------------------------
# New deterministic rules (2026-10-19): EIN, ABA routing, passport, DL,
# government ID, bank account.
# ---------------------------------------------------------------------------

class TestAbaRoutingOk:
    def test_valid_routing_numbers(self):
        assert rules.aba_routing_ok("021000021") is True   # JPMorgan Chase
        assert rules.aba_routing_ok("494969870") is True   # eval case
        assert rules.aba_routing_ok("336229607") is True
        assert rules.aba_routing_ok("481566888") is True
        assert rules.aba_routing_ok("746787360") is True

    def test_invalid_routing_numbers(self):
        assert rules.aba_routing_ok("123456789") is False  # checksum fails
        assert rules.aba_routing_ok("809096606") is False  # plain SSN, not routing
        assert rules.aba_routing_ok("12345678") is False   # wrong length
        assert rules.aba_routing_ok("1234567890") is False


class TestNewRules:
    def _labels(self, text):
        return {e["label"]: e["text"] for e in phi_redact.scan(text)}

    def test_tax_id_ein(self):
        found = self._labels("EIN 18-6721452 on file.")
        assert found.get("TAX_ID") == "18-6721452"

    def test_tax_id_not_confused_with_ssn_or_date(self):
        ents = phi_redact.scan("SSN 234-56-7890, EIN 18-6721452, DOB 03/14/1978.")
        by_label = {}
        for e in ents:
            by_label.setdefault(e["label"], []).append(e["text"])
        assert by_label["SSN"] == ["234-56-7890"]
        assert by_label["TAX_ID"] == ["18-6721452"]
        assert by_label["DATE"] == ["03/14/1978"]

    def test_routing_number(self):
        found = self._labels("Routing number 494969870 for direct deposit.")
        assert found.get("ROUTING_NUMBER") == "494969870"

    def test_routing_rejects_non_checksum(self):
        ents = phi_redact.scan("Code 123456789 is not a routing number.")
        assert not any(e["label"] == "ROUTING_NUMBER" for e in ents)

    def test_bank_account_context_anchored(self):
        ents = phi_redact.scan("Account 713591432266 at First Bank.")
        assert len(ents) == 1
        assert ents[0]["label"] == "BANK_ACCOUNT"
        assert ents[0]["text"] == "713591432266"

    def test_bank_account_span_excludes_anchor(self):
        text = "Account 713591432266 at First Bank."
        ents = phi_redact.scan(text)
        assert text[ents[0]["start"]:ents[0]["end"]] == "713591432266"

    def test_bank_account_needs_anchor(self):
        ents = phi_redact.scan("Number 713591432266 appears alone.")
        assert not any(e["label"] == "BANK_ACCOUNT" for e in ents)

    def test_passport(self):
        found = self._labels("Passport Y12973198 verified.")
        assert found.get("PASSPORT") == "Y12973198"

    def test_drivers_license(self):
        found = self._labels("DL D1315565 and CA-L123456 on file.")
        texts = [e["text"] for e in phi_redact.scan("DL D1315565 and CA-L123456 on file.")
                 if e["label"] == "DRIVERS_LICENSE"]
        assert texts == ["D1315565", "CA-L123456"]

    def test_government_id(self):
        ents = phi_redact.scan("USCIS LIN2190012345; alien A123456789.")
        by_label = {}
        for e in ents:
            by_label.setdefault(e["label"], []).append(e["text"])
        assert by_label["GOVERNMENT_ID"] == ["LIN2190012345", "A123456789"]

    def test_new_rules_redact_round_trip(self):
        text = "EIN 18-6721452, routing 494969870, passport Y12973198."
        r = phi_redact.redact(text)
        assert r.redacted == ("EIN [TAX_ID_1], routing [ROUTING_NUMBER_1], "
                              "passport [PASSPORT_1].")
        assert r.mapping == {
            "[TAX_ID_1]": "18-6721452",
            "[ROUTING_NUMBER_1]": "494969870",
            "[PASSPORT_1]": "Y12973198",
        }
        assert phi_redact.restore(r.redacted, r.mapping) == text

    def test_bank_account_anchor_variants(self):
        for text, expected in [
            ("Acct #987654321098 for payroll.", "987654321098"),
            ("ACCT-112233445566 on file.", "112233445566"),
            ("Account number 112233445566 on file.", "112233445566"),
        ]:
            ents = [e for e in phi_redact.scan(text) if e["label"] == "BANK_ACCOUNT"]
            assert len(ents) == 1, text
            assert ents[0]["text"] == expected, text
            assert text[ents[0]["start"]:ents[0]["end"]] == expected, text
