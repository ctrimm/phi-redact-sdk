"""Tests for the 0.5.0 features (Python SDK).

Custom recognizers (PatternRecognizer) and per-label thresholds with
needs-review mode. Mirrors typescript/test/recognizers.test.js.
Synthetic fixtures only.
"""

import pytest

import phi_redact
from phi_redact import PatternRecognizer, Redactor
from phi_redact.engine import (
    OnnxEngine,
    merge_entities,
    recognizer_spans,
)
from phi_redact.engine import redact as engine_redact

BADGE = PatternRecognizer(label="EMPLOYEE_ID", patterns=[r"EMP\d{6}"])
BADGE_TEXT = "Badge EMP123456 issued, call 415-555-0132."


# ---------------------------------------------------------------------------
# Custom recognizers
# ---------------------------------------------------------------------------

class TestRecognizerBasics:
    def test_custom_label_redacted_with_recognizer_source(self):
        r = phi_redact.redact(BADGE_TEXT, recognizers=[BADGE])
        assert r.redacted == "Badge [EMPLOYEE_ID_1] issued, call [PHONE_1]."
        (emp,) = [e for e in r.entities if e["label"] == "EMPLOYEE_ID"]
        assert emp["source"] == "recognizer"
        assert emp["score"] == 0.85  # PatternRecognizer default
        assert emp["text"] == "EMP123456"
        assert phi_redact.restore(r.redacted, r.mapping) == BADGE_TEXT

    def test_multiple_patterns_one_recognizer(self):
        rec = PatternRecognizer(
            label="EMPLOYEE_ID", patterns=[r"EMP\d{6}", r"STF-\d{4}"])
        r = phi_redact.redact("EMP123456 and STF-7788", recognizers=[rec])
        assert r.redacted == "[EMPLOYEE_ID_1] and [EMPLOYEE_ID_2]"

    def test_redactor_constructor_and_scan(self):
        r = Redactor(recognizers=[BADGE])
        assert [e["label"] for e in r.scan(BADGE_TEXT)] == \
            ["EMPLOYEE_ID", "PHONE"]
        assert r.redact(BADGE_TEXT).redacted == \
            "Badge [EMPLOYEE_ID_1] issued, call [PHONE_1]."

    def test_no_recognizers_no_change(self):
        r = phi_redact.redact(BADGE_TEXT)
        assert r.redacted == "Badge EMP123456 issued, call [PHONE_1]."

    def test_group_selects_value_span(self):
        rec = PatternRecognizer(label="BADGE_ID", group=1,
                                patterns=[r"(?:badge|Badge)\s+(\d{6})"])
        r = phi_redact.redact("badge 482913 verified", recognizers=[rec])
        assert r.redacted == "badge [BADGE_ID_1] verified"
        assert r.entities[0]["text"] == "482913"

    def test_first_registered_recognizer_wins_overlap(self):
        a = PatternRecognizer(label="SIX_DIGITS", patterns=[r"\b\d{6}\b"],
                              score=0.9)
        b = PatternRecognizer(label="EXACT_CODE", patterns=[r"482913"],
                              score=0.99)
        r = phi_redact.redact("code 482913 here", recognizers=[a, b])
        assert r.redacted == "code [SIX_DIGITS_1] here"
        r = phi_redact.redact("code 482913 here", recognizers=[b, a])
        assert r.redacted == "code [EXACT_CODE_1] here"

    def test_empty_matches_are_skipped(self):
        rec = PatternRecognizer(label="DIGITS", patterns=[r"\d*"],
                                score=0.9)
        r = phi_redact.redact("abc 123 def", recognizers=[rec])
        assert r.redacted == "abc [DIGITS_1] def"


class TestRecognizerOperators:
    def test_label_operator(self):
        r = phi_redact.redact(BADGE_TEXT, recognizers=[BADGE],
                              operator="label")
        assert r.redacted == "Badge <EMPLOYEE_ID> issued, call <PHONE>."
        assert r.mapping == {}

    def test_hash_operator_consistent_for_recognizer_spans(self):
        a = phi_redact.redact("EMP123456", recognizers=[BADGE],
                              operator="hash", salt="s")
        b = phi_redact.redact("again EMP123456", recognizers=[BADGE],
                              operator="hash", salt="s")
        assert a.redacted == b.redacted.split()[-1]
        assert a.redacted.startswith("[EMPLOYEE_ID_")

    def test_remove_operator(self):
        r = phi_redact.redact("EMP123456 gone", recognizers=[BADGE],
                              operator="remove")
        assert r.redacted == " gone"


class TestRecognizerContext:
    REC = PatternRecognizer(label="CASE_ID", patterns=[r"\b\d{6}\b"],
                            score=0.5, context=["case"],
                            context_boost=0.4)

    def test_context_boost_raises_score_over_threshold(self):
        r = phi_redact.redact("case 482913 open", recognizers=[self.REC])
        assert r.redacted == "case [CASE_ID_1] open"
        assert r.entities[0]["score"] == 0.9

    def test_no_context_stays_below_global_threshold(self):
        r = phi_redact.redact("number 482913 open", recognizers=[self.REC])
        assert r.redacted == "number 482913 open"
        assert r.entities == [] and r.review == []

    def test_context_outside_window_does_not_boost(self):
        rec = PatternRecognizer(label="CASE_ID", patterns=[r"\b\d{6}\b"],
                                score=0.5, context=["case"],
                                context_boost=0.4, context_window=5)
        r = phi_redact.redact("case number is far away: 482913",
                              recognizers=[rec])
        assert r.entities == []

    def test_boost_caps_at_one(self):
        rec = PatternRecognizer(label="CASE_ID", patterns=[r"\b\d{6}\b"],
                                score=0.95, context=["case"],
                                context_boost=0.4)
        (span,) = recognizer_spans("case 482913", [rec])
        assert span["score"] == 1.0

    def test_require_context_drops_uncontextualized_match(self):
        rec = PatternRecognizer(label="CASE_ID", patterns=[r"\b\d{6}\b"],
                                score=0.99, context=["case"],
                                require_context=True)
        r = phi_redact.redact("number 482913 open", recognizers=[rec])
        assert r.entities == []
        r = phi_redact.redact("case 482913 open", recognizers=[rec])
        assert r.redacted == "case [CASE_ID_1] open"


class TestRecognizerValidators:
    def test_callable_validator(self):
        rec = PatternRecognizer(
            label="ACCT_CODE", patterns=[r"ACCT-[A-Z]{2}\d{4}"],
            validator=lambda v: v.endswith("99"))
        r = phi_redact.redact("ACCT-XY1299 and ACCT-XY1234",
                              recognizers=[rec])
        assert r.redacted == "[ACCT_CODE_1] and ACCT-XY1234"

    def test_named_luhn_validator(self):
        rec = PatternRecognizer(label="CARD_LIKE", patterns=[r"\b\d{16}\b"],
                                validator="luhn")
        spans = recognizer_spans(
            "4111111111111111 vs 1234567812345678", [rec])
        assert [s["text"] for s in spans] == ["4111111111111111"]

    def test_named_iban_validator_accepts_spaced_iban(self):
        # The rules layer's IBAN pattern needs a contiguous token; the
        # spaced form is a recognizer + validator job.
        rec = PatternRecognizer(
            label="IBAN", validator="iban",
            patterns=[r"[A-Z]{2}\d{2}(?: \d{4}){4} \d{2}"])
        good = phi_redact.redact("IBAN DE89 3704 0044 0532 0130 00 end",
                                 recognizers=[rec])
        assert good.redacted == "IBAN [IBAN_1] end"
        bad = phi_redact.redact("IBAN DE89 3704 0044 0532 0130 01 end",
                                recognizers=[rec])
        # checksum fails -> the recognizer produces no IBAN span
        assert [e for e in bad.entities if e["label"] == "IBAN"] == []

    def test_invalid_configs_raise(self):
        with pytest.raises(ValueError):
            PatternRecognizer(label="", patterns=[r"x"])
        with pytest.raises(ValueError):
            PatternRecognizer(label="X", patterns=[])
        with pytest.raises(ValueError):
            PatternRecognizer(label="X", patterns=[r"x"], score=1.5)
        with pytest.raises(ValueError):
            PatternRecognizer(label="X", patterns=[r"x"],
                              require_context=True)
        with pytest.raises(ValueError):
            PatternRecognizer(label="X", patterns=[r"x"],
                              validator="nope")
        with pytest.raises(ValueError):
            PatternRecognizer(label="X", patterns=[r"x"], group=-1)


class TestRecognizerPrecedence:
    def test_recognizer_yields_to_overlapping_rule_span(self):
        # The ZIP rule claims "12345" inside "EMP-12345"; the
        # recognizer span overlaps it and is dropped (rules first).
        rec = PatternRecognizer(label="EMPLOYEE_ID",
                                patterns=[r"EMP-\d{5}"], score=0.95)
        r = phi_redact.redact("Badge EMP-12345 issued.", recognizers=[rec])
        assert r.redacted == "Badge EMP-[ZIP_1] issued."

    def test_merge_recognizer_beats_neural_rule_beats_recognizer(self):
        rule = {"start": 0, "end": 5, "label": "ZIP", "text": "90210",
                "source": "rule"}
        rec = {"start": 3, "end": 9, "label": "EMPLOYEE_ID",
               "text": "123456", "source": "recognizer", "score": 0.9}
        neural = {"start": 3, "end": 9, "label": "NAME",
                  "text": "123456", "source": "neural", "score": 0.99}
        merged = merge_entities([rule], [neural], [rec])
        assert [e["label"] for e in merged] == ["ZIP"]
        merged = merge_entities([], [neural], [rec])
        assert [e["label"] for e in merged] == ["EMPLOYEE_ID"]

    def test_allowlist_drops_recognizer_span(self):
        r = phi_redact.redact(BADGE_TEXT, recognizers=[BADGE],
                              allowlist=["EMP123456"])
        assert r.redacted == "Badge EMP123456 issued, call [PHONE_1]."

    def test_denylist_wins_overlap_with_recognizer(self):
        r = phi_redact.redact("id EMP123456 ok", recognizers=[BADGE],
                              denylist=["EMP123456"])
        assert r.redacted == "id [CUSTOM_1] ok"
        assert r.entities[0]["source"] == "denylist"


# ---------------------------------------------------------------------------
# Per-label thresholds + needs-review
# ---------------------------------------------------------------------------

REF = PatternRecognizer(label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"],
                        score=0.8)
REF_TEXT = "ref ZX-8842 on file"


class TestThresholds:
    def test_per_label_threshold_lowers_gate(self):
        low = PatternRecognizer(label="CASE_ID", patterns=[r"\b\d{6}\b"],
                                score=0.5)
        r = phi_redact.redact("code 482913 here", recognizers=[low],
                              thresholds={"CASE_ID": 0.4})
        assert r.redacted == "code [CASE_ID_1] here"

    def test_per_label_threshold_raises_gate_and_drops(self):
        r = phi_redact.redact(REF_TEXT, recognizers=[REF],
                              thresholds={"REF_CODE": 0.95})
        assert r.redacted == REF_TEXT
        assert r.entities == [] and r.review == []

    def test_score_exactly_at_threshold_redacts(self):
        r = phi_redact.redact(REF_TEXT, recognizers=[REF],
                              thresholds={"REF_CODE": 0.8})
        assert r.redacted == "ref [REF_CODE_1] on file"

    def test_other_labels_unaffected_by_override(self):
        r = phi_redact.redact("call 415-555-0132, ref ZX-8842",
                              recognizers=[REF],
                              thresholds={"REF_CODE": 0.95})
        assert r.redacted == "call [PHONE_1], ref ZX-8842"

    def test_invalid_threshold_values_raise(self):
        with pytest.raises(ValueError):
            phi_redact.redact("x", thresholds={"X": 1.5})
        with pytest.raises(ValueError):
            phi_redact.redact("x", review_margin=-0.1)
        with pytest.raises(ValueError):
            Redactor(thresholds={"X": -0.5})


class TestReviewMode:
    def test_span_in_band_becomes_review_candidate(self):
        r = phi_redact.redact(REF_TEXT, recognizers=[REF],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.2)
        assert r.redacted == REF_TEXT          # NOT redacted
        assert r.entities == [] and r.counts == {} and r.mapping == {}
        assert r.review == [{
            "start": 4, "end": 11, "label": "REF_CODE", "text": "ZX-8842",
            "score": 0.8, "source": "recognizer",
        }]

    def test_span_below_floor_is_dropped_not_reviewed(self):
        low = PatternRecognizer(label="REF_CODE",
                                patterns=[r"[A-Z]{2}-\d{4}"], score=0.5)
        r = phi_redact.redact(REF_TEXT, recognizers=[low],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.2)
        assert r.review == [] and r.entities == []

    def test_span_exactly_at_floor_is_reviewed(self):
        r = phi_redact.redact(REF_TEXT, recognizers=[REF],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.1)
        assert len(r.review) == 1

    def test_global_margin_without_per_label_thresholds(self):
        low = PatternRecognizer(label="REF_CODE",
                                patterns=[r"[A-Z]{2}-\d{4}"], score=0.55)
        r = phi_redact.redact(REF_TEXT, recognizers=[low],
                              review_margin=0.1)
        assert len(r.review) == 1
        r = phi_redact.redact(REF_TEXT, recognizers=[low],
                              review_margin=0.04)
        assert r.review == [] and r.entities == []

    def test_zero_margin_reviews_nothing(self):
        r = phi_redact.redact(REF_TEXT, recognizers=[REF],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.0)
        assert r.review == [] and r.entities == []

    def test_review_candidates_sorted_by_start(self):
        r = phi_redact.redact("QW-1093 then ZX-8842", recognizers=[REF],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.2)
        assert [c["text"] for c in r.review] == ["QW-1093", "ZX-8842"]

    def test_restore_ignores_review_candidates(self):
        text = "call 415-555-0132, ref ZX-8842"
        r = phi_redact.redact(text, recognizers=[REF],
                              thresholds={"REF_CODE": 0.9},
                              review_margin=0.2)
        assert len(r.review) == 1
        assert phi_redact.restore(r.redacted, r.mapping) == text

    def test_rules_spans_are_never_gated(self):
        # Documented edge: rule spans carry no score, so a per-label
        # threshold on SSN does not stop the checksum-gated SSN rule,
        # and the span never becomes a review candidate.
        r = phi_redact.redact("SSN 234-56-7890",
                              thresholds={"SSN": 0.99}, review_margin=0.5)
        assert r.redacted == "SSN [SSN_1]"
        assert r.review == []

    def test_denylist_spans_are_never_gated(self):
        r = phi_redact.redact("Codral", denylist=["Codral"],
                              thresholds={"CUSTOM": 0.99},
                              review_margin=0.5)
        assert r.redacted == "[CUSTOM_1]"
        assert r.review == []

    def test_batch_results_carry_review(self):
        r = Redactor(recognizers=[REF], thresholds={"REF_CODE": 0.9},
                     review_margin=0.2)
        batch = r.redact_batch([REF_TEXT, "nothing here"])
        assert len(batch[0].review) == 1
        assert batch[1].review == []
        assert batch[0].redacted == REF_TEXT


class _FakeNeural(OnnxEngine):
    """Scripted neural engine: returns fixed spans, filtering at the
    threshold it is handed exactly like the real BIOES decode does."""

    def __init__(self, spans, threshold=0.6):
        self._spans = spans
        self.threshold = threshold
        self.model_path = "fake.onnx"
        self.seen_gate = None

    def predict(self, text, threshold=None):
        gate = self.threshold if threshold is None else threshold
        self.seen_gate = gate
        return [dict(e) for e in self._spans if e["score"] >= gate]


def _name_span(text, name, score):
    s = text.index(name)
    return {"start": s, "end": s + len(name), "label": "NAME",
            "text": name, "source": "neural", "score": score}


class TestNeuralThresholdsAndReview:
    TEXT = "Dr. Alice Ferreira reviewed Ana Souza and Bob Lindqvist."

    def _engine(self, **kw):
        spans = [
            _name_span(self.TEXT, "Alice Ferreira", 0.95),
            _name_span(self.TEXT, "Ana Souza", 0.75),
            _name_span(self.TEXT, "Bob Lindqvist", 0.45),
        ]
        return _FakeNeural(spans, **kw)

    def test_neural_spans_partitioned_by_label_threshold(self):
        eng = self._engine()
        r = engine_redact(self.TEXT, eng, thresholds={"NAME": 0.8},
                          review_margin=0.3)
        # 0.95 redacts, 0.75 lands in the review band [0.5, 0.8),
        # 0.45 is below the floor and drops.
        assert "[NAME_1]" in r.redacted
        assert "Ana Souza" in r.redacted  # left in the clear for review
        assert [c["text"] for c in r.review] == ["Ana Souza"]
        assert "Bob Lindqvist" in r.redacted
        assert all(c["text"] != "Bob Lindqvist" for c in r.review)
        # the engine was asked for spans down to the lowest review
        # floor: min(global 0.6, NAME 0.8) - margin 0.3 = 0.3 (labels
        # without an override review against the global threshold)
        assert eng.seen_gate == pytest.approx(0.3)

    def test_recognizer_beats_neural_in_full_pipeline(self):
        text = "Alice Ferreira called."
        eng = _FakeNeural([_name_span(text, "Alice Ferreira", 0.99)])
        rec = PatternRecognizer(label="STAFF", patterns=[r"Alice"],
                                score=0.9)
        r = engine_redact(text, eng, recognizers=[rec])
        assert r.redacted == "[STAFF_1] Ferreira called."

    def test_engine_default_gate_unchanged_without_options(self):
        eng = self._engine()
        r = engine_redact(self.TEXT, eng)
        assert eng.seen_gate == 0.6
        assert [e["text"] for e in r.entities] == \
            ["Alice Ferreira", "Ana Souza"]
        assert r.review == []
