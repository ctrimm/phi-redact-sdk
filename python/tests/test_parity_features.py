"""Tests for the 0.4.0 Presidio-parity features (Python SDK).

Batch redaction, scrub/restore (incl. ChunkedRestorer), anonymize
operators, and allow/deny lists. The hash test vectors at the top of
this file are SHARED with the TypeScript suite
(typescript/test/features.test.js) — they are what keeps the two
SDKs' "hash" operator byte-identical. Synthetic fixtures only.
"""

import json
import os
import subprocess
import sys

import pytest

import phi_redact
from phi_redact import ChunkedRestorer, Redactor
from phi_redact.engine import hash_token

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "..", "examples", "batch_jsonl.py")

# (salt, label, value, expected token) — pinned in BOTH suites.
HASH_VECTORS = [
    ("", "SSN", "234-56-7890", "[SSN_5f6c8b16]"),
    ("pepper", "SSN", "234-56-7890", "[SSN_e2e9f611]"),
    ("", "EMAIL", "jane@example.com", "[EMAIL_34da503d]"),
    ("s3cret-salt", "NAME_PATIENT", "Maria Santos",
     "[NAME_PATIENT_09a6b457]"),
    ("parity-salt", "PHONE", "415-555-0132", "[PHONE_954b7ae0]"),
]

NOTE = "Call 415-555-0132 about SSN 234-56-7890."


# ---------------------------------------------------------------------------
# Anonymize operators
# ---------------------------------------------------------------------------

class TestHashVectors:
    @pytest.mark.parametrize("salt,label,value,expected", HASH_VECTORS)
    def test_shared_vectors(self, salt, label, value, expected):
        assert hash_token(label, value, salt) == expected

    def test_hash_via_redact(self):
        r = phi_redact.redact("SSN 234-56-7890", operator="hash",
                              salt="pepper")
        assert r.redacted == "SSN [SSN_e2e9f611]"
        assert r.mapping == {}  # hash is irreversible by design

    def test_hash_consistent_across_documents(self):
        a = phi_redact.redact("SSN 234-56-7890", operator="hash", salt="s")
        b = phi_redact.redact("again 234-56-7890 here", operator="hash",
                              salt="s")
        assert a.redacted.split()[-1] == b.redacted.split()[1]

    def test_hash_salt_changes_token(self):
        a = phi_redact.redact("SSN 234-56-7890", operator="hash", salt="a")
        b = phi_redact.redact("SSN 234-56-7890", operator="hash", salt="b")
        assert a.redacted != b.redacted


class TestOperators:
    def test_label_operator_presidio_style(self):
        r = phi_redact.redact(NOTE, operator="label")
        assert r.redacted == "Call <PHONE> about SSN <SSN>."
        assert r.mapping == {}  # irreversible by design
        assert r.counts == {"PHONE": 1, "SSN": 1}
        assert r.entities[0]["placeholder"] == "<PHONE>"

    def test_mask_structured_ids_keep_last_four(self):
        r = phi_redact.redact(
            "SSN 234-56-7890, card 4111111111111111, ph (415) 555-0132",
            operator="mask")
        assert r.redacted == ("SSN ***-**-7890, card ************1111, "
                              "ph (***) ***-0132")
        assert r.mapping == {}

    def test_mask_other_values_same_length_asterisks(self):
        r = phi_redact.redact("ZIP 90210, see https://example.com/x",
                              operator="mask")
        assert r.redacted == "ZIP *****, see *********************"

    def test_remove_operator_deletes_spans(self):
        r = phi_redact.redact("Call 415-555-0132 now", operator="remove")
        assert r.redacted == "Call  now"
        assert r.mapping == {}
        assert r.entities[0]["placeholder"] == ""

    def test_unknown_operator_rejected(self):
        with pytest.raises(ValueError):
            phi_redact.redact(NOTE, operator="bogus")
        with pytest.raises(ValueError):
            Redactor(operator="bogus")

    def test_redactor_operator_option(self):
        r = Redactor(operator="label").redact(NOTE)
        assert r.redacted == "Call <PHONE> about SSN <SSN>."


# ---------------------------------------------------------------------------
# Allow / deny lists
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_allowlisted_value_not_redacted(self):
        r = phi_redact.redact(NOTE, allowlist=["415-555-0132"])
        assert r.redacted == "Call 415-555-0132 about SSN [SSN_1]."
        assert r.counts == {"SSN": 1}

    def test_allowlist_is_case_insensitive(self):
        r = phi_redact.redact("mail JANE@EXAMPLE.COM",
                              allowlist=["jane@example.com"])
        assert r.redacted == "mail JANE@EXAMPLE.COM"

    def test_redactor_allowlist_applies_to_scan(self):
        r = Redactor(allowlist=["415-555-0132"])
        labels = [e["label"] for e in r.scan(NOTE)]
        assert labels == ["SSN"]


class TestDenylist:
    def test_denylist_term_redacted_as_custom(self):
        r = phi_redact.redact("Patient takes Codral daily.",
                              denylist=["Codral"])
        assert r.redacted == "Patient takes [CUSTOM_1] daily."
        (e,) = r.entities
        assert e["label"] == "CUSTOM"
        assert e["source"] == "denylist"
        assert phi_redact.restore(r.redacted, r.mapping) == \
            "Patient takes Codral daily."

    def test_deny_label_option(self):
        r = phi_redact.redact("Patient takes Codral daily.",
                              denylist=["Codral"],
                              deny_label="NAME_PATIENT")
        assert r.redacted == "Patient takes [NAME_PATIENT_1] daily."

    def test_denylist_case_insensitive_word_boundaries(self):
        r = phi_redact.redact("codral and CODRAL, not Codraline.",
                              denylist=["Codral"])
        assert r.redacted == "[CUSTOM_1] and [CUSTOM_2], not Codraline."

    def test_denylist_span_wins_overlap_with_detected(self):
        # The deny match sits inside the MRN rule span; the rule span is
        # dropped and only the denied digits are hidden.
        r = phi_redact.redact("MRN 48291375", denylist=["48291375"])
        assert r.redacted == "MRN [CUSTOM_1]"

    def test_deny_wins_over_allow(self):
        r = phi_redact.redact("Codral", allowlist=["Codral"],
                              denylist=["Codral"])
        assert r.redacted == "[CUSTOM_1]"

    def test_denylist_no_match_is_noop(self):
        r = phi_redact.redact(NOTE, denylist=["Zebra"])
        assert r.redacted == phi_redact.redact(NOTE).redacted


# ---------------------------------------------------------------------------
# Scrub / restore + ChunkedRestorer
# ---------------------------------------------------------------------------

class TestScrubRestore:
    def test_scrub_returns_redacted_and_mapping(self):
        redacted, mapping = Redactor().scrub(NOTE)
        base = phi_redact.redact(NOTE)
        assert redacted == base.redacted
        assert mapping == base.mapping
        assert Redactor.restore(redacted, mapping) == NOTE

    def test_scrub_with_label_operator_has_empty_mapping(self):
        redacted, mapping = Redactor(operator="label").scrub(NOTE)
        assert redacted == "Call <PHONE> about SSN <SSN>."
        assert mapping == {}

    def test_restore_many_same_label_placeholders(self):
        phones = " ".join(f"415-555-01{i:02d}" for i in range(12))
        r = phi_redact.redact(phones)
        assert r.counts["PHONE"] == 12
        assert phi_redact.restore(r.redacted, r.mapping) == phones


class TestChunkedRestorer:
    def _mapping(self):
        return phi_redact.redact(NOTE).mapping

    def test_char_by_char_round_trip(self):
        original = "The number is 415-555-0132 and SSN 234-56-7890."
        redacted, mapping = Redactor().scrub(original)
        cr = ChunkedRestorer(mapping)
        out = "".join(cr.feed(c) for c in redacted) + cr.flush()
        assert out == original

    def test_every_split_point(self):
        original = "Call 415-555-0132."
        redacted, mapping = Redactor().scrub(original)
        for i in range(len(redacted) + 1):
            cr = ChunkedRestorer(mapping)
            out = cr.feed(redacted[:i]) + cr.feed(redacted[i:]) + cr.flush()
            assert out == original, f"split at {i}"

    def test_unknown_placeholder_untouched(self):
        cr = ChunkedRestorer(self._mapping())
        out = cr.feed("ref [FOO_9] and [PHONE_") + cr.feed("1] done")
        out += cr.flush()
        assert out == "ref [FOO_9] and 415-555-0132 done"

    def test_partial_placeholder_at_flush_untouched(self):
        cr = ChunkedRestorer(self._mapping())
        out = cr.feed("ends with [PHO") + cr.flush()
        assert out == "ends with [PHO"

    def test_empty_mapping_passes_text_through(self):
        cr = ChunkedRestorer({})
        assert cr.feed("plain [TEXT_1] here") == "plain [TEXT_1] here"
        assert cr.flush() == ""


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    TEXTS = [NOTE, "nothing to see here", "SSN 234-56-7890"]

    def test_redact_batch_preserves_order_and_matches_single(self):
        r = Redactor()
        batch = r.redact_batch(self.TEXTS)
        assert [b.redacted for b in batch] == \
            [r.redact(t).redacted for t in self.TEXTS]
        assert batch[1].entities == []

    def test_redact_batch_empty(self):
        assert Redactor().redact_batch([]) == []

    def test_redact_iter_is_lazy(self):
        consumed = []

        def gen():
            for t in self.TEXTS:
                consumed.append(t)
                yield t

        it = Redactor().redact_iter(gen())
        first = next(it)
        assert first.redacted == Redactor().redact(self.TEXTS[0]).redacted
        assert consumed == [self.TEXTS[0]]  # nothing else pulled yet
        assert [x.redacted for x in it] == \
            [Redactor().redact(t).redacted for t in self.TEXTS[1:]]


class TestBatchJsonlCli:
    def _run(self, stdin_text, *extra):
        return subprocess.run(
            [sys.executable, CLI, *extra], input=stdin_text,
            capture_output=True, text=True, timeout=60)

    def test_stdin_to_stdout(self):
        inp = ('{"text": "Call 415-555-0132"}\n'
               '{"text": "clean"}\n'
               '{"id": 7}\n')
        proc = self._run(inp)
        assert proc.returncode == 0, proc.stderr
        rows = [json.loads(l) for l in proc.stdout.splitlines()]
        assert rows[0]["text"] == "Call [PHONE_1]"
        assert rows[0]["counts"] == {"PHONE": 1}
        assert rows[1] == {"text": "clean", "counts": {}}
        assert rows[2] == {"id": 7}  # no text field: passes through

    def test_file_args_and_operator(self, tmp_path):
        src = tmp_path / "in.jsonl"
        src.write_text('{"note": "SSN 234-56-7890", "text": "x"}\n')
        dst = tmp_path / "out.jsonl"
        proc = subprocess.run(
            [sys.executable, CLI, str(src), str(dst), "--field", "note",
             "--operator", "label"],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        (row,) = [json.loads(l) for l in dst.read_text().splitlines()]
        assert row["note"] == "SSN <SSN>"
        assert row["counts"] == {"SSN": 1}
