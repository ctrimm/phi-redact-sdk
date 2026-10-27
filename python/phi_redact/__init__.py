"""phi-redact Python SDK.

Two layers, one call::

    import phi_redact

    result = phi_redact.redact("Call Jane Doe at 415-555-0132")
    result.redacted   # 'Call Jane Doe at [PHONE_1]'
    result.entities   # [{'start':.., 'end':.., 'label':'PHONE', ...}]
    result.mapping    # {'[PHONE_1]': '415-555-0132'}
    phi_redact.restore(result.redacted, result.mapping)  # original back

The module-level functions are rules-only (no model needed). For the
neural layer (names, addresses, free-text PHI), use Redactor::

    r = phi_redact.Redactor(model="model.onnx", tokenizer="tokenizer.json")
    result = r.redact(text, threshold=0.6)

Merge policy: precedence rules > recognizers > neural — rule spans win
every overlap, recognizer spans beat neural spans but yield to rules,
neural spans that overlap a kept span are dropped; placeholders are
numbered per label in document order. Keep the mapping for
pseudonymized round-trips; delete it and the copy is anonymized.

0.4.0 adds Presidio-parity SDK features (English text only): batch
redaction (Redactor.redact_batch / redact_iter), a first-class
scrub/restore pair for scrubbing prompts before a cloud LLM call —
including ChunkedRestorer for streaming LLM output — anonymize
operators ("numbered", "label", "mask", "hash", "remove"), and
allow/deny lists. Operators and lists are Redactor constructor options
(or keyword arguments to the module-level redact()).

0.5.0 adds custom recognizers (PatternRecognizer: regex patterns with
base scores, context-word boosting, optional checksum/callable
validators, custom labels) and per-label thresholds with a
needs-review mode (review_margin): scored spans that miss their
threshold but land within the margin come back on result.review for a
human to decide instead of being silently dropped.

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.
"""

from __future__ import annotations

from redact import rules

from .engine import (
    DEFAULT_DENY_LABEL,
    DEFAULT_MAX_LEN,
    DEFAULT_THRESHOLD,
    OPERATORS,
    ChunkedRestorer,
    OnnxEngine,
    PatternRecognizer,
    RedactionResult,
    _validate_score_options,
    apply_placeholders,
    gather_entities,
    merge_entities,
    process_entities,
    recognizer_spans,
    redact as _engine_redact,
)

__version__ = "0.5.0"

__all__ = [
    "ChunkedRestorer",
    "DEFAULT_DENY_LABEL",
    "DEFAULT_MAX_LEN",
    "DEFAULT_THRESHOLD",
    "OPERATORS",
    "OnnxEngine",
    "PatternRecognizer",
    "RedactionResult",
    "Redactor",
    "__version__",
    "apply_placeholders",
    "merge_entities",
    "process_entities",
    "recognizer_spans",
    "redact",
    "restore",
    "scan",
]


def scan(text: str) -> list[dict]:
    """Rules-only entity scan. Returns entity dicts sorted by start."""
    return rules.scan(text)


def redact(text: str, *, operator: str = "numbered", salt: str = "",
           allowlist: list[str] | None = None,
           denylist: list[str] | None = None,
           deny_label: str = DEFAULT_DENY_LABEL,
           recognizers: list[PatternRecognizer] | None = None,
           thresholds: dict | None = None,
           review_margin: float | None = None) -> RedactionResult:
    """Rules-only redaction. No model needed.

    Accepts the same operator / salt / allowlist / denylist /
    recognizers / thresholds / review_margin options as Redactor for
    one-off calls. Scored spans (recognizer spans — the rules layer
    carries no scores) are gated at DEFAULT_THRESHOLD unless a
    per-label threshold overrides it.
    """
    return _engine_redact(text, operator=operator, salt=salt,
                          allowlist=allowlist, denylist=denylist,
                          deny_label=deny_label, recognizers=recognizers,
                          thresholds=thresholds,
                          review_margin=review_margin)


def restore(redacted: str, mapping: dict) -> str:
    """Put original values back using a mapping from redact()."""
    return rules.restore_text(redacted, mapping)


class Redactor:
    """Stateful redactor with optional ONNX neural layer.

    Without model/tokenizer this is rules-only, same as the module
    functions. With them, neural spans (names, addresses, free-text PHI)
    are merged under the rules > recognizers > neural policy.

    Options (all keyword): operator (placeholder style, see
    phi_redact.engine.OPERATORS; only "numbered" is reversible), salt
    (for the "hash" operator), allowlist (terms never redacted),
    denylist (terms always redacted), deny_label (label for denylist
    spans, default "CUSTOM"), recognizers (list of PatternRecognizer),
    thresholds (per-label score gates overriding the global threshold
    for scored spans), review_margin (needs-review band width below
    each label's threshold; None disables review mode).
    """

    def __init__(self, model: str | None = None,
                 tokenizer: str | None = None,
                 labels: str | None = None,
                 max_len: int = DEFAULT_MAX_LEN,
                 threshold: float = DEFAULT_THRESHOLD,
                 operator: str = "numbered",
                 salt: str = "",
                 allowlist: list[str] | None = None,
                 denylist: list[str] | None = None,
                 deny_label: str = DEFAULT_DENY_LABEL,
                 recognizers: list[PatternRecognizer] | None = None,
                 thresholds: dict | None = None,
                 review_margin: float | None = None):
        if operator not in OPERATORS:
            raise ValueError(
                f"unknown operator {operator!r}; expected one of {OPERATORS}")
        _validate_score_options(thresholds, review_margin)
        self.operator = operator
        self.salt = salt
        self.allowlist = list(allowlist) if allowlist else []
        self.denylist = list(denylist) if denylist else []
        self.deny_label = deny_label
        self.recognizers = list(recognizers) if recognizers else []
        self.thresholds = dict(thresholds) if thresholds else None
        self.review_margin = review_margin
        self.threshold = float(threshold)
        self.neural = (OnnxEngine(model, tokenizer, labels, max_len,
                                  threshold)
                       if model is not None else None)

    @property
    def engine_name(self) -> str:
        return "rules+onnx" if self.neural is not None else "rules"

    def scan(self, text: str) -> list[dict]:
        """Entities that would be redacted (rules + recognizers +
        neural when configured), with this Redactor's allow/deny lists,
        recognizers, and score gate applied. Review candidates are a
        redact()-time concept and are not returned here."""
        kept, _review = gather_entities(
            text, self.neural, recognizers=self.recognizers,
            allowlist=self.allowlist, denylist=self.denylist,
            deny_label=self.deny_label, thresholds=self.thresholds,
            review_margin=self.review_margin,
            global_threshold=self.threshold)
        return kept

    def redact(self, text: str,
               threshold: float | None = None) -> RedactionResult:
        """Redact text; optional per-call global threshold override
        (sticky: it becomes this Redactor's threshold, as before)."""
        if threshold is not None:
            self.threshold = float(threshold)
            if self.neural is not None:
                self.neural.threshold = float(threshold)
        return _engine_redact(text, self.neural, operator=self.operator,
                              salt=self.salt, allowlist=self.allowlist,
                              denylist=self.denylist,
                              deny_label=self.deny_label,
                              recognizers=self.recognizers,
                              thresholds=self.thresholds,
                              review_margin=self.review_margin,
                              global_threshold=self.threshold)

    def redact_batch(self, texts, threshold: float | None = None):
        """Redact many texts with this one loaded instance.

        Returns one RedactionResult per input, in input order — the
        batch-pipeline entry point (one model load, N documents).
        """
        return [self.redact(t, threshold=threshold) for t in texts]

    def redact_iter(self, texts, threshold: float | None = None):
        """Generator variant of redact_batch: yields results one at a
        time, so arbitrarily large inputs stream without holding every
        result in memory."""
        for t in texts:
            yield self.redact(t, threshold=threshold)

    def scrub(self, text: str,
              threshold: float | None = None) -> tuple[str, dict]:
        """Scrub text for a trip outside the trust boundary (e.g. a
        cloud LLM call): returns (redacted_text, mapping).

        Send the redacted text out; keep the mapping local; restore the
        response with restore() — or with ChunkedRestorer when the
        response streams. Round-trips require the default "numbered"
        operator; the other operators return an empty mapping by design
        (they are irreversible).
        """
        result = self.redact(text, threshold=threshold)
        return result.redacted, result.mapping

    @staticmethod
    def restore(redacted: str, mapping: dict) -> str:
        return rules.restore_text(redacted, mapping)
