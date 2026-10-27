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

Merge policy: rule spans always win overlaps; neural spans that overlap a
rule span are dropped; placeholders are numbered per label in document
order. Keep the mapping for pseudonymized round-trips; delete it and the
copy is anonymized.

0.4.0 adds Presidio-parity SDK features (English text only): batch
redaction (Redactor.redact_batch / redact_iter), a first-class
scrub/restore pair for scrubbing prompts before a cloud LLM call —
including ChunkedRestorer for streaming LLM output — anonymize
operators ("numbered", "label", "mask", "hash", "remove"), and
allow/deny lists. Operators and lists are Redactor constructor options
(or keyword arguments to the module-level redact()).

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
    RedactionResult,
    apply_placeholders,
    merge_entities,
    process_entities,
    redact as _engine_redact,
)

__version__ = "0.4.0"

__all__ = [
    "ChunkedRestorer",
    "DEFAULT_DENY_LABEL",
    "DEFAULT_MAX_LEN",
    "DEFAULT_THRESHOLD",
    "OPERATORS",
    "OnnxEngine",
    "RedactionResult",
    "Redactor",
    "__version__",
    "apply_placeholders",
    "merge_entities",
    "process_entities",
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
           deny_label: str = DEFAULT_DENY_LABEL) -> RedactionResult:
    """Rules-only redaction. No model needed.

    Accepts the same operator / salt / allowlist / denylist options as
    Redactor for one-off calls.
    """
    return _engine_redact(text, operator=operator, salt=salt,
                          allowlist=allowlist, denylist=denylist,
                          deny_label=deny_label)


def restore(redacted: str, mapping: dict) -> str:
    """Put original values back using a mapping from redact()."""
    return rules.restore_text(redacted, mapping)


class Redactor:
    """Stateful redactor with optional ONNX neural layer.

    Without model/tokenizer this is rules-only, same as the module
    functions. With them, neural spans (names, addresses, free-text PHI)
    are merged under the rules-win policy.

    Options (all keyword): operator (placeholder style, see
    phi_redact.engine.OPERATORS; only "numbered" is reversible), salt
    (for the "hash" operator), allowlist (terms never redacted),
    denylist (terms always redacted), deny_label (label for denylist
    spans, default "CUSTOM").
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
                 deny_label: str = DEFAULT_DENY_LABEL):
        if operator not in OPERATORS:
            raise ValueError(
                f"unknown operator {operator!r}; expected one of {OPERATORS}")
        self.operator = operator
        self.salt = salt
        self.allowlist = list(allowlist) if allowlist else []
        self.denylist = list(denylist) if denylist else []
        self.deny_label = deny_label
        self.neural = (OnnxEngine(model, tokenizer, labels, max_len,
                                  threshold)
                       if model is not None else None)

    @property
    def engine_name(self) -> str:
        return "rules+onnx" if self.neural is not None else "rules"

    def scan(self, text: str) -> list[dict]:
        """Merged entity list (rules + neural when configured), with
        this Redactor's allow/deny lists applied."""
        neural_ents = self.neural.predict(text) if self.neural else []
        entities = merge_entities(rules.scan(text), neural_ents)
        return process_entities(text, entities, self.allowlist,
                                self.denylist, self.deny_label)

    def redact(self, text: str,
               threshold: float | None = None) -> RedactionResult:
        """Redact text; optional per-call neural threshold override."""
        if threshold is not None and self.neural is not None:
            self.neural.threshold = float(threshold)
        return _engine_redact(text, self.neural, operator=self.operator,
                              salt=self.salt, allowlist=self.allowlist,
                              denylist=self.denylist,
                              deny_label=self.deny_label)

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
