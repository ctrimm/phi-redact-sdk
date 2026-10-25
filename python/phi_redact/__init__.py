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

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.
"""

from __future__ import annotations

from redact import rules

from .engine import (
    DEFAULT_MAX_LEN,
    DEFAULT_THRESHOLD,
    OnnxEngine,
    RedactionResult,
    apply_placeholders,
    merge_entities,
    redact as _engine_redact,
)

__version__ = "0.3.1"

__all__ = [
    "DEFAULT_MAX_LEN",
    "DEFAULT_THRESHOLD",
    "OnnxEngine",
    "RedactionResult",
    "Redactor",
    "__version__",
    "apply_placeholders",
    "merge_entities",
    "redact",
    "restore",
    "scan",
]


def scan(text: str) -> list[dict]:
    """Rules-only entity scan. Returns entity dicts sorted by start."""
    return rules.scan(text)


def redact(text: str) -> RedactionResult:
    """Rules-only redaction. No model needed."""
    return _engine_redact(text)


def restore(redacted: str, mapping: dict) -> str:
    """Put original values back using a mapping from redact()."""
    return rules.restore_text(redacted, mapping)


class Redactor:
    """Stateful redactor with optional ONNX neural layer.

    Without model/tokenizer this is rules-only, same as the module
    functions. With them, neural spans (names, addresses, free-text PHI)
    are merged under the rules-win policy.
    """

    def __init__(self, model: str | None = None,
                 tokenizer: str | None = None,
                 labels: str | None = None,
                 max_len: int = DEFAULT_MAX_LEN,
                 threshold: float = DEFAULT_THRESHOLD):
        self.neural = (OnnxEngine(model, tokenizer, labels, max_len,
                                  threshold)
                       if model is not None else None)

    @property
    def engine_name(self) -> str:
        return "rules+onnx" if self.neural is not None else "rules"

    def scan(self, text: str) -> list[dict]:
        """Merged entity list (rules + neural when configured)."""
        neural_ents = self.neural.predict(text) if self.neural else []
        return merge_entities(rules.scan(text), neural_ents)

    def redact(self, text: str,
               threshold: float | None = None) -> RedactionResult:
        """Redact text; optional per-call neural threshold override."""
        if threshold is not None and self.neural is not None:
            self.neural.threshold = float(threshold)
        return _engine_redact(text, self.neural)

    @staticmethod
    def restore(redacted: str, mapping: dict) -> str:
        return rules.restore_text(redacted, mapping)
