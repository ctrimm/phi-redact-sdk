"""Scrub PII/PHI out of log records before they are stored or shipped.

Three entry points, one idea — run every log line through a Redactor
at emit time, so raw identifiers never reach the log store:

- ``RedactingFilter`` — a stdlib ``logging.Filter`` for the message
  (and optionally the formatted exception) of every record.
- ``add_redacting_filter`` — attach one to a logger or handler in one
  call.
- ``make_structlog_processor`` — the same scrubbing as a structlog
  processor (structlog itself is NOT imported; the returned function
  is duck-typed to structlog's processor protocol).

Placement semantics, honestly stated: a filter at emit time — inside
the process that produces the log line — is the only placement where
PII never touches disk or the wire at all. Pipeline-side scrubbing
(the Lambda in ``examples/lambda_log_scrubber.py``, a batch pass over
exports) protects everything *downstream* of the pipeline, but the
raw lines still sit in the source log group. Pick the placement that
matches the requirement; the guide at docs/logging-integrations.md
walks through CloudWatch and Datadog for both.

Failure policy: fail-open. If redaction itself raises, the original
record passes through unchanged — a redaction bug must never take
down the application's logging (or the application). The tradeoff is
real: a silent scrubbing failure can leak the very values this module
exists to remove. Mitigate by monitoring redaction coverage outside
the log path (the Redactor's counts on sampled traffic), not by
failing the log call.
"""

from __future__ import annotations

import logging

__all__ = [
    "RedactingFilter",
    "add_redacting_filter",
    "make_structlog_processor",
]

_FORMATTER = logging.Formatter()


class RedactingFilter(logging.Filter):
    """A logging.Filter that redacts each record in place.

    ``redactor`` is anything with the SDK Redactor's shape —
    ``redact(text)`` returning a result with a ``redacted`` string
    (in practice, a ``phi_redact.Redactor``).

    The record's message is rendered with ``record.getMessage()``
    (which resolves ``%``-args), scrubbed, and written back as
    ``record.msg`` with ``record.args = ()``, so every downstream
    handler emits the scrubbed line. Note this mutates the shared
    record object: other handlers on the same logger see the scrubbed
    version too (usually what you want — there is one record).

    With ``scrub_args=False`` the message is left alone entirely —
    both the template and its args pass through unredacted. Only use
    it when log templates are known to be free of embedded values.

    With ``scrub_exc`` (default), a record's exception text is also
    scrubbed: if ``record.exc_text`` is already formatted it is
    scrubbed in place; otherwise the traceback is formatted once
    (cached on the record, the way Formatter.format does it), scrubbed
    and stored back, so tracebacks that mention identifiers — a
    connection string, a patient id in a ValueError — are covered too.
    """

    def __init__(self, redactor, *, scrub_args: bool = True,
                 scrub_exc: bool = True, name: str = ""):
        super().__init__(name)
        self.redactor = redactor
        self.scrub_args = scrub_args
        self.scrub_exc = scrub_exc

    def _scrub(self, text: str) -> str:
        return self.redactor.redact(text).redacted

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if self.scrub_args:
                rendered = record.getMessage()
                scrubbed = self._scrub(rendered)
                if scrubbed != rendered:
                    record.msg = scrubbed
                    record.args = ()
            if self.scrub_exc:
                if record.exc_text:
                    scrubbed = self._scrub(record.exc_text)
                    if scrubbed != record.exc_text:
                        record.exc_text = scrubbed
                elif record.exc_info:
                    text = _FORMATTER.formatException(record.exc_info)
                    record.exc_text = self._scrub(text)
        except Exception:
            # Fail-open: never break logging (see module docstring).
            pass
        return True


def add_redacting_filter(target, redactor, **kw) -> RedactingFilter:
    """Build a RedactingFilter and attach it to a logger or handler.

    ``target`` is anything with ``addFilter`` (a ``logging.Logger``
    or ``logging.Handler``); extra keyword arguments pass through to
    RedactingFilter (``scrub_args``, ``scrub_exc``). Returns the
    filter so the caller can remove or inspect it later.
    """
    filt = RedactingFilter(redactor, **kw)
    target.addFilter(filt)
    return filt


def make_structlog_processor(redactor):
    """Return a structlog processor that scrubs an event dict.

    The returned ``processor(logger, method_name, event_dict)``
    scrubs the ``event`` string and, recursively, every string leaf
    of nested dicts/lists/tuples in the event dict; non-string values
    (numbers, booleans, None) pass through untouched. Dict KEYS are
    not scrubbed: keys are field names chosen by the developer, not
    data — a key that carries a raw value is a logging bug to fix at
    the call site. Returns a new dict; the input is not mutated.

    structlog is not imported or required — the processor is
    duck-typed, so it also drops into any pipeline that calls
    processors with structlog's signature. Like the filter, it fails
    open: on a redaction error the original event dict is returned.
    """

    def _scrub_value(value):
        if isinstance(value, str):
            return redactor.redact(value).redacted
        if isinstance(value, dict):
            return {k: _scrub_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_scrub_value(v) for v in value)
        return value

    def processor(logger, method_name, event_dict):
        try:
            return {k: _scrub_value(v) for k, v in event_dict.items()}
        except Exception:
            return event_dict

    return processor
