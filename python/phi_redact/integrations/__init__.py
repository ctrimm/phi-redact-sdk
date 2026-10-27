"""Integration helpers for phi-redact.

Currently: `phi_redact.integrations.logging_kit` — scrub PII/PHI out of
log records before they are stored or shipped (stdlib `logging` filter,
structlog processor). Import from the submodule directly:

    from phi_redact.integrations.logging_kit import RedactingFilter
"""
