"""
Deterministic validation layer for phi-redact.

Structured PII/PHI (cards, SSNs, IBANs, phones, emails, ...) have checksums or
rigid formats. Validating them with rules instead of model capacity buys
near-free recall and lets the neural model focus on what rules can't do:
names, addresses, and PHI in free-text clinical prose.

`scan(text)` returns a list of entity dicts:
    {"start": int, "end": int, "label": str, "text": str, "source": "rule"}

Checks are deliberately strict (checksum-gated) so a random 16-digit string
is never masked as a card number.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Checksum validators
# ---------------------------------------------------------------------------

def luhn_ok(number: str) -> bool:
    """True if the digit string passes the Luhn checksum (credit cards)."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def ssn_valid(ssn: str) -> bool:
    """True for a structurally valid US SSN (area/group/serial rules).

    Rejects 000/666/900-999 areas, 00 groups, 0000 serials. This is a format
    check, not proof the SSN is issued.
    """
    m = re.fullmatch(r"(\d{3})-(\d{2})-(\d{4})", ssn)
    if not m:
        return False
    area, group, serial = m.groups()
    if area == "000" or area == "666" or 900 <= int(area) <= 999:
        return False
    if group == "00" or serial == "0000":
        return False
    return True


def iban_valid(iban: str) -> bool:
    """True if the IBAN passes the ISO 13616 mod-97 checksum."""
    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(digits) % 97 == 1


# ---------------------------------------------------------------------------
# Pattern scanners: regex -> candidate spans, gated by validators above.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, callable | None]] = [
    # (label, compiled regex, validator or None)
    ("CREDITCARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"),
     lambda s: luhn_ok(s)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ssn_valid),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), iban_valid),
    ("PHONE", re.compile(
        # (?<![\w]) instead of a leading \b: a "(" or "+" directly before
        # the digits belongs to the number ("(415) 555-0132",
        # "+1 415-555-0132") and must be consumed, not left behind.
        r"(?<![\w])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b"),
     None),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    ("URL", re.compile(r"\bhttps?://[^\s<>\"]+"), None),
    ("IP", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"),
     None),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b"), None),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        re.IGNORECASE), None),
]

_MRN_RES = [
    re.compile(r"\bMRN \d{6,9}\b"),
    re.compile(r"\bMR#\d{6,9}\b"),
    re.compile(r"\bMedical Record \d{6,9}\b"),
]


def scan(text: str) -> list[dict]:
    """Run the deterministic layer over text. Returns entity dicts."""
    entities: list[dict] = []
    claimed: list[tuple[int, int]] = []  # spans already taken

    def free(s: int, e: int) -> bool:
        return all(e <= cs or s >= ce for cs, ce in claimed)

    for label, rx, validator in _PATTERNS:
        for m in rx.finditer(text):
            s, e = m.span()
            value = m.group().rstrip(" -")  # regex can swallow a trailing separator
            e = s + len(value)
            if not free(s, e):
                continue
            if validator is not None and not validator(value):
                continue
            entities.append({"start": s, "end": e, "label": label,
                             "text": value, "source": "rule"})
            claimed.append((s, e))

    for rx in _MRN_RES:
        for m in rx.finditer(text):
            s, e = m.span()
            if not free(s, e):
                continue
            entities.append({"start": s, "end": e, "label": "MRN",
                             "text": m.group(), "source": "rule"})
            claimed.append((s, e))

    entities.sort(key=lambda d: d["start"])
    return entities


def redact_text(text: str, entities: list[dict] | None = None) -> tuple[str, dict]:
    """Redact rule-found entities with numbered placeholders.

    Returns (redacted_text, placeholder_map). Keep the map for pseudonymized
    round-trips; delete it and the copy is anonymized.
    """
    entities = scan(text) if entities is None else entities
    counters: dict[str, int] = {}
    mapping: dict[str, str] = {}
    # Replace back-to-front so offsets stay valid.
    out = text
    for e in sorted(entities, key=lambda d: d["start"], reverse=True):
        counters[e["label"]] = counters.get(e["label"], 0) + 1
        ph = f"[{e['label']}_{counters[e['label']]}]"
        mapping[ph] = e["text"]
        out = out[:e["start"]] + ph + out[e["end"]:]
    return out, mapping


def restore_text(redacted: str, mapping: dict[str, str]) -> str:
    """Put real values back into a redacted text (LLM round-trip)."""
    out = redacted
    for ph, value in mapping.items():
        out = out.replace(ph, value)
    return out


if __name__ == "__main__":
    demo = ("Patient John Smith, SSN 123-45-6789, called from 415-555-0132. "
            "Card 4111111111111111 charged; random 1234567812345678 ignored. "
            "Email jsmith@example.com, MRN 48291375.")
    for e in scan(demo):
        print(e["label"], repr(e["text"]))
    red, mapping = redact_text(demo)
    print("REDACTED:", red)
    print("RESTORED OK:", restore_text(red, mapping) == demo)
