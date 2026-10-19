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


def aba_routing_ok(number: str) -> bool:
    """True if the 9-digit string passes the ABA routing-number checksum.

    Checksum: 3*(d1+d4+d7) + 7*(d2+d5+d8) + (d3+d6+d9) == 0 (mod 10).
    Only ~1 in 10 random 9-digit strings pass, so this is a strong gate
    against plain 9-digit SSNs and other numeric noise.
    """
    d = [int(c) for c in number if c.isdigit()]
    if len(d) != 9:
        return False
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7])
            + (d[2] + d[5] + d[8])) % 10 == 0


# ---------------------------------------------------------------------------
# Pattern scanners: regex -> candidate spans, gated by validators above.
# ---------------------------------------------------------------------------
# Tuple: (label, compiled regex, validator or None, match group to use).
# Most patterns use group 0 (whole match); context-anchored patterns like
# BANK_ACCOUNT capture just the value in group 1 so the span covers only
# the number, not the anchor word.

_PATTERNS: list[tuple[str, re.Pattern, callable | None, int]] = [
    # (label, compiled regex, validator or None, group)
    ("CREDITCARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"),
     lambda s: luhn_ok(s), 0),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ssn_valid, 0),
    ("TAX_ID", re.compile(r"\b\d{2}-\d{7}\b"), None, 0),  # EIN: 12-3456789
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), iban_valid, 0),
    ("ROUTING_NUMBER", re.compile(r"\b\d{9}\b"), aba_routing_ok, 0),
    ("BANK_ACCOUNT", re.compile(r"\b(?:account|acct)(?:\s*(?:#|no\.?|number))?[-.\s]*(\d{9,17})\b",
                                re.IGNORECASE), None, 1),
    ("PASSPORT", re.compile(r"\b[A-Z]\d{8}\b"), None, 0),  # Y12973198
    ("DRIVERS_LICENSE", re.compile(r"\b[A-Z]\d{7}\b"), None, 0),  # D1315565
    ("DRIVERS_LICENSE", re.compile(r"\b[A-Z]{2}-L\d{6}\b"), None, 0),  # CA-L123456
    ("GOVERNMENT_ID", re.compile(r"\b[A-Z]{3}\d{10}\b"), None, 0),  # USCIS LIN2190012345
    ("GOVERNMENT_ID", re.compile(r"\bA\d{9}\b"), None, 0),  # alien number A123456789
    ("PHONE", re.compile(
        # A leading "(" or "+" belongs to the number and must be consumed.
        # (?<![\w]) is the normal start; (?<=\w)\( additionally lets the
        # match begin at a "(" glued to a word char, as in "x(312) 555-0167".
        r"(?:(?<=\w)\(|(?<![\w]))(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b"),
     None, 0),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None, 0),
    ("URL", re.compile(r"\bhttps?://[^\s<>\"]+"), None, 0),
    ("IP", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"),
     None, 0),
    ("ZIP", re.compile(r"\b\d{5}(?:-\d{4})?\b"), None, 0),
    ("DATE", re.compile(
        r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        re.IGNORECASE), None, 0),
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

    for label, rx, validator, grp in _PATTERNS:
        for m in rx.finditer(text):
            if grp:
                # Context-anchored pattern: the value is in group `grp`,
                # the span covers only the value, not the anchor.
                s, e = m.span(grp)
                value = m.group(grp)
            else:
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
            "Email jsmith@example.com, MRN 48291375. EIN 12-3456789, "
            "routing 021000021, Account 713591432266, passport Y12973198, "
            "DL D1315565, USCIS LIN2190012345.")
    for e in scan(demo):
        print(e["label"], repr(e["text"]))
    red, mapping = redact_text(demo)
    print("REDACTED:", red)
    print("RESTORED OK:", restore_text(red, mapping) == demo)
