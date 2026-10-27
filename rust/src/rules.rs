//! Deterministic validation layer for phi-redact (Rust port of
//! `redact/rules.py`).
//!
//! Structured PII/PHI (cards, SSNs, IBANs, phones, emails, ...) have
//! checksums or rigid formats. Validating them with rules instead of
//! model capacity buys near-free recall and lets the neural model
//! focus on what rules can't do: names, addresses, and PHI in
//! free-text clinical prose.
//!
//! `scan(text)` returns entities `{start, end, label, text, source}`
//! with character offsets, exactly like the Python layer. Checks are
//! deliberately strict (checksum-gated) so a random 16-digit string
//! is never masked as a card number.
//!
//! ## Regex-dialect notes (Python `re` -> Rust `regex`)
//!
//! The Rust `regex` crate has no lookahead/lookbehind/backreferences.
//! Auditing every pattern in `rules.py`:
//!
//! - All patterns except PHONE translate verbatim (`\d`, `\b`, `\s`
//!   are Unicode-aware in both engines; alternation is
//!   leftmost-first in both).
//! - PHONE uses `(?<=\w)\(` / `(?<![\w])` lookbehinds to decide
//!   whether a match may start at a given position. It is
//!   re-implemented in `phone_matches`: the body pattern runs without
//!   the lookbehinds and each candidate's start is checked manually
//!   (a match may start at a position whose preceding character is
//!   not a word character, or at a `(` glued to a word character).
//!   Rejected candidates advance the scan by one character, exactly
//!   like the Python engine advancing its start position.

use indexmap::IndexMap;
use regex::Regex;
use std::sync::LazyLock;

use crate::text::{is_word_char, next_char_boundary, prev_char, CharIndex};
use crate::types::{Entity, EntitySource};

// ---------------------------------------------------------------------------
// Checksum validators
// ---------------------------------------------------------------------------

/// Digits of a string as values, Python `[int(d) for d in s if
/// d.isdigit()]` semantics (Unicode decimal digits).
fn digit_values(s: &str) -> Vec<u32> {
    s.chars().filter_map(|c| c.to_digit(10)).collect()
}

/// True if the digit string passes the Luhn checksum (credit cards).
pub fn luhn_ok(number: &str) -> bool {
    let digits = digit_values(number);
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut total: u32 = 0;
    for (i, d) in digits.iter().rev().enumerate() {
        let mut d = *d;
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        total += d;
    }
    total % 10 == 0
}

/// True for a structurally valid US SSN (area/group/serial rules).
/// Rejects 000/666/900-999 areas, 00 groups, 0000 serials. This is a
/// format check, not proof the SSN is issued.
pub fn ssn_valid(ssn: &str) -> bool {
    let chars: Vec<char> = ssn.chars().collect();
    if chars.len() != 11 || chars[3] != '-' || chars[6] != '-' {
        return false;
    }
    let mut digits: Vec<u32> = Vec::with_capacity(9);
    for (i, c) in chars.iter().enumerate() {
        if i == 3 || i == 6 {
            continue;
        }
        match c.to_digit(10) {
            Some(d) => digits.push(d),
            None => return false,
        }
    }
    if digits.len() != 9 {
        return false;
    }
    let area_s: String = chars[0..3].iter().collect();
    let area = digits[0] * 100 + digits[1] * 10 + digits[2];
    let group_s: String = chars[4..6].iter().collect();
    let serial_s: String = chars[7..11].iter().collect();
    if area_s == "000" || area_s == "666" || (900..=999).contains(&area) {
        return false;
    }
    if group_s == "00" || serial_s == "0000" {
        return false;
    }
    true
}

/// True if the IBAN passes the ISO 13616 mod-97 checksum.
pub fn iban_valid(iban: &str) -> bool {
    let s: String = iban
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_uppercase();
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 15 || chars.len() > 34 {
        return false;
    }
    if !chars[0].is_ascii_uppercase() || !chars[1].is_ascii_uppercase() {
        return false;
    }
    if !chars[2].is_ascii_digit() || !chars[3].is_ascii_digit() {
        return false;
    }
    if !chars[4..]
        .iter()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return false;
    }
    // mod-97 over the rearranged digit expansion, computed
    // incrementally (equivalent to Python's big-int computation).
    let mut rem: u32 = 0;
    for c in chars[4..].iter().chain(chars[..4].iter()) {
        if c.is_ascii_alphabetic() {
            let v = *c as u32 - 55; // A=10 .. Z=35, two digits
            rem = (rem * 100 + v) % 97;
        } else {
            rem = (rem * 10 + c.to_digit(10).unwrap()) % 97;
        }
    }
    rem == 1
}

/// True if the 9-digit string passes the ABA routing-number checksum:
/// 3*(d1+d4+d7) + 7*(d2+d5+d8) + (d3+d6+d9) == 0 (mod 10). Only ~1 in
/// 10 random 9-digit strings pass, so this is a strong gate against
/// plain 9-digit SSNs and other numeric noise.
pub fn aba_routing_ok(number: &str) -> bool {
    let d = digit_values(number);
    if d.len() != 9 {
        return false;
    }
    (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0
}

// ---------------------------------------------------------------------------
// Pattern scanners: regex -> candidate spans, gated by validators.
// ---------------------------------------------------------------------------

type Validator = fn(&str) -> bool;

struct PatternSpec {
    label: &'static str,
    validator: Option<Validator>,
    /// Match group whose span/text is the entity (0 = whole match).
    group: usize,
    /// PHONE needs the lookbehind re-implementation.
    phone: bool,
}

struct CompiledPattern {
    spec: PatternSpec,
    re: Regex,
}

fn compile(label: &'static str, source: &str, validator: Option<Validator>, group: usize) -> CompiledPattern {
    CompiledPattern {
        spec: PatternSpec { label, validator, group, phone: false },
        re: Regex::new(source).expect("built-in pattern must compile"),
    }
}

static PATTERNS: LazyLock<Vec<CompiledPattern>> = LazyLock::new(|| {
    vec![
        compile("CREDITCARD", r"\b(?:\d[ -]?){13,19}\b", Some(luhn_ok), 0),
        compile("SSN", r"\b\d{3}-\d{2}-\d{4}\b", Some(ssn_valid), 0),
        // EIN: 12-3456789
        compile("TAX_ID", r"\b\d{2}-\d{7}\b", None, 0),
        compile("IBAN", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", Some(iban_valid), 0),
        compile("ROUTING_NUMBER", r"\b\d{9}\b", Some(aba_routing_ok), 0),
        compile(
            "BANK_ACCOUNT",
            r"(?i)\b(?:account|acct)(?:\s*(?:#|no\.?|number))?[-.\s]*(\d{9,17})\b",
            None,
            1,
        ),
        // Y12973198
        compile("PASSPORT", r"\b[A-Z]\d{8}\b", None, 0),
        // D1315565
        compile("DRIVERS_LICENSE", r"\b[A-Z]\d{7}\b", None, 0),
        // CA-L123456
        compile("DRIVERS_LICENSE", r"\b[A-Z]{2}-L\d{6}\b", None, 0),
        // USCIS LIN2190012345
        compile("GOVERNMENT_ID", r"\b[A-Z]{3}\d{10}\b", None, 0),
        // alien number A123456789
        compile("GOVERNMENT_ID", r"\bA\d{9}\b", None, 0),
        CompiledPattern {
            spec: PatternSpec { label: "PHONE", validator: None, group: 0, phone: true },
            re: Regex::new(PHONE_BODY).expect("phone body must compile"),
        },
        compile("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", None, 0),
        compile("URL", r#"\bhttps?://[^\s<>"]+"#, None, 0),
        compile(
            "IP",
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
            None,
            0,
        ),
        compile("ZIP", r"\b\d{5}(?:-\d{4})?\b", None, 0),
        compile(
            "DATE",
            r"(?i)\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
            None,
            0,
        ),
        // MRN patterns run after the main table, like rules.py.
        compile("MRN", r"\bMRN \d{6,9}\b", None, 0),
        compile("MRN", r"\bMR#\d{6,9}\b", None, 0),
        compile("MRN", r"\bMedical Record \d{6,9}\b", None, 0),
    ]
});

/// The PHONE pattern body — everything after the lookbehind
/// alternation in the Python original:
///
/// ```text
/// (?:(?<=\w)\(|(?<![\w]))(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b
/// ```
const PHONE_BODY: &str =
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b";

/// One raw regex match, in byte offsets. `group` is the byte range
/// of the configured capture group for group-anchored patterns.
struct RawMatch {
    start: usize,
    end: usize,
    group: Option<(usize, usize)>,
}

/// Whether a PHONE body match starting at byte `start` satisfies the
/// Python lookbehind alternation: the preceding character is absent
/// or not a word character (the `(?<![\w])` branch), or the match
/// begins with a `(` whose preceding character is a word character
/// (the `(?<=\w)\(` branch — the body necessarily consumes that `(`).
fn phone_start_ok(text: &str, start: usize) -> bool {
    match prev_char(text, start) {
        None => true,
        Some(c) if !is_word_char(c) => true,
        Some(_) => text[start..].starts_with('('),
    }
}

/// All boundary-valid PHONE matches, in document order. A candidate
/// rejected by the boundary check advances the scan by one character
/// from its start, mirroring the Python engine trying the next start
/// position; accepted (or consumed) matches advance to their end,
/// mirroring `finditer`.
fn phone_matches(text: &str, re: &Regex) -> Vec<RawMatch> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos <= text.len() {
        let Some(m) = re.find_at(text, pos) else { break };
        if phone_start_ok(text, m.start()) {
            out.push(RawMatch { start: m.start(), end: m.end(), group: None });
            // Consumed even if the caller later rejects the span
            // (Python finditer semantics). Guard against zero-width.
            pos = if m.end() > m.start() { m.end() } else { next_char_boundary(text, m.end()) };
        } else {
            pos = next_char_boundary(text, m.start());
        }
    }
    out
}

fn raw_matches(text: &str, pat: &CompiledPattern) -> Vec<RawMatch> {
    if pat.spec.phone {
        return phone_matches(text, &pat.re);
    }
    let mut out = Vec::new();
    for caps in pat.re.captures_iter(text) {
        let whole = caps.get(0).expect("group 0 always present");
        if pat.spec.group == 0 {
            out.push(RawMatch { start: whole.start(), end: whole.end(), group: None });
        } else if let Some(g) = caps.get(pat.spec.group) {
            out.push(RawMatch {
                start: whole.start(),
                end: whole.end(),
                group: Some((g.start(), g.end())),
            });
        }
        // A non-participating group yields no candidate (Python skips
        // the match the same way).
    }
    out
}

/// Run the deterministic layer over text. Returns entity objects
/// sorted by start, with character offsets.
pub fn scan(text: &str) -> Vec<Entity> {
    let idx = CharIndex::new(text);
    let mut entities: Vec<Entity> = Vec::new();
    let mut claimed: Vec<(usize, usize)> = Vec::new(); // char ranges

    let free = |claimed: &[(usize, usize)], s: usize, e: usize| {
        claimed.iter().all(|&(cs, ce)| e <= cs || s >= ce)
    };

    for pat in PATTERNS.iter() {
        for m in raw_matches(text, pat) {
            let (bs, be, value) = match m.group {
                Some((gs, ge)) => (gs, ge, &text[gs..ge]),
                None => {
                    // Group 0: the regex can swallow a trailing
                    // separator; Python rstrips " -" off the value.
                    let full = &text[m.start..m.end];
                    let trimmed = full.trim_end_matches([' ', '-']);
                    (m.start, m.start + trimmed.len(), trimmed)
                }
            };
            let s = idx.char_of_byte(bs);
            let e = idx.char_of_byte(be);
            if !free(&claimed, s, e) {
                continue;
            }
            if let Some(validate) = pat.spec.validator {
                if !validate(value) {
                    continue;
                }
            }
            entities.push(Entity::new(s, e, pat.spec.label, value, EntitySource::Rule));
            claimed.push((s, e));
        }
    }

    entities.sort_by_key(|e| e.start);
    entities
}

/// Redact rule-found entities with numbered placeholders. Returns
/// (redacted_text, placeholder_map). Keep the map for pseudonymized
/// round-trips; delete it and the copy is anonymized.
///
/// NOTE: like the Python/TS rules modules, replacements are computed
/// back-to-front and counters increment in that pass — so within this
/// standalone helper the LAST span of a label in the document gets
/// `_1`. (The engine's `apply_placeholders` numbers in document
/// order; this mirrors `redact/rules.py::redact_text` exactly.)
pub fn redact_text(
    text: &str,
    entities: Option<Vec<Entity>>,
) -> (String, IndexMap<String, String>) {
    let entities = entities.unwrap_or_else(|| scan(text));
    let idx = CharIndex::new(text);
    let mut counters: IndexMap<String, usize> = IndexMap::new();
    let mut mapping: IndexMap<String, String> = IndexMap::new();
    let mut out = text.to_string();
    let mut sorted: Vec<&Entity> = entities.iter().collect();
    // Stable descending sort: ties keep their original relative
    // order, matching Python's sorted(..., reverse=True).
    sorted.sort_by(|a, b| b.start.cmp(&a.start));
    for e in sorted {
        let n = {
            let c = counters.entry(e.label.clone()).or_insert(0);
            *c += 1;
            *c
        };
        let ph = format!("[{}_{}]", e.label, n);
        mapping.insert(ph.clone(), e.text.clone());
        let (bs, be) = (idx.byte_of(e.start), idx.byte_of(e.end));
        out.replace_range(bs..be, &ph);
    }
    (out, mapping)
}

/// Put real values back into a redacted text (LLM round-trip).
///
/// Replacements run longest-placeholder-first: defensive ordering so
/// a mapping value that itself contains placeholder-shaped text can
/// never be re-matched by a shorter key later in the pass. (The
/// closing "]" already disambiguates keys like "[PHONE_1]" /
/// "[PHONE_12]" from each other.)
pub fn restore_text(redacted: &str, mapping: &IndexMap<String, String>) -> String {
    let mut entries: Vec<(&String, &String)> = mapping.iter().collect();
    // Stable sort: insertion order breaks length ties, like Python.
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    let mut out = redacted.to_string();
    for (ph, value) in entries {
        out = out.replace(ph.as_str(), value);
    }
    out
}
