/**
 * Deterministic validation layer for phi-redact (TypeScript port of
 * redact/rules.py).
 *
 * Structured PII/PHI (cards, SSNs, IBANs, phones, emails, ...) have
 * checksums or rigid formats. Validating them with rules instead of model
 * capacity buys near-free recall and lets the neural model focus on what
 * rules can't do: names, addresses, and PHI in free-text clinical prose.
 *
 * `scan(text)` returns entity objects:
 *   { start, end, label, text, source: "rule" }
 *
 * Checks are deliberately strict (checksum-gated) so a random 16-digit
 * string is never masked as a card number.
 */

import type { Entity } from "./types.js";

// ---------------------------------------------------------------------------
// Checksum validators
// ---------------------------------------------------------------------------

/** True if the digit string passes the Luhn checksum (credit cards). */
export function luhnOk(number: string): boolean {
  const digits = number.replace(/\D/g, "").split("").map(Number);
  if (digits.length < 13 || digits.length > 19) return false;
  let total = 0;
  const rev = [...digits].reverse();
  for (let i = 0; i < rev.length; i++) {
    let d = rev[i];
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    total += d;
  }
  return total % 10 === 0;
}

/**
 * True for a structurally valid US SSN (area/group/serial rules).
 * Rejects 000/666/900-999 areas, 00 groups, 0000 serials. This is a format
 * check, not proof the SSN is issued.
 */
export function ssnValid(ssn: string): boolean {
  const m = /^(\d{3})-(\d{2})-(\d{4})$/.exec(ssn);
  if (!m) return false;
  const [, area, group, serial] = m;
  if (area === "000" || area === "666" || parseInt(area, 10) >= 900)
    return false;
  if (group === "00" || serial === "0000") return false;
  return true;
}

/** True if the IBAN passes the ISO 13616 mod-97 checksum. */
export function ibanValid(iban: string): boolean {
  const s = iban.replace(/\s+/g, "").toUpperCase();
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(s)) return false;
  const rearranged = s.slice(4) + s.slice(0, 4);
  const digits = rearranged
    .split("")
    .map((c) => (/[A-Z]/.test(c) ? String(c.charCodeAt(0) - 55) : c))
    .join("");
  return BigInt(digits) % 97n === 1n;
}

// ---------------------------------------------------------------------------
// Pattern scanners: regex -> candidate spans, gated by validators above.
// ---------------------------------------------------------------------------

interface Pattern {
  label: string;
  source: string;
  flags: string;
  validate?: (s: string) => boolean;
}

const PATTERNS: Pattern[] = [
  // (label, regex source, flags, validator or undefined)
  {
    label: "CREDITCARD",
    source: String.raw`\b(?:\d[ -]?){13,19}\b`,
    flags: "g",
    validate: luhnOk,
  },
  {
    label: "SSN",
    source: String.raw`\b\d{3}-\d{2}-\d{4}\b`,
    flags: "g",
    validate: ssnValid,
  },
  {
    label: "IBAN",
    source: String.raw`\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b`,
    flags: "g",
    validate: ibanValid,
  },
  {
    label: "PHONE",
    // (?<![\w]) instead of a leading \b: a "(" or "+" directly before the
    // digits belongs to the number and must be consumed, not left behind.
    source: String.raw`(?<![\w])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b`,
    flags: "g",
  },
  {
    label: "EMAIL",
    source: String.raw`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`,
    flags: "g",
  },
  { label: "URL", source: String.raw`\bhttps?://[^\s<>"]+`, flags: "g" },
  {
    label: "IP",
    source: String.raw`\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b`,
    flags: "g",
  },
  { label: "ZIP", source: String.raw`\b\d{5}(?:-\d{4})?\b`, flags: "g" },
  {
    label: "DATE",
    source: String.raw`\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b`,
    flags: "gi",
  },
];

const MRN_PATTERNS: Pattern[] = [
  { label: "MRN", source: String.raw`\bMRN \d{6,9}\b`, flags: "g" },
  { label: "MRN", source: String.raw`\bMR#\d{6,9}\b`, flags: "g" },
  { label: "MRN", source: String.raw`\bMedical Record \d{6,9}\b`, flags: "g" },
];

/** Run the deterministic layer over text. Returns entity objects. */
export function scan(text: string): Entity[] {
  const entities: Entity[] = [];
  const claimed: Array<[number, number]> = [];

  const free = (s: number, e: number): boolean =>
    claimed.every(([cs, ce]) => e <= cs || s >= ce);

  for (const { label, source, flags, validate } of [
    ...PATTERNS,
    ...MRN_PATTERNS,
  ]) {
    // Fresh RegExp per scan: /g regexes are stateful (lastIndex).
    const re = new RegExp(source, flags);
    for (const m of text.matchAll(re)) {
      const start = m.index ?? 0;
      // The card regex can swallow a trailing separator; trim it.
      const value = m[0].replace(/[ -]+$/, "");
      const end = start + value.length;
      if (!free(start, end)) continue;
      if (validate && !validate(value)) continue;
      entities.push({ start, end, label, text: value, source: "rule" });
      claimed.push([start, end]);
    }
  }

  entities.sort((a, b) => a.start - b.start);
  return entities;
}

/**
 * Redact rule-found entities with numbered placeholders.
 * Returns { redacted, mapping }. Keep the map for pseudonymized
 * round-trips; delete it and the copy is anonymized.
 */
export function redactText(
  text: string,
  entities?: Entity[],
): { redacted: string; mapping: Record<string, string> } {
  const found = entities ?? scan(text);
  const counters: Record<string, number> = {};
  const mapping: Record<string, string> = {};
  let out = text;
  // Replace back-to-front so offsets stay valid.
  for (const e of [...found].sort((a, b) => b.start - a.start)) {
    counters[e.label] = (counters[e.label] ?? 0) + 1;
    const ph = `[${e.label}_${counters[e.label]}]`;
    mapping[ph] = e.text;
    out = out.slice(0, e.start) + ph + out.slice(e.end);
  }
  return { redacted: out, mapping };
}

/** Put real values back into a redacted text (LLM round-trip). */
export function restoreText(
  redacted: string,
  mapping: Record<string, string>,
): string {
  let out = redacted;
  for (const [ph, value] of Object.entries(mapping)) {
    out = out.split(ph).join(value);
  }
  return out;
}
