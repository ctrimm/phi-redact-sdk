/**
 * Merge + placeholder layer (TypeScript port of sdk/python/phi_redact/engine.py).
 *
 * Local engine is rules-only. `mergeEntities` accepts neural spans so a
 * future onnxruntime-based layer can plug in without changing the API:
 * rules win every overlap, neural-neural overlaps keep the higher score.
 *
 * 0.4.0 (Presidio parity, English text only): batch redaction
 * (redactBatch), scrub() for the LLM round-trip, ChunkedRestorer for
 * streaming restore, anonymize operators ("numbered", "label", "mask",
 * "hash", "remove"), and allow/deny lists — same semantics as the
 * Python engine, with shared test vectors pinning the "hash" operator.
 *
 * 0.5.0: custom recognizers (regex patterns with base scores,
 * context-word boosting, optional validators; spans carry source
 * "recognizer" and merge under rules > recognizers > neural), plus
 * per-label thresholds with a needs-review mode (reviewMargin):
 * scored spans that miss their threshold but land within the margin
 * come back on result.review instead of being silently dropped. Spans
 * without a score (rules, denylist) are never gated.
 */

import {
  abaRoutingOk,
  ibanValid,
  luhnOk,
  restoreText,
  scan as rulesScan,
  ssnValid,
} from "./rules.js";
import type {
  AnonymizeOperator,
  Entity,
  Recognizer,
  RedactOptions,
  RedactionResult,
  ReviewCandidate,
  ValidatorName,
} from "./types.js";

export const OPERATORS: readonly AnonymizeOperator[] = [
  "numbered",
  "label",
  "mask",
  "hash",
  "remove",
];

export const DEFAULT_DENY_LABEL = "CUSTOM";

/** Global score gate for scored spans without a per-label override.
    Matches the Python engine's DEFAULT_THRESHOLD (the v9 ship gate). */
export const DEFAULT_THRESHOLD = 0.6;

const overlaps = (a: Entity, b: Entity): boolean =>
  !(a.end <= b.start || a.start >= b.end);

/**
 * Merge the detection layers. Precedence: rules > recognizers >
 * neural. Rules win every overlap (checksum-gated precision).
 * Recognizer spans are deterministic user intent: they win overlaps
 * against neural spans, but yield to rule spans — the built-in rules
 * are effectively "registered first". recognizerEnts must arrive in
 * precedence order (recognizerSpans returns them so); among
 * themselves the first wins. Neural-neural overlaps keep the best
 * score. Returns entities sorted by start.
 */
export function mergeEntities(
  ruleEnts: Entity[],
  neuralEnts: Entity[] = [],
  recognizerEnts: Entity[] = [],
): Entity[] {
  const base: Entity[] = [...ruleEnts];
  for (const r of recognizerEnts) {
    if (base.some((b) => overlaps(r, b))) continue;
    base.push(r);
  }
  const kept: Entity[] = [];
  for (const n of [...neuralEnts].sort(
    (a, b) => (b.score ?? 0) - (a.score ?? 0),
  )) {
    if (base.some((b) => overlaps(n, b))) continue;
    if (kept.some((k) => overlaps(n, k))) continue;
    kept.push(n);
  }
  return [...base, ...kept].sort((a, b) => a.start - b.start);
}

// ---------------------------------------------------------------------------
// SHA-256 — pure TS: the SDK is zero-dependency and browser-safe, so
// node:crypto is not an option. Used only by the "hash" operator.
// ---------------------------------------------------------------------------

const SHA256_K = [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));

/** SHA-256 of the UTF-8 encoding, as lowercase hex. */
export function sha256Hex(input: string): string {
  const bytes = new TextEncoder().encode(input);
  const bitLen = bytes.length * 8;
  const paddedLen = (((bytes.length + 8) >> 6) + 1) << 6;
  const buf = new Uint8Array(paddedLen);
  buf.set(bytes);
  buf[bytes.length] = 0x80;
  const dv = new DataView(buf.buffer);
  dv.setUint32(paddedLen - 4, bitLen >>> 0);
  dv.setUint32(paddedLen - 8, Math.floor(bitLen / 2 ** 32));

  let h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
  let h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;
  const w = new Int32Array(64);
  for (let block = 0; block < paddedLen; block += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getInt32(block + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    for (let i = 0; i < 64; i++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + s1 + ch + SHA256_K[i] + w[i]) | 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + maj) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0;
      d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }
  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((x) => (x >>> 0).toString(16).padStart(8, "0"))
    .join("");
}

/**
 * Consistent pseudonym token for the "hash" operator:
 * [LABEL_<8 hex>] from SHA-256(salt \0 label \0 value). Identical
 * construction to the Python SDK (shared test vectors pin it). The
 * salt defaults to "" but should be a secret in practice: low-entropy
 * values (SSNs) are dictionary-attackable from an unsalted hash.
 */
export function hashToken(label: string, value: string, salt = ""): string {
  const digest = sha256Hex(`${salt}\0${label}\0${value}`).slice(0, 8);
  return `[${label}_${digest}]`;
}

// Labels whose values are structured IDs: masking keeps the last 4
// characters when the value is long enough for that to be safe.
const MASK_KEEP_LAST = new Set([
  "SSN", "CREDITCARD", "PHONE", "TAX_ID", "ROUTING_NUMBER",
  "BANK_ACCOUNT", "ACCOUNT", "IBAN", "PASSPORT", "DRIVERS_LICENSE",
  "LICENSE", "GOVERNMENT_ID", "MRN",
]);

/**
 * Masked rendering for the "mask" operator. Structured IDs of length
 * >= 8 keep their last 4 characters; preceding alphanumeric characters
 * become "*" and separators are preserved ("234-56-7890" ->
 * "***-**-7890"). Every other value is masked to its own length.
 */
export function maskValue(label: string, value: string): string {
  if (MASK_KEEP_LAST.has(label) && value.length >= 8) {
    const head = [...value.slice(0, -4)]
      .map((c) => (/[A-Za-z0-9]/.test(c) ? "*" : c))
      .join("");
    return head + value.slice(-4);
  }
  return "*".repeat(value.length);
}

function replacement(
  e: Entity,
  n: number,
  operator: AnonymizeOperator,
  salt: string,
): string {
  switch (operator) {
    case "numbered":
      return `[${e.label}_${n}]`;
    case "label":
      return `<${e.label}>`;
    case "hash":
      return hashToken(e.label, e.text, salt);
    case "mask":
      return maskValue(e.label, e.text);
    case "remove":
      return "";
  }
}

/**
 * Render entities into text under the chosen anonymize operator.
 * Replacements are computed in forward document order (so "numbered"
 * numbers per label in reading order), then spliced back-to-front so
 * offsets stay valid. The returned mapping is populated only for
 * "numbered" — the other operators are irreversible by design.
 */
export function applyPlaceholders(
  text: string,
  entities: Entity[],
  options: RedactOptions = {},
): { redacted: string; mapping: Record<string, string> } {
  const operator = options.operator ?? "numbered";
  if (!OPERATORS.includes(operator)) {
    throw new Error(
      `unknown operator ${JSON.stringify(operator)}; expected one of ${OPERATORS.join(", ")}`,
    );
  }
  const salt = options.salt ?? "";
  const counters: Record<string, number> = {};
  const mapping: Record<string, string> = {};
  const replacements = new Map<Entity, string>();
  for (const e of [...entities].sort((a, b) => a.start - b.start)) {
    counters[e.label] = (counters[e.label] ?? 0) + 1;
    const rep = replacement(e, counters[e.label], operator, salt);
    e.placeholder = rep;
    replacements.set(e, rep);
    if (operator === "numbered") mapping[rep] = e.text;
  }
  let out = text;
  for (const e of [...entities].sort((a, b) => b.start - a.start)) {
    out = out.slice(0, e.start) + replacements.get(e)! + out.slice(e.end);
  }
  return { redacted: out, mapping };
}

// ---------------------------------------------------------------------------
// Allow / deny lists
// ---------------------------------------------------------------------------

/**
 * Drop spans whose text matches an allowlist term (case-insensitive,
 * exact string — a term protects exactly the value it names).
 */
export function applyAllowlist(
  entities: Entity[],
  allowlist?: string[],
): Entity[] {
  if (!allowlist || allowlist.length === 0) return entities;
  const allowed = new Set(allowlist.map((t) => t.toLowerCase()));
  return entities.filter((e) => !allowed.has(e.text.toLowerCase()));
}

const escapeRegExp = (s: string): string =>
  s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Exact-match spans for denylist terms: case-insensitive, word-boundary
 * aware (a term never matches inside a larger word).
 */
export function denylistSpans(
  text: string,
  denylist: string[],
  label: string = DEFAULT_DENY_LABEL,
): Entity[] {
  const spans: Entity[] = [];
  for (const term of denylist) {
    if (!term) continue;
    const re = new RegExp(`(?<!\\w)${escapeRegExp(term)}(?!\\w)`, "gi");
    for (const m of text.matchAll(re)) {
      spans.push({
        start: m.index,
        end: m.index + m[0].length,
        label,
        text: m[0],
        source: "denylist",
      });
    }
  }
  return spans;
}

/**
 * Add denylist spans to a detected entity list. A deny span wins any
 * overlap: a detected span it touches is dropped. Overlapping deny
 * matches keep the first in document order.
 */
export function injectDenylist(
  text: string,
  entities: Entity[],
  denylist?: string[],
  label: string = DEFAULT_DENY_LABEL,
): Entity[] {
  if (!denylist || denylist.length === 0) return entities;
  const deny = denylistSpans(text, denylist, label);
  if (deny.length === 0) return entities;
  const kept = entities.filter((e) => !deny.some((d) => overlaps(e, d)));
  const merged: Entity[] = [];
  for (const d of [...deny].sort((a, b) => a.start - b.start)) {
    if (merged.some((k) => overlaps(d, k))) continue;
    merged.push(d);
  }
  return [...kept, ...merged].sort((a, b) => a.start - b.start);
}

/**
 * Allowlist filter, then denylist injection. Deny wins over allow: a
 * term in both lists is redacted (as a deny span).
 */
export function processEntities(
  text: string,
  entities: Entity[],
  options: RedactOptions = {},
): Entity[] {
  let out = applyAllowlist(entities, options.allowlist);
  out = injectDenylist(text, out, options.denylist,
                       options.denyLabel ?? DEFAULT_DENY_LABEL);
  return out;
}

// ---------------------------------------------------------------------------
// Custom recognizers (0.5.0, Presidio parity)
// ---------------------------------------------------------------------------

/** Named checksum validators a recognizer can reference by string. */
export const NAMED_VALIDATORS: Record<
  ValidatorName,
  (value: string) => boolean
> = {
  luhn: luhnOk,
  ssn: ssnValid,
  iban: ibanValid,
  aba: abaRoutingOk,
};

const round4 = (x: number): number => Math.round(x * 10000) / 10000;

function validateRecognizer(rec: Recognizer): void {
  if (!rec.label) throw new Error("recognizer label must be non-empty");
  if (!rec.patterns || rec.patterns.length === 0) {
    throw new Error("recognizer needs at least one pattern");
  }
  const score = rec.score ?? 0.85;
  if (!(score >= 0 && score <= 1)) {
    throw new Error(`recognizer score ${score} outside [0, 1]`);
  }
  if ((rec.contextWindow ?? 40) < 0) {
    throw new Error("contextWindow must be >= 0");
  }
  if ((rec.contextBoost ?? 0.1) < 0) {
    throw new Error("contextBoost must be >= 0");
  }
  if (rec.requireContext && (!rec.context || rec.context.length === 0)) {
    throw new Error("requireContext=true needs context words");
  }
  if (
    typeof rec.validator === "string" &&
    !(rec.validator in NAMED_VALIDATORS)
  ) {
    throw new Error(
      `unknown validator ${JSON.stringify(rec.validator)}; expected a ` +
        `function or one of ${Object.keys(NAMED_VALIDATORS).join(", ")}`,
    );
  }
  if ((rec.group ?? 0) < 0) throw new Error("group must be >= 0");
}

function resolveValidator(
  validator: Recognizer["validator"],
): ((value: string) => boolean) | undefined {
  if (validator === undefined || validator === null) return undefined;
  if (typeof validator === "function") return validator;
  const fn = NAMED_VALIDATORS[validator as ValidatorName];
  if (!fn) {
    throw new Error(
      `unknown validator ${JSON.stringify(validator)}; expected a ` +
        `function or one of ${Object.keys(NAMED_VALIDATORS).join(", ")}`,
    );
  }
  return fn;
}

function contextPresent(
  text: string,
  s: number,
  e: number,
  rec: Recognizer,
): boolean {
  const w = rec.contextWindow ?? 40;
  const lo = Math.max(0, s - w);
  const hi = Math.min(text.length, e + w);
  const win = (text.slice(lo, s) + text.slice(e, hi)).toLowerCase();
  return (rec.context ?? []).some(
    (word) => word !== "" && win.includes(word.toLowerCase()),
  );
}

/**
 * Run user recognizers over text. Returns scored spans (source
 * "recognizer") in PRECEDENCE order — recognizer registration order,
 * then pattern order, then document order — with overlaps among them
 * already resolved first-wins in that order. mergeEntities relies on
 * this order; the final entity list is sorted by start there.
 */
export function recognizerSpans(
  text: string,
  recognizers: Recognizer[] = [],
): Entity[] {
  const accepted: Entity[] = [];
  for (const rec of recognizers) {
    validateRecognizer(rec);
    const validate = resolveValidator(rec.validator);
    const baseScore = rec.score ?? 0.85;
    const grp = rec.group ?? 0;
    for (const pattern of rec.patterns) {
      // Fresh RegExp per scan: /g regexes are stateful (lastIndex).
      const re = new RegExp(pattern, "g");
      for (const m of text.matchAll(re)) {
        const value = m[grp];
        if (value === undefined || value === "") continue;
        const start =
          (m.index ?? 0) + (grp ? m[0].indexOf(value) : 0);
        const end = start + value.length;
        if (validate && !validate(value)) continue;
        let score = baseScore;
        if (rec.context && rec.context.length > 0) {
          if (contextPresent(text, start, end, rec)) {
            score = Math.min(1.0, score + (rec.contextBoost ?? 0.1));
          } else if (rec.requireContext) {
            continue;
          }
        }
        const span: Entity = {
          start,
          end,
          label: rec.label,
          text: value,
          source: "recognizer",
          score: round4(score),
        };
        if (accepted.some((a) => overlaps(span, a))) continue;
        accepted.push(span);
      }
    }
  }
  return accepted;
}

// ---------------------------------------------------------------------------
// Score gate: per-label thresholds + needs-review mode (0.5.0)
// ---------------------------------------------------------------------------

/** The score gate for one label: its per-label override when set,
    else the global threshold. */
export function effectiveThreshold(
  label: string,
  globalThreshold: number,
  thresholds?: Record<string, number>,
): number {
  if (thresholds && label in thresholds) return thresholds[label];
  return globalThreshold;
}

function validateScoreOptions(options: RedactOptions): void {
  if (options.thresholds) {
    for (const [label, t] of Object.entries(options.thresholds)) {
      if (!(t >= 0 && t <= 1)) {
        throw new Error(
          `threshold for ${JSON.stringify(label)} is ${t}, outside [0, 1]`,
        );
      }
    }
  }
  if (
    options.reviewMargin !== undefined &&
    options.reviewMargin !== null &&
    !(options.reviewMargin >= 0 && options.reviewMargin <= 1)
  ) {
    throw new Error(`reviewMargin ${options.reviewMargin} outside [0, 1]`);
  }
}

/**
 * Split merged entities into { kept, review } by score, against the
 * global gate DEFAULT_THRESHOLD (the local engine's threshold; the
 * neural layer's own gate lives server-side / in the Python engine).
 *
 * Only spans carrying a score are gated — neural spans and recognizer
 * spans. Spans WITHOUT a score (the checksum-gated rules layer,
 * denylist injections) are deterministic by construction and always
 * redact: a per-label threshold on "SSN" does not stop the SSN rule,
 * and a threshold never turns a rule span into a review candidate.
 *
 * A scored span redacts when score >= its effective threshold
 * (per-label override, else global). Below that, when reviewMargin is
 * set, a span scoring at or above (threshold - reviewMargin) becomes
 * a review candidate: NOT redacted, returned for a human to decide.
 * Anything lower is dropped. With reviewMargin undefined (the
 * default) the review list is always empty.
 */
export function partitionByScore(
  entities: Entity[],
  options: RedactOptions = {},
): { kept: Entity[]; review: ReviewCandidate[] } {
  validateScoreOptions(options);
  const kept: Entity[] = [];
  const review: ReviewCandidate[] = [];
  const margin = options.reviewMargin;
  for (const e of entities) {
    if (e.score === undefined || e.score === null) {
      kept.push(e);
      continue;
    }
    const eff = effectiveThreshold(e.label, DEFAULT_THRESHOLD,
                                   options.thresholds);
    if (e.score >= eff) {
      kept.push(e);
    } else if (
      margin !== undefined &&
      margin !== null &&
      e.score >= eff - margin
    ) {
      review.push({
        start: e.start,
        end: e.end,
        label: e.label,
        text: e.text,
        score: e.score,
        source: e.source,
      });
    }
  }
  review.sort((a, b) => a.start - b.start);
  return { kept, review };
}

// ---------------------------------------------------------------------------
// Redact / batch / scrub
// ---------------------------------------------------------------------------

/** Rules-only entity scan. */
export function scan(text: string): Entity[] {
  return rulesScan(text);
}

/** Rules-only redaction, with optional neural spans and options. */
export function redact(
  text: string,
  neuralEnts: Entity[] = [],
  options: RedactOptions = {},
): RedactionResult {
  const merged = mergeEntities(
    rulesScan(text),
    neuralEnts,
    recognizerSpans(text, options.recognizers ?? []),
  );
  const processed = processEntities(text, merged, options);
  const { kept, review } = partitionByScore(processed, options);
  const { redacted, mapping } = applyPlaceholders(text, kept, options);
  const counts: Record<string, number> = {};
  for (const e of kept) counts[e.label] = (counts[e.label] ?? 0) + 1;
  return {
    redacted,
    entities: kept,
    mapping,
    counts,
    engine: neuralEnts.length > 0 ? "rules+neural" : "rules",
    model: null,
    review,
  };
}

/**
 * Redact many texts, one RedactionResult per input, in input order —
 * the batch-pipeline entry point.
 */
export function redactBatch(
  texts: string[],
  options: RedactOptions = {},
): RedactionResult[] {
  return texts.map((t) => redact(t, [], options));
}

/**
 * Scrub text for a trip outside the trust boundary (e.g. a cloud LLM
 * call): returns { redacted, mapping }. Send the redacted text out,
 * keep the mapping local, restore the response with restore() — or
 * ChunkedRestorer when the response streams. Round-trips require the
 * default "numbered" operator; the other operators return an empty
 * mapping by design (they are irreversible).
 */
export function scrub(
  text: string,
  neuralEnts: Entity[] = [],
  options: RedactOptions = {},
): { redacted: string; mapping: Record<string, string> } {
  const r = redact(text, neuralEnts, options);
  return { redacted: r.redacted, mapping: r.mapping };
}

/** Put real values back using a mapping from redact(). */
export function restore(
  redacted: string,
  mapping: Record<string, string>,
): string {
  return restoreText(redacted, mapping);
}

// ---------------------------------------------------------------------------
// Streaming restore (LLM round-trips)
// ---------------------------------------------------------------------------

/**
 * Length of the longest buffer suffix that is a proper prefix of some
 * mapping key — text that might still grow into a placeholder when
 * the next chunk arrives, so it must not be emitted yet.
 */
function partialSuffixLen(buf: string, keys: string[]): number {
  if (keys.length === 0) return 0;
  const maxk = Math.max(...keys.map((k) => k.length));
  let best = 0;
  const limit = Math.min(buf.length, maxk - 1);
  for (let n = 1; n <= limit; n++) {
    const suffix = buf.slice(buf.length - n);
    if (keys.some((k) => k.length > n && k.startsWith(suffix))) best = n;
  }
  return best;
}

/**
 * Restore placeholders from text that arrives in chunks (e.g. a
 * streaming LLM response built on scrubbed input).
 *
 * feed() returns the restored text that is safe to emit now; a
 * trailing partial placeholder is buffered until it completes or
 * flush() proves it was never a placeholder. Placeholders split
 * across chunks are restored whole; unknown or partial placeholders
 * pass through untouched — feed/flush never throw.
 *
 *   const cr = new ChunkedRestorer(mapping);
 *   out = cr.feed(chunk1) + cr.feed(chunk2) + cr.flush();
 */
export class ChunkedRestorer {
  private readonly mapping: Record<string, string>;
  private buf = "";

  constructor(mapping: Record<string, string>) {
    this.mapping = { ...mapping };
  }

  feed(chunk: string): string {
    this.buf += chunk;
    const hold = partialSuffixLen(this.buf, Object.keys(this.mapping));
    let emit: string;
    if (hold > 0) {
      emit = this.buf.slice(0, this.buf.length - hold);
      this.buf = this.buf.slice(this.buf.length - hold);
    } else {
      emit = this.buf;
      this.buf = "";
    }
    return restoreText(emit, this.mapping);
  }

  flush(): string {
    const out = restoreText(this.buf, this.mapping);
    this.buf = "";
    return out;
  }
}
