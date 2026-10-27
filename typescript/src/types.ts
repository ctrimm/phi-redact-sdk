export type EntitySource = "rule" | "neural" | "denylist" | "recognizer";

/**
 * Anonymize operator: how entity spans are rendered in the output.
 * Only "numbered" is reversible (it returns a mapping); the others
 * return an empty mapping by design.
 */
export type AnonymizeOperator =
  | "numbered"
  | "label"
  | "mask"
  | "hash"
  | "remove";

/** Named checksum validators a recognizer can reference by string. */
export type ValidatorName = "luhn" | "ssn" | "iban" | "aba";

/**
 * A user-defined recognizer (Presidio-style): regex patterns that
 * produce scored spans under a built-in or custom label.
 */
export interface Recognizer {
  /** Entity label for matches, e.g. "EMPLOYEE_ID" (uppercase by
      convention; new labels are allowed). */
  label: string;
  /** One or more regex source strings. The entity span is match
      group `group` (default 0 = whole match). Patterns should stick
      to the regex subset shared by JS RegExp and Python re. */
  patterns: string[];
  /** Base confidence for a match. Default 0.85. Gated by the
      effective threshold like every scored span. */
  score?: number;
  /** Context words: if any appears within `contextWindow` characters
      on either side of a match (the match itself excluded,
      case-insensitive substring), the score rises by `contextBoost`,
      capped at 1.0. */
  context?: string[];
  /** Characters of surrounding text searched for context words.
      Default 40 per side. */
  contextWindow?: number;
  /** Score increase on a context hit. Default 0.10. */
  contextBoost?: number;
  /** When true, a match with no context word nearby is dropped
      instead of kept at its base score. */
  requireContext?: boolean;
  /** Gate on the matched value: a predicate, or a named built-in
      checksum — "luhn" | "ssn" | "iban" | "aba". A value that fails
      the validator is not a match. */
  validator?: ValidatorName | ((value: string) => boolean);
  /** Regex group whose span/text is the entity. Use 1+ with an
      anchored pattern to exclude a keyword prefix. Default 0. */
  group?: number;
}

/**
 * A scored span held back by the threshold gate for human review:
 * NOT redacted, never in the mapping, never counted.
 */
export interface ReviewCandidate {
  start: number;
  end: number;
  label: string;
  text: string;
  score: number;
  source: EntitySource;
}

/** Options accepted by redact() / redactBatch() / scrub(). */
export interface RedactOptions {
  /** Placeholder style. Default "numbered". */
  operator?: AnonymizeOperator;
  /** Salt for the "hash" operator. Default "". Use a secret salt:
      unsalted hashes of low-entropy values (SSNs) are
      dictionary-attackable. */
  salt?: string;
  /** Terms that are never redacted (case-insensitive exact match). */
  allowlist?: string[];
  /** Terms that are always redacted (case-insensitive, word-boundary
      aware), labeled with denyLabel. Deny wins over allow. */
  denylist?: string[];
  /** Label for denylist spans. Default "CUSTOM". */
  denyLabel?: string;
  /** User-defined recognizers (Presidio-style patterns with scores,
      context boosting, and optional validators). */
  recognizers?: Recognizer[];
  /** Per-label score gates: a label's threshold overrides the global
      threshold (the SDK default, 0.6) for scored spans of that label
      (neural and recognizer spans). Spans without a score — rules and
      denylist spans — are never gated. */
  thresholds?: Record<string, number>;
  /** Needs-review band width: a scored span that misses its
      threshold but scores at or above (threshold - reviewMargin) is
      returned on result.review instead of being dropped. Undefined
      (the default) disables review mode. */
  reviewMargin?: number;
}

export interface Entity {
  start: number;
  end: number;
  label: string;
  text: string;
  source: EntitySource;
  /** Mean token confidence; present only for neural entities. */
  score?: number;
  /** Assigned by redact(), e.g. "[PHONE_1]". */
  placeholder?: string;
}

export interface RedactionResult {
  redacted: string;
  entities: Entity[];
  /** placeholder -> original text. Keep for pseudonymized round-trips;
      delete it and the copy is anonymized. */
  mapping: Record<string, string>;
  /** label -> count */
  counts: Record<string, number>;
  /** "rules" for the local engine, "rules+onnx" via RemoteRedactor. */
  engine: string;
  model: string | null;
  /** Scored spans held back by the threshold gate for human review.
      Empty unless reviewMargin is set. */
  review: ReviewCandidate[];
}
