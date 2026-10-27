export type EntitySource = "rule" | "neural" | "denylist";

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
}
