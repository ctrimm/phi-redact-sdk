export type EntitySource = "rule" | "neural";

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
