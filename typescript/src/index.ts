/**
 * phi-redact TypeScript SDK.
 *
 * Local engine (rules-only, sync, zero dependencies):
 *
 *   import { redact, restore, scan } from "phi-redact";
 *   const r = redact("Call Jane Doe at 415-555-0132");
 *   // r.redacted, r.entities, r.mapping, r.counts
 *   restore(r.redacted, r.mapping); // original back
 *
 * Remote client (rules + ONNX neural layer via `redact serve`):
 *
 *   import { RemoteRedactor } from "phi-redact";
 *   const rr = new RemoteRedactor("http://127.0.0.1:8080");
 *   const r = await rr.redact(text, 0.5);
 *
 * This SDK redacts text. It is not a certified de-identification solution
 * and makes no HIPAA Expert Determination / Safe Harbor claim.
 */

export type { Entity, EntitySource, RedactionResult } from "./types.js";
export {
  ibanValid,
  luhnOk,
  redactText,
  restoreText,
  scan as scanRules,
  ssnValid,
} from "./rules.js";
export {
  applyPlaceholders,
  mergeEntities,
  redact,
  restore,
  scan,
} from "./engine.js";
export { RemoteRedactor } from "./remote.js";
export type { RemoteOptions } from "./remote.js";

export const VERSION = "0.1.0";
