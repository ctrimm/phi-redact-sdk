/**
 * Merge + placeholder layer (TypeScript port of sdk/python/phi_redact/engine.py).
 *
 * Local engine is rules-only. `mergeEntities` accepts neural spans so a
 * future onnxruntime-based layer can plug in without changing the API:
 * rules win every overlap, neural-neural overlaps keep the higher score.
 */

import { restoreText, scan as rulesScan } from "./rules.js";
import type { Entity, RedactionResult } from "./types.js";

const overlaps = (a: Entity, b: Entity): boolean =>
  !(a.end <= b.start || a.start >= b.end);

/**
 * Rules win every overlap. Neural-neural overlaps keep the best score.
 * Returns entities sorted by start.
 */
export function mergeEntities(
  ruleEnts: Entity[],
  neuralEnts: Entity[] = [],
): Entity[] {
  const kept: Entity[] = [];
  for (const n of [...neuralEnts].sort(
    (a, b) => (b.score ?? 0) - (a.score ?? 0),
  )) {
    if (ruleEnts.some((r) => overlaps(n, r))) continue;
    if (kept.some((k) => overlaps(n, k))) continue;
    kept.push(n);
  }
  return [...ruleEnts, ...kept].sort((a, b) => a.start - b.start);
}

/**
 * Replace back-to-front with temp tokens, then renumber per label in
 * forward document order: [NAME_PATIENT_1], [PHONE_2], ...
 */
export function applyPlaceholders(
  text: string,
  entities: Entity[],
): { redacted: string; mapping: Record<string, string> } {
  const temps = new Map<Entity, string>();
  const tempFor = (e: Entity): string => temps.get(e)!;
  let out = text;
  const staging: Record<string, string> = {};

  [...entities]
    .sort((a, b) => b.start - a.start)
    .forEach((e, i) => {
      const tmp = `[\0${i}]`;
      temps.set(e, tmp);
      staging[tmp] = e.text;
      out = out.slice(0, e.start) + tmp + out.slice(e.end);
    });

  const counters: Record<string, number> = {};
  const mapping: Record<string, string> = {};
  for (const e of [...entities].sort((a, b) => a.start - b.start)) {
    counters[e.label] = (counters[e.label] ?? 0) + 1;
    const ph = `[${e.label}_${counters[e.label]}]`;
    e.placeholder = ph;
    mapping[ph] = staging[tempFor(e)];
    out = out.split(tempFor(e)).join(ph);
  }
  return { redacted: out, mapping };
}

/** Rules-only entity scan. */
export function scan(text: string): Entity[] {
  return rulesScan(text);
}

/** Rules-only redaction. */
export function redact(text: string, neuralEnts: Entity[] = []): RedactionResult {
  const entities = mergeEntities(rulesScan(text), neuralEnts);
  const { redacted, mapping } = applyPlaceholders(text, entities);
  const counts: Record<string, number> = {};
  for (const e of entities) counts[e.label] = (counts[e.label] ?? 0) + 1;
  return {
    redacted,
    entities,
    mapping,
    counts,
    engine: neuralEnts.length > 0 ? "rules+neural" : "rules",
    model: null,
  };
}

/** Put real values back using a mapping from redact(). */
export function restore(
  redacted: string,
  mapping: Record<string, string>,
): string {
  return restoreText(redacted, mapping);
}
