// Tests for the 0.4.0 Presidio-parity features (TypeScript SDK).
//
// Mirrors python/tests/test_parity_features.py. The hash vectors below
// are SHARED with the Python suite — they are what keeps the two SDKs'
// "hash" operator byte-identical. Synthetic fixtures only.

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  ChunkedRestorer,
  hashToken,
  redact,
  redactBatch,
  restore,
  scrub,
  sha256Hex,
} from "../dist/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const CLI = join(HERE, "..", "examples", "batch_jsonl.mjs");
const NOTE = "Call 415-555-0132 about SSN 234-56-7890.";

// [salt, label, value, expected token] — pinned in BOTH suites.
const HASH_VECTORS = [
  ["", "SSN", "234-56-7890", "[SSN_5f6c8b16]"],
  ["pepper", "SSN", "234-56-7890", "[SSN_e2e9f611]"],
  ["", "EMAIL", "jane@example.com", "[EMAIL_34da503d]"],
  ["s3cret-salt", "NAME_PATIENT", "Maria Santos", "[NAME_PATIENT_09a6b457]"],
  ["parity-salt", "PHONE", "415-555-0132", "[PHONE_954b7ae0]"],
];

describe("sha256 (pure TS, hash operator primitive)", () => {
  it("matches published known-answer vectors", () => {
    assert.equal(
      sha256Hex(""),
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    assert.equal(
      sha256Hex("abc"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    // 56-byte input crosses the padding boundary into a second block.
    assert.equal(
      sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
  });
});

describe("hash operator — shared cross-SDK vectors", () => {
  for (const [salt, label, value, expected] of HASH_VECTORS) {
    it(`hashToken(${label}, salt=${JSON.stringify(salt)})`, () => {
      assert.equal(hashToken(label, value, salt), expected);
    });
  }

  it("hash via redact; mapping empty (irreversible)", () => {
    const r = redact("SSN 234-56-7890", [], {
      operator: "hash",
      salt: "pepper",
    });
    assert.equal(r.redacted, "SSN [SSN_e2e9f611]");
    assert.deepEqual(r.mapping, {});
  });

  it("consistent across documents; salt changes the token", () => {
    const a = redact("SSN 234-56-7890", [], { operator: "hash", salt: "s" });
    const b = redact("again 234-56-7890 here", [], {
      operator: "hash",
      salt: "s",
    });
    assert.equal(a.redacted.split(" ").at(-1), b.redacted.split(" ")[1]);
    const c = redact("SSN 234-56-7890", [], { operator: "hash", salt: "x" });
    assert.notEqual(a.redacted, c.redacted);
  });
});

describe("operators", () => {
  it("label operator is Presidio-style and irreversible", () => {
    const r = redact(NOTE, [], { operator: "label" });
    assert.equal(r.redacted, "Call <PHONE> about SSN <SSN>.");
    assert.deepEqual(r.mapping, {});
    assert.deepEqual(r.counts, { PHONE: 1, SSN: 1 });
    assert.equal(r.entities[0].placeholder, "<PHONE>");
  });

  it("mask keeps last 4 for structured IDs, separators preserved", () => {
    const r = redact(
      "SSN 234-56-7890, card 4111111111111111, ph (415) 555-0132",
      [],
      { operator: "mask" },
    );
    assert.equal(
      r.redacted,
      "SSN ***-**-7890, card ************1111, ph (***) ***-0132",
    );
    assert.deepEqual(r.mapping, {});
  });

  it("mask falls back to same-length asterisks for other values", () => {
    const r = redact("ZIP 90210, see https://example.com/x", [], {
      operator: "mask",
    });
    assert.equal(r.redacted, "ZIP *****, see *********************");
  });

  it("remove deletes spans entirely", () => {
    const r = redact("Call 415-555-0132 now", [], { operator: "remove" });
    assert.equal(r.redacted, "Call  now");
    assert.deepEqual(r.mapping, {});
    assert.equal(r.entities[0].placeholder, "");
  });

  it("unknown operator throws", () => {
    assert.throws(() => redact(NOTE, [], { operator: "bogus" }));
  });
});

describe("allowlist", () => {
  it("allowlisted value is not redacted", () => {
    const r = redact(NOTE, [], { allowlist: ["415-555-0132"] });
    assert.equal(r.redacted, "Call 415-555-0132 about SSN [SSN_1].");
    assert.deepEqual(r.counts, { SSN: 1 });
  });

  it("matching is case-insensitive", () => {
    const r = redact("mail JANE@EXAMPLE.COM", [], {
      allowlist: ["jane@example.com"],
    });
    assert.equal(r.redacted, "mail JANE@EXAMPLE.COM");
  });
});

describe("denylist", () => {
  it("term redacted as CUSTOM with denylist source", () => {
    const r = redact("Patient takes Codral daily.", [], {
      denylist: ["Codral"],
    });
    assert.equal(r.redacted, "Patient takes [CUSTOM_1] daily.");
    assert.equal(r.entities.length, 1);
    assert.equal(r.entities[0].label, "CUSTOM");
    assert.equal(r.entities[0].source, "denylist");
    assert.equal(restore(r.redacted, r.mapping), "Patient takes Codral daily.");
  });

  it("denyLabel option picks the label", () => {
    const r = redact("Patient takes Codral daily.", [], {
      denylist: ["Codral"],
      denyLabel: "NAME_PATIENT",
    });
    assert.equal(r.redacted, "Patient takes [NAME_PATIENT_1] daily.");
  });

  it("case-insensitive and word-boundary aware", () => {
    const r = redact("codral and CODRAL, not Codraline.", [], {
      denylist: ["Codral"],
    });
    assert.equal(r.redacted, "[CUSTOM_1] and [CUSTOM_2], not Codraline.");
  });

  it("deny span wins overlap with a detected span", () => {
    const r = redact("MRN 48291375", [], { denylist: ["48291375"] });
    assert.equal(r.redacted, "MRN [CUSTOM_1]");
  });

  it("deny wins over allow", () => {
    const r = redact("Codral", [], {
      allowlist: ["Codral"],
      denylist: ["Codral"],
    });
    assert.equal(r.redacted, "[CUSTOM_1]");
  });

  it("no match is a no-op", () => {
    const r = redact(NOTE, [], { denylist: ["Zebra"] });
    assert.equal(r.redacted, redact(NOTE).redacted);
  });
});

describe("scrub / restore", () => {
  it("scrub returns the redacted text and mapping; restore round-trips", () => {
    const { redacted, mapping } = scrub(NOTE);
    const base = redact(NOTE);
    assert.equal(redacted, base.redacted);
    assert.deepEqual(mapping, base.mapping);
    assert.equal(restore(redacted, mapping), NOTE);
  });

  it("scrub with the label operator has an empty mapping", () => {
    const { redacted, mapping } = scrub(NOTE, [], { operator: "label" });
    assert.equal(redacted, "Call <PHONE> about SSN <SSN>.");
    assert.deepEqual(mapping, {});
  });

  it("restore handles many same-label placeholders", () => {
    const phones = Array.from(
      { length: 12 },
      (_, i) => `415-555-01${String(i).padStart(2, "0")}`,
    ).join(" ");
    const r = redact(phones);
    assert.equal(r.counts.PHONE, 12);
    assert.equal(restore(r.redacted, r.mapping), phones);
  });
});

describe("ChunkedRestorer", () => {
  const mapping = () => redact(NOTE).mapping;

  it("round-trips char by char", () => {
    const original = "The number is 415-555-0132 and SSN 234-56-7890.";
    const { redacted, mapping: m } = scrub(original);
    const cr = new ChunkedRestorer(m);
    let out = "";
    for (const c of redacted) out += cr.feed(c);
    out += cr.flush();
    assert.equal(out, original);
  });

  it("round-trips at every split point", () => {
    const original = "Call 415-555-0132.";
    const { redacted, mapping: m } = scrub(original);
    for (let i = 0; i <= redacted.length; i++) {
      const cr = new ChunkedRestorer(m);
      const out =
        cr.feed(redacted.slice(0, i)) + cr.feed(redacted.slice(i)) + cr.flush();
      assert.equal(out, original, `split at ${i}`);
    }
  });

  it("leaves unknown placeholders untouched", () => {
    const cr = new ChunkedRestorer(mapping());
    let out = cr.feed("ref [FOO_9] and [PHONE_") + cr.feed("1] done");
    out += cr.flush();
    assert.equal(out, "ref [FOO_9] and 415-555-0132 done");
  });

  it("leaves a partial placeholder at flush untouched", () => {
    const cr = new ChunkedRestorer(mapping());
    const out = cr.feed("ends with [PHO") + cr.flush();
    assert.equal(out, "ends with [PHO");
  });

  it("passes text through with an empty mapping", () => {
    const cr = new ChunkedRestorer({});
    assert.equal(cr.feed("plain [TEXT_1] here"), "plain [TEXT_1] here");
    assert.equal(cr.flush(), "");
  });
});

describe("redactBatch", () => {
  const TEXTS = [NOTE, "nothing to see here", "SSN 234-56-7890"];

  it("preserves order and matches single redact", () => {
    const batch = redactBatch(TEXTS);
    assert.deepEqual(
      batch.map((b) => b.redacted),
      TEXTS.map((t) => redact(t).redacted),
    );
    assert.deepEqual(batch[1].entities, []);
  });

  it("handles an empty list", () => {
    assert.deepEqual(redactBatch([]), []);
  });

  it("applies options to every item", () => {
    const batch = redactBatch(TEXTS, { operator: "label" });
    assert.equal(batch[0].redacted, "Call <PHONE> about SSN <SSN>.");
  });
});

describe("batch_jsonl example CLI", () => {
  it("stdin -> stdout, records without text pass through", () => {
    const input =
      '{"text": "Call 415-555-0132"}\n{"text": "clean"}\n{"id": 7}\n';
    const proc = spawnSync(process.execPath, [CLI], {
      input,
      encoding: "utf8",
      timeout: 60000,
    });
    assert.equal(proc.status, 0, proc.stderr);
    const rows = proc.stdout.trim().split("\n").map((l) => JSON.parse(l));
    assert.equal(rows[0].text, "Call [PHONE_1]");
    assert.deepEqual(rows[0].counts, { PHONE: 1 });
    assert.deepEqual(rows[1], { text: "clean", counts: {} });
    assert.deepEqual(rows[2], { id: 7 });
  });

  it("file args, --field and --operator", () => {
    const dir = mkdtempSync(join(tmpdir(), "phi-redact-batch-"));
    const src = join(dir, "in.jsonl");
    const dst = join(dir, "out.jsonl");
    writeFileSync(src, '{"note": "SSN 234-56-7890", "text": "x"}\n');
    const proc = spawnSync(
      process.execPath,
      [CLI, src, dst, "--field", "note", "--operator", "label"],
      { encoding: "utf8", timeout: 60000 },
    );
    assert.equal(proc.status, 0, proc.stderr);
    const rows = readFileSync(dst, "utf8")
      .trim()
      .split("\n")
      .map((l) => JSON.parse(l));
    assert.equal(rows[0].note, "SSN <SSN>");
    assert.deepEqual(rows[0].counts, { SSN: 1 });
  });
});
