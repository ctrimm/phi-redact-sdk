import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  ibanValid,
  luhnOk,
  mergeEntities,
  redact,
  restore,
  scan,
  ssnValid,
} from "../dist/index.js";

const NOTE =
  "Patient Maria Santos, DOB 03/14/1978, SSN 234-56-7890. " +
  "Call 415-555-0132 or msantos@example.com. Card 4111111111111111 " +
  "charged; 1234567812345678 is not a real card. MRN 48291375.";

describe("validators", () => {
  it("luhn accepts valid cards, rejects random digit strings", () => {
    assert.equal(luhnOk("4111111111111111"), true);
    assert.equal(luhnOk("1234567812345678"), false);
    assert.equal(luhnOk("123"), false);
  });
  it("ssnValid enforces area/group/serial rules", () => {
    assert.equal(ssnValid("234-56-7890"), true);
    assert.equal(ssnValid("000-12-3456"), false);
    assert.equal(ssnValid("666-12-3456"), false);
    assert.equal(ssnValid("900-12-3456"), false);
    assert.equal(ssnValid("234-00-7890"), false);
  });
  it("ibanValid checks mod-97", () => {
    assert.equal(ibanValid("GB29NWBK60161331926819"), true);
    assert.equal(ibanValid("GB29NWBK60161331926818"), false);
  });
});

describe("scan", () => {
  it("finds structured entities and skips Luhn-invalid cards", () => {
    const ents = scan(NOTE);
    const labels = ents.map((e) => e.label).sort();
    assert.deepEqual(labels, [
      "CREDITCARD",
      "DATE",
      "EMAIL",
      "MRN",
      "PHONE",
      "SSN",
    ]);
    assert.ok(!ents.some((e) => e.text === "1234567812345678"));
  });
  it("returns spans sorted by start with exact text", () => {
    for (const e of scan(NOTE)) {
      assert.equal(NOTE.slice(e.start, e.end), e.text);
    }
  });
});

describe("redact", () => {
  it("replaces with forward-numbered placeholders", () => {
    const r = redact("Call 415-555-0132 or 212-555-0199");
    assert.equal(r.redacted, "Call [PHONE_1] or [PHONE_2]");
    assert.deepEqual(r.mapping, {
      "[PHONE_1]": "415-555-0132",
      "[PHONE_2]": "212-555-0199",
    });
    assert.deepEqual(r.counts, { PHONE: 2 });
    assert.equal(r.engine, "rules");
  });
  it("round-trips through restore", () => {
    const r = redact(NOTE);
    assert.equal(restore(r.redacted, r.mapping), NOTE);
  });
});

describe("mergeEntities", () => {
  it("rules win overlaps; neural-neural keeps best score", () => {
    const rule = {
      start: 5, end: 17, label: "PHONE",
      text: "415-555-0132", source: "rule",
    };
    const neuralOverlap = {
      start: 5, end: 10, label: "NAME_PATIENT",
      text: "415-5", source: "neural", score: 0.99,
    };
    const neuralA = {
      start: 0, end: 4, label: "NAME_PATIENT",
      text: "Jane", source: "neural", score: 0.6,
    };
    const neuralB = {
      start: 0, end: 4, label: "NAME_FAMILY",
      text: "Jane", source: "neural", score: 0.9,
    };
    const merged = mergeEntities([rule], [neuralOverlap, neuralA, neuralB]);
    assert.deepEqual(
      merged.map((e) => e.label),
      ["NAME_FAMILY", "PHONE"],
    );
  });
});
