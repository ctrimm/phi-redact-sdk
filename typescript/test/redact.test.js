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

describe("scan edge cases", () => {
  it("returns [] for empty and clean text", () => {
    assert.deepEqual(scan(""), []);
    assert.deepEqual(scan("nothing to see here, just words"), []);
  });
  it("finds URL, IP, IBAN, ZIP", () => {
    const ents = scan(
      "see https://example.com/x from 10.0.0.1, " +
        "IBAN GB29NWBK60161331926819, ZIP 90210",
    );
    const byLabel = Object.fromEntries(ents.map((e) => [e.label, e.text]));
    assert.equal(byLabel.URL, "https://example.com/x");
    assert.equal(byLabel.IP, "10.0.0.1");
    assert.equal(byLabel.IBAN, "GB29NWBK60161331926819");
    assert.equal(byLabel.ZIP, "90210");
  });
  it("marks every entity with source 'rule'", () => {
    for (const e of scan(NOTE)) assert.equal(e.source, "rule");
  });
});

describe("redact edge cases", () => {
  it("leaves text without entities untouched", () => {
    const r = redact("nothing to see here");
    assert.equal(r.redacted, "nothing to see here");
    assert.deepEqual(r.mapping, {});
    assert.deepEqual(r.counts, {});
    assert.deepEqual(r.entities, []);
  });
  it("numbers interleaved labels independently in document order", () => {
    const r = redact("a 415-555-0132 b 234-56-7890 c 212-555-0199");
    assert.equal(r.redacted, "a [PHONE_1] b [SSN_1] c [PHONE_2]");
    assert.equal(r.mapping["[PHONE_2]"], "212-555-0199");
  });
  it("reports engine 'rules' and no model", () => {
    const r = redact("Call 415-555-0132");
    assert.equal(r.engine, "rules");
    assert.equal(r.model, null);
  });
  it("removes original values from the redacted text", () => {
    const r = redact("Call 415-555-0132");
    assert.ok(!r.redacted.includes("415-555-0132"));
  });
});

describe("VERSION", () => {
  it("is 0.2.0", async () => {
    const { VERSION } = await import("../dist/index.js");
    assert.equal(VERSION, "0.2.0");
  });
});

describe("phone boundaries", () => {
  it("consumes a leading paren or plus instead of leaving it behind", () => {
    const ents = scan("Reach her at (415) 555-0132 or +1 212-555-0199.");
    assert.deepEqual(
      ents.map((e) => e.text),
      ["(415) 555-0132", "+1 212-555-0199"],
    );
  });
  it("leaves no stray characters when redacting", () => {
    const r = redact("Call (415) 555-0132 or +1 212-555-0199.");
    assert.equal(r.redacted, "Call [PHONE_1] or [PHONE_2].");
    assert.deepEqual(r.mapping, {
      "[PHONE_1]": "(415) 555-0132",
      "[PHONE_2]": "+1 212-555-0199",
    });
  });
  it("does not match a phone glued to a word", () => {
    assert.ok(!scan("abc415-555-0132").some((e) => e.label === "PHONE"));
  });
});

describe("phone paren glued to word", () => {
  it("consumes the paren in x(312) 555-0167", () => {
    const r = redact("x(312) 555-0167");
    assert.equal(r.redacted, "x[PHONE_1]");
    assert.deepEqual(r.mapping, { "[PHONE_1]": "(312) 555-0167" });
  });
  it("matches nothing for x(abc)", () => {
    assert.deepEqual(scan("x(abc)"), []);
  });
});

describe("new deterministic rules (2026-10-19)", () => {
  it("abaRoutingOk gates on the checksum", async () => {
    const { abaRoutingOk } = await import("../dist/index.js");
    assert.equal(abaRoutingOk("021000021"), true);
    assert.equal(abaRoutingOk("494969870"), true);
    assert.equal(abaRoutingOk("123456789"), false);
    assert.equal(abaRoutingOk("809096606"), false); // plain SSN, not routing
    assert.equal(abaRoutingOk("12345678"), false);
  });
  it("finds TAX_ID, ROUTING_NUMBER, BANK_ACCOUNT, PASSPORT, DRIVERS_LICENSE, GOVERNMENT_ID", () => {
    const text =
      "EIN 18-6721452, routing 494969870, Account 713591432266, " +
      "passport Y12973198, DL D1315565, USCIS LIN2190012345.";
    const byLabel = {};
    for (const e of scan(text)) {
      (byLabel[e.label] ??= []).push(e.text);
      assert.equal(text.slice(e.start, e.end), e.text);
    }
    assert.deepEqual(byLabel["TAX_ID"], ["18-6721452"]);
    assert.deepEqual(byLabel["ROUTING_NUMBER"], ["494969870"]);
    assert.deepEqual(byLabel["BANK_ACCOUNT"], ["713591432266"]);
    assert.deepEqual(byLabel["PASSPORT"], ["Y12973198"]);
    assert.deepEqual(byLabel["DRIVERS_LICENSE"], ["D1315565"]);
    assert.deepEqual(byLabel["GOVERNMENT_ID"], ["LIN2190012345"]);
  });
  it("bank account span excludes the anchor word", () => {
    const text = "Account 713591432266 at First Bank.";
    const [e] = scan(text);
    assert.equal(e.label, "BANK_ACCOUNT");
    assert.equal(text.slice(e.start, e.end), "713591432266");
  });
  it("does not label non-checksum 9-digit numbers as routing", () => {
    assert.ok(
      !scan("Code 123456789 is not a routing number.").some(
        (e) => e.label === "ROUTING_NUMBER",
      ),
    );
  });
  it("keeps SSN, TAX_ID and DATE distinct", () => {
    const ents = scan("SSN 234-56-7890, EIN 18-6721452, DOB 03/14/1978.");
    const byLabel = {};
    for (const e of ents) (byLabel[e.label] ??= []).push(e.text);
    assert.deepEqual(byLabel["SSN"], ["234-56-7890"]);
    assert.deepEqual(byLabel["TAX_ID"], ["18-6721452"]);
    assert.deepEqual(byLabel["DATE"], ["03/14/1978"]);
  });
});

describe("bank account anchor variants (red-team 2026-10-19)", () => {
  it("matches acct, ACCT-, and 'account number' anchors", () => {
    for (const [text, expected] of [
      ["Acct #987654321098 for payroll.", "987654321098"],
      ["ACCT-112233445566 on file.", "112233445566"],
      ["Account number 112233445566 on file.", "112233445566"],
    ]) {
      const ents = scan(text).filter((e) => e.label === "BANK_ACCOUNT");
      assert.equal(ents.length, 1, text);
      assert.equal(ents[0].text, expected, text);
      assert.equal(text.slice(ents[0].start, ents[0].end), expected, text);
    }
  });
});
