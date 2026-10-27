// Tests for the 0.5.0 features (TypeScript SDK).
//
// Custom recognizers and per-label thresholds with needs-review mode.
// Mirrors python/tests/test_recognizers_thresholds.py. Synthetic
// fixtures only.

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  mergeEntities,
  recognizerSpans,
  redact,
  redactBatch,
  restore,
} from "../dist/index.js";

const BADGE = { label: "EMPLOYEE_ID", patterns: ["EMP\\d{6}"] };
const BADGE_TEXT = "Badge EMP123456 issued, call 415-555-0132.";

// ---------------------------------------------------------------------------
// Custom recognizers
// ---------------------------------------------------------------------------

describe("recognizers — basics", () => {
  it("custom label redacted with recognizer source and base score", () => {
    const r = redact(BADGE_TEXT, [], { recognizers: [BADGE] });
    assert.equal(r.redacted, "Badge [EMPLOYEE_ID_1] issued, call [PHONE_1].");
    const emp = r.entities.find((e) => e.label === "EMPLOYEE_ID");
    assert.equal(emp.source, "recognizer");
    assert.equal(emp.score, 0.85); // default base score
    assert.equal(emp.text, "EMP123456");
    assert.equal(restore(r.redacted, r.mapping), BADGE_TEXT);
  });

  it("multiple patterns in one recognizer", () => {
    const rec = {
      label: "EMPLOYEE_ID",
      patterns: ["EMP\\d{6}", "STF-\\d{4}"],
    };
    const r = redact("EMP123456 and STF-7788", [], { recognizers: [rec] });
    assert.equal(r.redacted, "[EMPLOYEE_ID_1] and [EMPLOYEE_ID_2]");
  });

  it("no recognizers, no change", () => {
    const r = redact(BADGE_TEXT);
    assert.equal(r.redacted, "Badge EMP123456 issued, call [PHONE_1].");
    assert.deepEqual(r.review, []);
  });

  it("group selects the value span", () => {
    const rec = {
      label: "BADGE_ID",
      group: 1,
      patterns: ["(?:badge|Badge)\\s+(\\d{6})"],
    };
    const r = redact("badge 482913 verified", [], { recognizers: [rec] });
    assert.equal(r.redacted, "badge [BADGE_ID_1] verified");
    assert.equal(r.entities[0].text, "482913");
  });

  it("first registered recognizer wins an overlap", () => {
    const a = { label: "SIX_DIGITS", patterns: ["\\b\\d{6}\\b"], score: 0.9 };
    const b = { label: "EXACT_CODE", patterns: ["482913"], score: 0.99 };
    assert.equal(
      redact("code 482913 here", [], { recognizers: [a, b] }).redacted,
      "code [SIX_DIGITS_1] here",
    );
    assert.equal(
      redact("code 482913 here", [], { recognizers: [b, a] }).redacted,
      "code [EXACT_CODE_1] here",
    );
  });

  it("empty matches are skipped", () => {
    const rec = { label: "DIGITS", patterns: ["\\d*"], score: 0.9 };
    const r = redact("abc 123 def", [], { recognizers: [rec] });
    assert.equal(r.redacted, "abc [DIGITS_1] def");
  });
});

describe("recognizers — operators", () => {
  it("label operator renders recognizer spans too", () => {
    const r = redact(BADGE_TEXT, [], {
      recognizers: [BADGE],
      operator: "label",
    });
    assert.equal(r.redacted, "Badge <EMPLOYEE_ID> issued, call <PHONE>.");
    assert.deepEqual(r.mapping, {});
  });

  it("hash operator is consistent for recognizer spans", () => {
    const a = redact("EMP123456", [], {
      recognizers: [BADGE],
      operator: "hash",
      salt: "s",
    });
    const b = redact("again EMP123456", [], {
      recognizers: [BADGE],
      operator: "hash",
      salt: "s",
    });
    assert.equal(a.redacted, b.redacted.split(" ").at(-1));
    assert.ok(a.redacted.startsWith("[EMPLOYEE_ID_"));
  });

  it("remove operator deletes recognizer spans", () => {
    const r = redact("EMP123456 gone", [], {
      recognizers: [BADGE],
      operator: "remove",
    });
    assert.equal(r.redacted, " gone");
  });
});

describe("recognizers — context", () => {
  const REC = {
    label: "CASE_ID",
    patterns: ["\\b\\d{6}\\b"],
    score: 0.5,
    context: ["case"],
    contextBoost: 0.4,
  };

  it("context boost raises the score over the threshold", () => {
    const r = redact("case 482913 open", [], { recognizers: [REC] });
    assert.equal(r.redacted, "case [CASE_ID_1] open");
    assert.equal(r.entities[0].score, 0.9);
  });

  it("no context stays below the global threshold", () => {
    const r = redact("number 482913 open", [], { recognizers: [REC] });
    assert.equal(r.redacted, "number 482913 open");
    assert.deepEqual(r.entities, []);
    assert.deepEqual(r.review, []);
  });

  it("context outside the window does not boost", () => {
    const rec = { ...REC, contextWindow: 5 };
    const r = redact("case number is far away: 482913", [], {
      recognizers: [rec],
    });
    assert.deepEqual(r.entities, []);
  });

  it("boost caps at 1.0", () => {
    const rec = { ...REC, score: 0.95 };
    const [span] = recognizerSpans("case 482913", [rec]);
    assert.equal(span.score, 1.0);
  });

  it("requireContext drops an uncontextualized match", () => {
    const rec = {
      label: "CASE_ID",
      patterns: ["\\b\\d{6}\\b"],
      score: 0.99,
      context: ["case"],
      requireContext: true,
    };
    assert.deepEqual(
      redact("number 482913 open", [], { recognizers: [rec] }).entities,
      [],
    );
    assert.equal(
      redact("case 482913 open", [], { recognizers: [rec] }).redacted,
      "case [CASE_ID_1] open",
    );
  });
});

describe("recognizers — validators", () => {
  it("callable validator gates matches", () => {
    const rec = {
      label: "ACCT_CODE",
      patterns: ["ACCT-[A-Z]{2}\\d{4}"],
      validator: (v) => v.endsWith("99"),
    };
    const r = redact("ACCT-XY1299 and ACCT-XY1234", [], {
      recognizers: [rec],
    });
    assert.equal(r.redacted, "[ACCT_CODE_1] and ACCT-XY1234");
  });

  it("named luhn validator", () => {
    const rec = {
      label: "CARD_LIKE",
      patterns: ["\\b\\d{16}\\b"],
      validator: "luhn",
    };
    const spans = recognizerSpans(
      "4111111111111111 vs 1234567812345678",
      [rec],
    );
    assert.deepEqual(spans.map((s) => s.text), ["4111111111111111"]);
  });

  it("named iban validator accepts a spaced IBAN, rejects a bad one", () => {
    const rec = {
      label: "IBAN",
      validator: "iban",
      patterns: ["[A-Z]{2}\\d{2}(?: \\d{4}){4} \\d{2}"],
    };
    const good = redact("IBAN DE89 3704 0044 0532 0130 00 end", [], {
      recognizers: [rec],
    });
    assert.equal(good.redacted, "IBAN [IBAN_1] end");
    const bad = redact("IBAN DE89 3704 0044 0532 0130 01 end", [], {
      recognizers: [rec],
    });
    // checksum fails -> the recognizer produces no IBAN span
    assert.deepEqual(bad.entities.filter((e) => e.label === "IBAN"), []);
  });

  it("invalid configs throw", () => {
    const bad = [
      { label: "", patterns: ["x"] },
      { label: "X", patterns: [] },
      { label: "X", patterns: ["x"], score: 1.5 },
      { label: "X", patterns: ["x"], requireContext: true },
      { label: "X", patterns: ["x"], validator: "nope" },
      { label: "X", patterns: ["x"], group: -1 },
    ];
    for (const rec of bad) {
      assert.throws(() => redact("x", [], { recognizers: [rec] }));
    }
  });
});

describe("recognizers — precedence and lists", () => {
  it("recognizer yields to an overlapping rule span", () => {
    const rec = {
      label: "EMPLOYEE_ID",
      patterns: ["EMP-\\d{5}"],
      score: 0.95,
    };
    const r = redact("Badge EMP-12345 issued.", [], { recognizers: [rec] });
    assert.equal(r.redacted, "Badge EMP-[ZIP_1] issued.");
  });

  it("mergeEntities: recognizer beats neural, rule beats recognizer", () => {
    const rule = { start: 0, end: 5, label: "ZIP", text: "90210", source: "rule" };
    const rec = {
      start: 3, end: 9, label: "EMPLOYEE_ID", text: "123456",
      source: "recognizer", score: 0.9,
    };
    const neural = {
      start: 3, end: 9, label: "NAME", text: "123456",
      source: "neural", score: 0.99,
    };
    assert.deepEqual(
      mergeEntities([rule], [neural], [rec]).map((e) => e.label),
      ["ZIP"],
    );
    assert.deepEqual(
      mergeEntities([], [neural], [rec]).map((e) => e.label),
      ["EMPLOYEE_ID"],
    );
  });

  it("allowlist drops a recognizer span", () => {
    const r = redact(BADGE_TEXT, [], {
      recognizers: [BADGE],
      allowlist: ["EMP123456"],
    });
    assert.equal(r.redacted, "Badge EMP123456 issued, call [PHONE_1].");
  });

  it("denylist wins an overlap with a recognizer span", () => {
    const r = redact("id EMP123456 ok", [], {
      recognizers: [BADGE],
      denylist: ["EMP123456"],
    });
    assert.equal(r.redacted, "id [CUSTOM_1] ok");
    assert.equal(r.entities[0].source, "denylist");
  });
});

// ---------------------------------------------------------------------------
// Per-label thresholds + needs-review
// ---------------------------------------------------------------------------

const REF = {
  label: "REF_CODE",
  patterns: ["[A-Z]{2}-\\d{4}"],
  score: 0.8,
};
const REF_TEXT = "ref ZX-8842 on file";

describe("thresholds", () => {
  it("per-label threshold lowers the gate", () => {
    const low = { label: "CASE_ID", patterns: ["\\b\\d{6}\\b"], score: 0.5 };
    const r = redact("code 482913 here", [], {
      recognizers: [low],
      thresholds: { CASE_ID: 0.4 },
    });
    assert.equal(r.redacted, "code [CASE_ID_1] here");
  });

  it("per-label threshold raises the gate and drops", () => {
    const r = redact(REF_TEXT, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.95 },
    });
    assert.equal(r.redacted, REF_TEXT);
    assert.deepEqual(r.entities, []);
    assert.deepEqual(r.review, []);
  });

  it("score exactly at the threshold redacts", () => {
    const r = redact(REF_TEXT, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.8 },
    });
    assert.equal(r.redacted, "ref [REF_CODE_1] on file");
  });

  it("other labels are unaffected by an override", () => {
    const r = redact("call 415-555-0132, ref ZX-8842", [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.95 },
    });
    assert.equal(r.redacted, "call [PHONE_1], ref ZX-8842");
  });

  it("invalid threshold values throw", () => {
    assert.throws(() => redact("x", [], { thresholds: { X: 1.5 } }));
    assert.throws(() => redact("x", [], { reviewMargin: -0.1 }));
  });
});

describe("review mode", () => {
  it("a span in the band becomes a review candidate", () => {
    const r = redact(REF_TEXT, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    });
    assert.equal(r.redacted, REF_TEXT); // NOT redacted
    assert.deepEqual(r.entities, []);
    assert.deepEqual(r.counts, {});
    assert.deepEqual(r.mapping, {});
    assert.deepEqual(r.review, [
      {
        start: 4, end: 11, label: "REF_CODE", text: "ZX-8842",
        score: 0.8, source: "recognizer",
      },
    ]);
  });

  it("a span below the floor is dropped, not reviewed", () => {
    const low = { ...REF, score: 0.5 };
    const r = redact(REF_TEXT, [], {
      recognizers: [low],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    });
    assert.deepEqual(r.review, []);
    assert.deepEqual(r.entities, []);
  });

  it("a span exactly at the floor is reviewed", () => {
    const r = redact(REF_TEXT, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.1,
    });
    assert.equal(r.review.length, 1);
  });

  it("global margin without per-label thresholds", () => {
    const low = { ...REF, score: 0.55 };
    const r = redact(REF_TEXT, [], {
      recognizers: [low],
      reviewMargin: 0.1,
    });
    assert.equal(r.review.length, 1);
    const r2 = redact(REF_TEXT, [], {
      recognizers: [low],
      reviewMargin: 0.04,
    });
    assert.deepEqual(r2.review, []);
    assert.deepEqual(r2.entities, []);
  });

  it("zero margin reviews nothing", () => {
    const r = redact(REF_TEXT, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.0,
    });
    assert.deepEqual(r.review, []);
    assert.deepEqual(r.entities, []);
  });

  it("review candidates are sorted by start", () => {
    const r = redact("QW-1093 then ZX-8842", [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    });
    assert.deepEqual(r.review.map((c) => c.text), ["QW-1093", "ZX-8842"]);
  });

  it("restore ignores review candidates", () => {
    const text = "call 415-555-0132, ref ZX-8842";
    const r = redact(text, [], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    });
    assert.equal(r.review.length, 1);
    assert.equal(restore(r.redacted, r.mapping), text);
  });

  it("rule spans are never gated", () => {
    // Documented edge: rule spans carry no score, so a per-label
    // threshold on SSN does not stop the checksum-gated SSN rule,
    // and the span never becomes a review candidate.
    const r = redact("SSN 234-56-7890", [], {
      thresholds: { SSN: 0.99 },
      reviewMargin: 0.5,
    });
    assert.equal(r.redacted, "SSN [SSN_1]");
    assert.deepEqual(r.review, []);
  });

  it("denylist spans are never gated", () => {
    const r = redact("Codral", [], {
      denylist: ["Codral"],
      thresholds: { CUSTOM: 0.99 },
      reviewMargin: 0.5,
    });
    assert.equal(r.redacted, "[CUSTOM_1]");
    assert.deepEqual(r.review, []);
  });

  it("batch results carry review per result", () => {
    const batch = redactBatch([REF_TEXT, "nothing here"], {
      recognizers: [REF],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    });
    assert.equal(batch[0].review.length, 1);
    assert.deepEqual(batch[1].review, []);
    assert.equal(batch[0].redacted, REF_TEXT);
  });
});

describe("thresholds and review on injected neural spans", () => {
  const TEXT = "Dr. Alice Ferreira reviewed Ana Souza and Bob Lindqvist.";
  const span = (name, score) => {
    const start = TEXT.indexOf(name);
    return {
      start, end: start + name.length, label: "NAME", text: name,
      source: "neural", score,
    };
  };
  const spans = () => [
    span("Alice Ferreira", 0.95),
    span("Ana Souza", 0.75),
    span("Bob Lindqvist", 0.45),
  ];

  it("neural spans partition by label threshold and margin", () => {
    const r = redact(TEXT, spans(), {
      thresholds: { NAME: 0.8 },
      reviewMargin: 0.3,
    });
    assert.ok(r.redacted.includes("[NAME_1]"));
    assert.ok(r.redacted.includes("Ana Souza")); // left for review
    assert.deepEqual(r.review.map((c) => c.text), ["Ana Souza"]);
    assert.ok(r.redacted.includes("Bob Lindqvist")); // below floor
    assert.ok(r.review.every((c) => c.text !== "Bob Lindqvist"));
  });

  it("recognizer beats neural in the full pipeline", () => {
    const text = "Alice Ferreira called.";
    const start = text.indexOf("Alice Ferreira");
    const neural = [{
      start, end: start + "Alice Ferreira".length, label: "NAME",
      text: "Alice Ferreira", source: "neural", score: 0.99,
    }];
    const rec = { label: "STAFF", patterns: ["Alice"], score: 0.9 };
    const r = redact(text, neural, { recognizers: [rec] });
    assert.equal(r.redacted, "[STAFF_1] Ferreira called.");
  });

  it("the default gate drops sub-0.6 neural spans, review empty", () => {
    const r = redact(TEXT, spans());
    assert.deepEqual(
      r.entities.map((e) => e.text),
      ["Alice Ferreira", "Ana Souza"],
    );
    assert.deepEqual(r.review, []);
  });
});
