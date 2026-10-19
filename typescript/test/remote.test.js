import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { RemoteRedactor } from "../dist/index.js";

/**
 * Integration tests for RemoteRedactor against the `redact serve` HTTP
 * contract. Uses an injected fetch stub instead of a live server, so the
 * suite stays hermetic while asserting the exact URLs, methods, and
 * bodies the client sends.
 */

function stubFetch(handler) {
  const calls = [];
  const fetchFn = async (url, init = {}) => {
    calls.push({ url, init });
    return handler(url, init);
  };
  return { fetchFn, calls };
}

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

describe("RemoteRedactor", () => {
  it("trims a trailing slash from baseUrl and GETs /health", async () => {
    const { fetchFn, calls } = stubFetch(async () => jsonResponse({ ok: true, engine: "rules" }));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080/", fetchFn });
    assert.deepEqual(await r.health(), { ok: true, engine: "rules" });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://x:8080/health");
    assert.equal(calls[0].init.method ?? "GET", "GET");
  });

  it("accepts a bare string as baseUrl", async () => {
    const { fetchFn, calls } = stubFetch(async () => jsonResponse({ ok: true, engine: "rules" }));
    const r = new RemoteRedactor("http://y:1234");
    // default fetch would hit the network; just check URL construction
    const r2 = new RemoteRedactor({ baseUrl: "http://y:1234", fetchFn });
    await r2.health();
    assert.equal(calls[0].url, "http://y:1234/health");
    assert.equal(r instanceof RemoteRedactor, true);
  });

  it("POSTs {text} to /redact and returns the parsed result", async () => {
    const result = {
      redacted: "Call [PHONE_1]",
      entities: [],
      mapping: { "[PHONE_1]": "415-555-0132" },
      counts: { PHONE: 1 },
      engine: "rules",
      model: null,
    };
    const { fetchFn, calls } = stubFetch(async () => jsonResponse(result));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    assert.deepEqual(await r.redact("Call 415-555-0132"), result);
    assert.equal(calls[0].url, "http://x:8080/redact");
    assert.equal(calls[0].init.method, "POST");
    assert.deepEqual(JSON.parse(calls[0].init.body), {
      text: "Call 415-555-0132",
    });
  });

  it("includes threshold in the body only when passed", async () => {
    const { fetchFn, calls } = stubFetch(async () => jsonResponse({ redacted: "x" }));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    await r.redact("hi", 0.9);
    assert.deepEqual(JSON.parse(calls[0].init.body), { text: "hi", threshold: 0.9 });
  });

  it("POSTs {redacted, mapping} to /restore and returns restored", async () => {
    const { fetchFn, calls } = stubFetch(async () =>
      jsonResponse({ restored: "Call 415-555-0132" }),
    );
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    const out = await r.restore("Call [PHONE_1]", {
      "[PHONE_1]": "415-555-0132",
    });
    assert.equal(out, "Call 415-555-0132");
    assert.equal(calls[0].url, "http://x:8080/restore");
    assert.deepEqual(JSON.parse(calls[0].init.body), {
      redacted: "Call [PHONE_1]",
      mapping: { "[PHONE_1]": "415-555-0132" },
    });
  });

  it("throws with the status on failed health checks", async () => {
    const { fetchFn } = stubFetch(async () => jsonResponse({ error: "nope" }, 500));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    await assert.rejects(() => r.health(), /health check failed: 500/);
  });

  it("throws with the status and body on failed redact", async () => {
    const { fetchFn } = stubFetch(async () => ({
      ok: false,
      status: 400,
      json: async () => ({}),
      text: async () => "{'error': 'bad text'}",
    }));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    await assert.rejects(() => r.redact("x"), /redact failed \(400\)/);
  });

  it("throws with the status on failed restore", async () => {
    const { fetchFn } = stubFetch(async () => ({
      ok: false,
      status: 400,
      json: async () => ({}),
      text: async () => "bad mapping",
    }));
    const r = new RemoteRedactor({ baseUrl: "http://x:8080", fetchFn });
    await assert.rejects(() => r.restore("x", {}), /restore failed \(400\)/);
  });
});
