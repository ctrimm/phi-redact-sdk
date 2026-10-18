/**
 * HTTP client for `redact serve` (sdk/python + serve/cli.py).
 *
 * Use this when the ONNX neural layer is needed from Node: run
 * `serve/redact serve --port 8080` (optionally with --model) and point
 * the client at it. The local engine (index.ts) is rules-only, sync,
 * and dependency-free.
 */

import type { RedactionResult } from "./types.js";

export interface RemoteOptions {
  baseUrl?: string;
  /** Override fetch (tests, custom agents, non-Node runtimes). */
  fetchFn?: typeof fetch;
}

export class RemoteRedactor {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;

  constructor(options: RemoteOptions | string = {}) {
    const opts = typeof options === "string" ? { baseUrl: options } : options;
    this.baseUrl = (opts.baseUrl ?? "http://127.0.0.1:8080").replace(
      /\/$/,
      "",
    );
    this.fetchFn = opts.fetchFn ?? fetch;
  }

  async health(): Promise<{ ok: boolean; engine: string }> {
    const res = await this.fetchFn(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`health check failed: ${res.status}`);
    return (await res.json()) as { ok: boolean; engine: string };
  }

  /**
   * Redact via the server. Pass threshold to override the server's
   * neural confidence gate for this call.
   */
  async redact(text: string, threshold?: number): Promise<RedactionResult> {
    const res = await this.fetchFn(`${this.baseUrl}/redact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        threshold === undefined ? { text } : { text, threshold },
      ),
    });
    if (!res.ok) {
      const err = await res.text().catch(() => "");
      throw new Error(`redact failed (${res.status}): ${err}`);
    }
    return (await res.json()) as RedactionResult;
  }

  /** Restore originals using a mapping returned by redact(). */
  async restore(
    redacted: string,
    mapping: Record<string, string>,
  ): Promise<string> {
    const res = await this.fetchFn(`${this.baseUrl}/restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ redacted, mapping }),
    });
    if (!res.ok) {
      const err = await res.text().catch(() => "");
      throw new Error(`restore failed (${res.status}): ${err}`);
    }
    const body = (await res.json()) as { restored: string };
    return body.restored;
  }
}
