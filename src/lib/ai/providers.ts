/** Server-only provider calls. Errors are classified; keys and user data are never logged. */
import { AI_CONFIG } from "@/config/ai";

export type ProviderErrorKind = "rate" | "quota" | "server" | "timeout" | "auth" | "network" | "bad-output";
export class ProviderError extends Error {
  constructor(public provider: string, public kind: ProviderErrorKind, public status?: number, public retryAfterMs?: number) {
    super(`${provider} ${kind}${status ? ` ${status}` : ""}`);
  }
}

export interface ProviderCall {
  (system: string, user: string): Promise<string>;
}

/** Parse "retry-after" seconds or Gemini's `retryDelay: "12s"` from an error body. */
export function parseRetryAfter(header: string | null | undefined, body?: string): number | undefined {
  if (header) {
    const s = Number(header);
    if (Number.isFinite(s)) return s * 1000;
    const d = Date.parse(header);
    if (Number.isFinite(d)) return Math.max(0, d - Date.now());
  }
  const m = body?.match(/retryDelay["'\s:]+"?(\d+(?:\.\d+)?)s/i) ?? body?.match(/retry in (\d+(?:\.\d+)?)s/i);
  return m ? Number(m[1]) * 1000 : undefined;
}

function classify(provider: string, status: number, body: string, retryAfterMs?: number): ProviderError {
  if (status === 429) {
    // Daily quota exhaustion is different from a per-minute limit.
    const daily = /per.?day|daily|quota.*exceeded.*day|RESOURCE_EXHAUSTED.*day/i.test(body);
    return new ProviderError(provider, daily ? "quota" : "rate", status, retryAfterMs);
  }
  if (status === 401 || status === 403) return new ProviderError(provider, "auth", status);
  if (status >= 500) return new ProviderError(provider, "server", status);
  return new ProviderError(provider, "server", status);
}

export function log(entry: { provider: string; status?: number; type: string; retryAfterMs?: number }) {
  // Only metadata — never keys, prompts or user data.
  console.info("[tutor]", JSON.stringify(entry));
}

export function geminiProvider(apiKey: string): ProviderCall {
  return async (system, user) => {
    const { GoogleGenAI, ThinkingLevel } = await import("@google/genai");
    const ai = new GoogleGenAI({ apiKey });
    try {
      const res = await ai.models.generateContent({
        model: AI_CONFIG.gemini.model,
        contents: user,
        config: {
          systemInstruction: system,
          maxOutputTokens: AI_CONFIG.gemini.maxOutputTokens,
          temperature: AI_CONFIG.gemini.temperature,
          responseMimeType: "application/json",
          thinkingConfig: { thinkingLevel: ThinkingLevel[AI_CONFIG.gemini.thinkingLevel] },
          httpOptions: { timeout: AI_CONFIG.timeoutMs },
          abortSignal: AbortSignal.timeout(AI_CONFIG.timeoutMs),
        },
      });
      return res.text ?? "";
    } catch (e) {
      const err = e as { status?: number; message?: string; name?: string };
      if (err.name === "AbortError" || err.name === "TimeoutError" || /timed? ?out|abort/i.test(err.message ?? "")) throw new ProviderError("gemini", "timeout");
      if (typeof err.status === "number") throw classify("gemini", err.status, err.message ?? "", parseRetryAfter(null, err.message));
      throw new ProviderError("gemini", "network");
    }
  };
}

export function groqProvider(apiKey: string, fetchImpl: typeof fetch = fetch): ProviderCall {
  return async (system, user) => {
    let res: Response;
    try {
      res = await fetchImpl(AI_CONFIG.groq.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: AI_CONFIG.groq.model,
          messages: [{ role: "system", content: system }, { role: "user", content: user }],
          max_tokens: AI_CONFIG.groq.maxOutputTokens,
          temperature: AI_CONFIG.groq.temperature,
          response_format: { type: "json_object" },
        }),
        signal: AbortSignal.timeout(AI_CONFIG.timeoutMs),
      });
    } catch (e) {
      throw new ProviderError("groq", (e as Error).name === "TimeoutError" ? "timeout" : "network");
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw classify("groq", res.status, body, parseRetryAfter(res.headers.get("retry-after"), body));
    }
    const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    return json.choices?.[0]?.message?.content ?? "";
  };
}
