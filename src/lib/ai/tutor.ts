/**
 * Tutor orchestration (server side, provider-agnostic so it can be tested with mocks):
 *  Gemini → on 429 wait retry-after (≤ 10 s) and retry once → Groq fallback.
 *  Also falls back on 5xx, timeout, network errors, or invalid JSON after 1 corrective retry.
 */
import { AI_CONFIG } from "@/config/ai";
import { LESSON_IDS } from "@/data/lessons";
import { buildUserPrompt, SYSTEM_PROMPT } from "./prompts";
import { log, ProviderError, type ProviderCall } from "./providers";
import { OUTPUT_SCHEMAS, type TutorRequest, type TutorResponse } from "./schemas";

export interface TutorDeps {
  gemini?: ProviderCall | null;
  groq?: ProviderCall | null;
  sleep?: (ms: number) => Promise<void>;
}

const realSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function parseJson(text: string): unknown {
  const t = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "");
  return JSON.parse(t);
}

/** Validate a reply; strip lesson links that don't exist. */
function validate(req: TutorRequest, text: string): { ok: true; data: unknown } | { ok: false; error: string } {
  let raw: unknown;
  try {
    raw = parseJson(text);
  } catch {
    return { ok: false, error: "not valid JSON" };
  }
  const parsed = OUTPUT_SCHEMAS[req.mode].safeParse(raw);
  if (!parsed.success) return { ok: false, error: parsed.error.issues.slice(0, 3).map((i) => `${i.path.join(".")}: ${i.message}`).join("; ") };
  const data = parsed.data as Record<string, unknown>;
  if (req.mode === "explain") data.conceptLinks = (data.conceptLinks as string[]).filter((l) => LESSON_IDS.includes(l));
  if (req.mode === "ask" && data.conceptLink && !LESSON_IDS.includes(data.conceptLink as string)) data.conceptLink = null;
  return { ok: true, data };
}

/** One provider: call, validate, and retry once with the validation error appended. */
async function runProvider(name: string, call: ProviderCall, req: TutorRequest): Promise<unknown> {
  let user = buildUserPrompt(req.mode, req.summary, req.question);
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await call(SYSTEM_PROMPT, user);
    const v = validate(req, text);
    if (v.ok) return v.data;
    log({ provider: name, type: "invalid-json", status: 200 });
    user = buildUserPrompt(req.mode, req.summary, req.question, v.error);
  }
  throw new ProviderError(name, "bad-output");
}

export async function runTutor(req: TutorRequest, deps: TutorDeps): Promise<TutorResponse> {
  const sleep = deps.sleep ?? realSleep;
  let minuteLimitMs: number | undefined;
  const ok = (data: unknown, provider: string) => ({ ok: true, mode: req.mode, data, provider }) as TutorResponse;

  if (deps.gemini) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return ok(await runProvider("gemini", deps.gemini, req), "gemini");
      } catch (e) {
        const err = e instanceof ProviderError ? e : new ProviderError("gemini", "network");
        log({ provider: "gemini", status: err.status, type: err.kind, retryAfterMs: err.retryAfterMs });
        if (err.kind === "rate" && attempt === 0) {
          const wait = err.retryAfterMs ?? 2000;
          if (wait <= AI_CONFIG.maxRetryAfterWaitMs) {
            await sleep(wait);
            continue;
          }
          minuteLimitMs = wait;
        } else if (err.kind === "rate") {
          minuteLimitMs = err.retryAfterMs ?? minuteLimitMs;
        }
        break; // quota / 5xx / timeout / auth / bad output → fall back
      }
    }
  }

  if (deps.groq) {
    try {
      return ok(await runProvider("groq", deps.groq, req), "groq");
    } catch (e) {
      const err = e instanceof ProviderError ? e : new ProviderError("groq", "network");
      log({ provider: "groq", status: err.status, type: err.kind, retryAfterMs: err.retryAfterMs });
      if (err.kind === "rate") minuteLimitMs = Math.max(minuteLimitMs ?? 0, err.retryAfterMs ?? 30_000);
    }
  }

  if (minuteLimitMs !== undefined) return { ok: false, error: "busy", retryAfterSec: Math.max(1, Math.ceil(minuteLimitMs / 1000)) };
  return { ok: false, error: "offline" };
}
