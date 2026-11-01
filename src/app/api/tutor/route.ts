import { NextResponse, type NextRequest } from "next/server";
import { AI_CONFIG, isRealKey } from "@/config/ai";
import { mockProvider } from "@/lib/ai/mock";
import { geminiProvider, groqProvider, log } from "@/lib/ai/providers";
import { checkRateLimit } from "@/lib/ai/ratelimit";
import { TutorRequestSchema, type TutorResponse } from "@/lib/ai/schemas";
import { runTutor } from "@/lib/ai/tutor";

export const runtime = "nodejs";
// Vercel Hobby allows up to 60 s; one Gemini wait (≤10 s) + calls (≤25 s each) fit inside.
export const maxDuration = 60;

const json = (body: TutorResponse, status = 200, headers?: Record<string, string>) => NextResponse.json(body, { status, headers });

export async function POST(req: NextRequest) {
  const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || req.headers.get("x-real-ip") || "local";
  const rl = checkRateLimit(ip, AI_CONFIG.rateLimit.perHour);
  if (!rl.ok) return json({ ok: false, error: "rate-limited", retryAfterSec: rl.retryAfterSec }, 429, { "Retry-After": String(rl.retryAfterSec) });

  const text = await req.text();
  if (text.length > AI_CONFIG.maxRequestBytes) return json({ ok: false, error: "bad-request" }, 413);
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return json({ ok: false, error: "bad-request" }, 400);
  }
  const parsed = TutorRequestSchema.safeParse(body);
  if (!parsed.success) return json({ ok: false, error: "bad-request" }, 400);

  // MOCK_AI only works in development builds — never in production.
  if (process.env.NODE_ENV !== "production" && process.env.MOCK_AI === "true") {
    const r = await runTutor(parsed.data, { gemini: mockProvider, groq: null });
    return json(r.ok ? { ...r, provider: "mock" } : r);
  }

  const byok = req.headers.get("x-gemini-key")?.trim();
  const geminiKey = byok && byok.length < 200 ? byok : process.env.GEMINI_API_KEY;
  const groqKey = process.env.GROQ_API_KEY;
  const deps = {
    gemini: isRealKey(geminiKey) ? geminiProvider(geminiKey) : null,
    groq: isRealKey(groqKey) ? groqProvider(groqKey) : null,
  };
  if (!deps.gemini && !deps.groq) {
    log({ provider: "none", type: "no-keys" });
    return json({ ok: false, error: "offline" });
  }
  return json(await runTutor(parsed.data, deps));
}
