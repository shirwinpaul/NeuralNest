/**
 * ALL AI provider settings live here. Model IDs checked against the official docs
 * (ai.google.dev/gemini-api/docs/models + pricing, console.groq.com/docs/models) on 2026-10-24.
 * Both are free-tier models.
 */
export const AI_CONFIG = {
  gemini: {
    model: "gemini-3.6-flash",
    // Keep "thinking" to the minimum: the tutor explains numbers it is given, it doesn't need to reason long.
    thinkingLevel: "MINIMAL" as const,
    maxOutputTokens: 900,
    temperature: 0.4,
  },
  groq: {
    model: "llama-3.3-70b-versatile",
    endpoint: "https://api.groq.com/openai/v1/chat/completions",
    maxOutputTokens: 900,
    temperature: 0.4,
  },
  timeoutMs: 25_000,
  /** Wait at most this long for a Gemini retry-after before falling back. */
  maxRetryAfterWaitMs: 10_000,
  rateLimit: { perHour: 40 },
  /** Max request body accepted by /api/tutor (well under Vercel's ~4.5 MB limit). */
  maxRequestBytes: 16_000,
  questionMaxChars: 300,
} as const;

export const PLACEHOLDER_KEYS = ["PASTE_YOUR_GEMINI_KEY_HERE", "PASTE_YOUR_GROQ_KEY_HERE", ""];
export const isRealKey = (k: string | undefined): k is string => !!k && !PLACEHOLDER_KEYS.includes(k.trim());
