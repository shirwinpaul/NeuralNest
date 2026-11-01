/**
 * Per-IP sliding-window rate limit (in memory). Best-effort on serverless: each
 * instance has its own memory and may be recycled at any time.
 */
const hits = new Map<string, number[]>();
const WINDOW_MS = 60 * 60 * 1000;

export function checkRateLimit(ip: string, limit: number, now = Date.now()): { ok: true } | { ok: false; retryAfterSec: number } {
  const arr = (hits.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  if (arr.length >= limit) {
    hits.set(ip, arr);
    return { ok: false, retryAfterSec: Math.ceil((arr[0] + WINDOW_MS - now) / 1000) };
  }
  arr.push(now);
  hits.set(ip, arr);
  if (hits.size > 5000) for (const [k, v] of hits) if (!v.some((t) => now - t < WINDOW_MS)) hits.delete(k);
  return { ok: true };
}

export function resetRateLimit() {
  hits.clear();
}
