/** Seeded pseudo-random utilities so every split and toy dataset is reproducible. */
import seedrandom from "seedrandom";

export type Rng = () => number;

export function makeRng(seed: number | string): Rng {
  return seedrandom(String(seed));
}

/** Standard normal sample via the Box–Muller transform. */
export function gaussian(rng: Rng, mean = 0, std = 1): number {
  let u = 0;
  while (u === 0) u = rng(); // avoid log(0)
  const v = rng();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** In-place Fisher–Yates shuffle. */
export function shuffleInPlace<T>(arr: T[], rng: Rng): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function permutation(n: number, rng: Rng): number[] {
  return shuffleInPlace(Array.from({ length: n }, (_, i) => i), rng);
}

/** Derive a stable 31-bit integer seed from another seed + a tag (for per-layer initialisers). */
export function deriveSeed(seed: number, tag: number): number {
  let h = (seed ^ 0x9e3779b9) + tag * 0x85ebca6b;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  return (h ^ (h >>> 16)) & 0x7fffffff;
}
