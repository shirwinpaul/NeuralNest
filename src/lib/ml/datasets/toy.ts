/** Seeded 2D toy datasets for the Neural Playground. Coordinates lie roughly in [-1, 1]. */
import { gaussian, makeRng, type Rng } from "../rng";

export type ToyKind = "circles" | "xor" | "spirals" | "moons" | "blobs" | "sine";
export interface ToyPoint { x1: number; x2: number; label: number } // label: 0/1, or y value for "sine"
export interface ToyOptions { n: number; noise: number; seed: number }

export const TOY_DATASETS: { id: ToyKind; name: string; emoji: string; task: "classification" | "regression" }[] = [
  { id: "circles", name: "Circles", emoji: "⭕", task: "classification" },
  { id: "xor", name: "XOR", emoji: "❎", task: "classification" },
  { id: "spirals", name: "Two spirals", emoji: "🌀", task: "classification" },
  { id: "moons", name: "Two moons", emoji: "🌙", task: "classification" },
  { id: "blobs", name: "Gaussian blobs", emoji: "🫧", task: "classification" },
  { id: "sine", name: "Noisy sine (regression)", emoji: "〰️", task: "regression" },
];

const clampUnit = (v: number) => Math.max(-1.2, Math.min(1.2, v));

export function generateToy(kind: ToyKind, { n, noise, seed }: ToyOptions): ToyPoint[] {
  const rng = makeRng(`${kind}:${seed}`);
  const half = Math.floor(n / 2);
  const pts: ToyPoint[] = [];
  const jitter = (r: Rng) => gaussian(r, 0, noise);
  for (let i = 0; i < n; i++) {
    const label = i < half ? 0 : 1;
    switch (kind) {
      case "circles": {
        // inner disc (label 1) vs outer ring (label 0)
        const a = rng() * 2 * Math.PI;
        const r = label === 1 ? rng() * 0.4 : 0.65 + rng() * 0.3;
        pts.push({ x1: clampUnit(r * Math.cos(a) + jitter(rng)), x2: clampUnit(r * Math.sin(a) + jitter(rng)), label });
        break;
      }
      case "xor": {
        const x1 = rng() * 2 - 1, x2 = rng() * 2 - 1;
        const pad = (v: number) => v + (v > 0 ? 0.05 : -0.05); // keep points off the axes
        const l = x1 * x2 > 0 ? 1 : 0;
        pts.push({ x1: clampUnit(pad(x1) + jitter(rng)), x2: clampUnit(pad(x2) + jitter(rng)), label: l });
        break;
      }
      case "spirals": {
        // two interleaved arms, 1.75 turns each
        const t = (i % half) / half;
        const angle = t * 1.75 * 2 * Math.PI + (label ? Math.PI : 0);
        const r = 0.05 + t * 0.95;
        pts.push({ x1: clampUnit(r * Math.cos(angle) + jitter(rng)), x2: clampUnit(r * Math.sin(angle) + jitter(rng)), label });
        break;
      }
      case "moons": {
        const a = rng() * Math.PI;
        const x1 = label === 0 ? Math.cos(a) : 1 - Math.cos(a);
        const x2 = label === 0 ? Math.sin(a) : 0.5 - Math.sin(a);
        // rescale from [-1,2]×[-0.5,1] into ~[-1,1]
        pts.push({ x1: clampUnit((x1 - 0.5) / 1.5 + jitter(rng)), x2: clampUnit((x2 - 0.25) / 0.75 + jitter(rng)), label });
        break;
      }
      case "blobs": {
        const c = label === 0 ? [-0.45, -0.4] : [0.45, 0.4];
        pts.push({ x1: clampUnit(gaussian(rng, c[0], 0.22 + noise)), x2: clampUnit(gaussian(rng, c[1], 0.22 + noise)), label });
        break;
      }
      case "sine": {
        // regression: predict y from x1 (x2 ignored → 0)
        const x = rng() * 2 - 1;
        pts.push({ x1: x, x2: 0, label: Math.sin(Math.PI * x) * 0.8 + jitter(rng) });
        break;
      }
    }
  }
  return pts;
}

// ---------- Feature engineering ----------
export type FeatureId = "x1" | "x2" | "x1sq" | "x2sq" | "x1x2" | "sinx1" | "sinx2";
export const FEATURES: { id: FeatureId; label: string; fn: (a: number, b: number) => number }[] = [
  { id: "x1", label: "x₁", fn: (a) => a },
  { id: "x2", label: "x₂", fn: (_a, b) => b },
  { id: "x1sq", label: "x₁²", fn: (a) => a * a },
  { id: "x2sq", label: "x₂²", fn: (_a, b) => b * b },
  { id: "x1x2", label: "x₁·x₂", fn: (a, b) => a * b },
  { id: "sinx1", label: "sin(x₁)", fn: (a) => Math.sin(Math.PI * a) },
  { id: "sinx2", label: "sin(x₂)", fn: (_a, b) => Math.sin(Math.PI * b) },
];

export function featurize(x1: number, x2: number, features: FeatureId[]): number[] {
  return features.map((f) => FEATURES.find((d) => d.id === f)!.fn(x1, x2));
}
