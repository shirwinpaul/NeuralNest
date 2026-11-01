/** Pre-configured Playground setups, opened from Academy "Try it in the lab" buttons. */
import type { FeatureId, ToyKind } from "@/lib/ml/datasets/toy";
import type { Activation } from "@/lib/ml/builder";

export interface PlaygroundPreset {
  kind: ToyKind;
  noise: number;
  n: number;
  trainPct: number;
  features: FeatureId[];
  hidden: number[];
  activation: Activation;
  learningRate: number;
  l2: number;
  batchSize: number;
  optimizer: "sgd" | "adam";
}

export const DEFAULT_PRESET: PlaygroundPreset = {
  kind: "circles", noise: 0.1, n: 200, trainPct: 70, features: ["x1", "x2"], hidden: [4, 2],
  activation: "tanh", learningRate: 0.03, l2: 0, batchSize: 10, optimizer: "adam",
};

export const PLAYGROUND_PRESETS: Record<string, Partial<PlaygroundPreset>> = {
  spiral: { kind: "spirals", noise: 0.05, n: 300, hidden: [8, 8], activation: "tanh", learningRate: 0.03, features: ["x1", "x2", "sinx1", "sinx2"] },
  overfit: { kind: "moons", noise: 0.3, n: 40, trainPct: 50, hidden: [8, 8, 8], activation: "relu", learningRate: 0.03, l2: 0 },
  xor: { kind: "xor", hidden: [], features: ["x1", "x2"], learningRate: 0.1 },
  linear: { kind: "blobs", hidden: [], features: ["x1", "x2"], learningRate: 0.1 },
  neurons: { kind: "circles", hidden: [3], activation: "relu" },
  "lr-high": { kind: "circles", learningRate: 2.5, optimizer: "sgd" },
  "lr-low": { kind: "circles", learningRate: 0.0005, optimizer: "sgd" },
  batches: { kind: "moons", batchSize: 1, noise: 0.15 },
  regularise: { kind: "moons", noise: 0.3, n: 60, trainPct: 50, hidden: [8, 8, 8], l2: 0.01, activation: "relu" },
  regression: { kind: "sine", noise: 0.05, hidden: [8, 8], activation: "tanh", learningRate: 0.03 },
};

export function resolvePreset(id: string | null): PlaygroundPreset {
  return { ...DEFAULT_PRESET, ...(id ? PLAYGROUND_PRESETS[id] ?? {} : {}) };
}
