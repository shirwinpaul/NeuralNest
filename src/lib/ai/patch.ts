/**
 * AI-suggested hyperparameter changes are untrusted: validate and clamp them to the
 * lab's limits in code. Unknown keys are dropped. Applied only when the user taps "Try it".
 */
import { LIMITS, type LabKey } from "@/config/limits";
import { clamp, nearestBatch } from "@/lib/ml/builder";
import { ConfigPatchSchema, type ConfigPatch } from "./schemas";

const ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear"] as const;
const LAB_KEYS: Record<LabKey, (keyof ConfigPatch)[]> = {
  playground: ["learningRate", "batchSize", "l2", "hiddenLayers", "activation", "optimizer"],
  data: ["learningRate", "batchSize", "epochs", "l2", "dropout", "hiddenLayers", "activation"],
  doodle: ["learningRate", "epochs"],
  image: [],
};
const MAX_HIDDEN: Record<LabKey, number> = { playground: 4, data: 4, doodle: 0, image: 0 };

export type SafePatch = {
  learningRate?: number;
  batchSize?: number;
  epochs?: number;
  l2?: number;
  dropout?: number;
  hiddenLayers?: number[];
  activation?: (typeof ACTIVATIONS)[number];
  optimizer?: "adam" | "sgd";
};

export function sanitizePatch(lab: LabKey, raw: unknown): SafePatch | null {
  const parsed = ConfigPatchSchema.safeParse(raw);
  if (!parsed.success) return null;
  const p = parsed.data, L = LIMITS[lab], allowed = new Set(LAB_KEYS[lab]);
  const out: SafePatch = {};
  const ok = (k: keyof ConfigPatch) => allowed.has(k) && p[k] !== undefined && p[k] !== null;
  if (ok("learningRate") && Number.isFinite(p.learningRate)) out.learningRate = clamp(p.learningRate!, L.minLearningRate, L.maxLearningRate);
  if (ok("batchSize") && Number.isFinite(p.batchSize)) out.batchSize = nearestBatch(p.batchSize!, L.batchSizes);
  if (ok("epochs") && Number.isFinite(p.epochs)) out.epochs = Math.round(clamp(p.epochs!, 1, L.maxEpochs));
  if (ok("l2") && Number.isFinite(p.l2)) out.l2 = clamp(p.l2!, 0, L.maxL2);
  if (ok("dropout") && Number.isFinite(p.dropout)) out.dropout = clamp(p.dropout!, 0, L.maxDropout);
  if (ok("hiddenLayers")) {
    const h = p.hiddenLayers!.filter(Number.isFinite).slice(0, MAX_HIDDEN[lab]).map((u) => Math.round(clamp(u, 1, L.maxUnits)));
    out.hiddenLayers = h;
  }
  if (ok("activation") && (ACTIVATIONS as readonly string[]).includes(p.activation!.toLowerCase())) out.activation = p.activation!.toLowerCase() as SafePatch["activation"];
  if (ok("optimizer") && ["adam", "sgd"].includes(p.optimizer!.toLowerCase())) out.optimizer = p.optimizer!.toLowerCase() as "adam" | "sgd";
  return Object.keys(out).length ? out : null;
}

export function describePatch(p: SafePatch): string {
  return Object.entries(p).map(([k, v]) => `${k} = ${Array.isArray(v) ? `[${v.join(", ")}]` : v}`).join(", ");
}
