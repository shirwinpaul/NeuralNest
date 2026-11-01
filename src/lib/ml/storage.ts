/**
 * Local persistence: experiment tracker (localStorage, metadata + metrics only),
 * model metadata export/import, and storage usage. No raw data is ever stored here.
 */
import { z } from "zod";
import { downsample } from "./metrics";
import type { EpochLog } from "./trainer";

export const LabTypeSchema = z.enum(["playground", "data", "doodle", "image"]);
export type LabType = z.infer<typeof LabTypeSchema>;

const EpochLogSchema = z.object({
  epoch: z.number(),
  loss: z.number().nullable().transform((v) => v ?? NaN),
  acc: z.number().optional(),
  valLoss: z.number().optional(),
  valAcc: z.number().optional(),
});

export const RunRecordSchema = z.object({
  id: z.string(),
  lab: LabTypeSchema,
  name: z.string().max(120),
  createdAt: z.number(),
  durationMs: z.number(),
  backend: z.string(),
  architecture: z.string().max(500),
  params: z.number(),
  hyperparams: z.record(z.string(), z.union([z.number(), z.string(), z.boolean()])),
  dataset: z.object({
    name: z.string().max(120),
    nTrain: z.number(),
    nTest: z.number(),
    classNames: z.array(z.string().max(60)).max(20).optional(),
    features: z.array(z.string().max(80)).max(60).optional(),
  }),
  history: z.array(EpochLogSchema).max(60),
  metrics: z.record(z.string(), z.number()),
  starred: z.boolean().default(false),
});
export type RunRecord = z.infer<typeof RunRecordSchema>;
export type NewRun = Omit<RunRecord, "id" | "createdAt" | "starred" | "history"> & { history: EpochLog[] };

const RUNS_KEY = "nn.runs.v1";
export const MAX_RUNS = 100;
export const MAX_COMPARE = 4;

/** Minimal Storage interface so tests can inject an in-memory store. */
export interface KV {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
  removeItem(k: string): void;
}
export function memoryKV(): KV & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => void data.set(k, v), removeItem: (k) => void data.delete(k) };
}
let fallback: KV | null = null;
function defaultKV(): KV {
  try {
    if (typeof localStorage !== "undefined") return localStorage;
  } catch {
    /* blocked */
  }
  return (fallback ??= memoryKV());
}

function notify() {
  if (typeof window !== "undefined") window.dispatchEvent(new Event("nn-runs"));
}

export function listRuns(kv: KV = defaultKV()): RunRecord[] {
  try {
    const raw = kv.getItem(RUNS_KEY);
    if (!raw) return [];
    const arr = z.array(z.unknown()).parse(JSON.parse(raw));
    return arr.flatMap((r) => {
      const p = RunRecordSchema.safeParse(r);
      return p.success ? [p.data] : [];
    }).sort((a, b) => b.createdAt - a.createdAt);
  } catch {
    return [];
  }
}

function writeRuns(runs: RunRecord[], kv: KV) {
  kv.setItem(RUNS_KEY, JSON.stringify(runs));
  notify();
}

export function saveRun(run: NewRun, kv: KV = defaultKV()): RunRecord {
  const rec = RunRecordSchema.parse({
    ...run,
    id: `${run.lab}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
    createdAt: Date.now(),
    starred: false,
    history: downsample(run.history, 50).map((h) => ({ ...h, loss: Number.isFinite(h.loss) ? h.loss : null })),
  });
  const runs = [rec, ...listRuns(kv)];
  // Keep starred runs; trim the oldest unstarred ones past the cap.
  while (runs.length > MAX_RUNS) {
    const idx = runs.map((r) => r.starred).lastIndexOf(false);
    if (idx < 0) break;
    runs.splice(idx, 1);
  }
  try {
    writeRuns(runs, kv);
  } catch {
    // storage full → drop the older half of unstarred runs and retry once
    const kept = runs.filter((r, i) => r.starred || i < runs.length / 2);
    writeRuns(kept, kv);
  }
  return rec;
}

export function deleteRun(id: string, kv: KV = defaultKV()): void {
  writeRuns(listRuns(kv).filter((r) => r.id !== id), kv);
}

export function toggleStar(id: string, kv: KV = defaultKV()): void {
  writeRuns(listRuns(kv).map((r) => (r.id === id ? { ...r, starred: !r.starred } : r)), kv);
}

export function clearRuns(opts: { keepStarred: boolean }, kv: KV = defaultKV()): void {
  writeRuns(opts.keepStarred ? listRuns(kv).filter((r) => r.starred) : [], kv);
}

export interface Comparison {
  runs: RunRecord[];
  metricKeys: string[];
  /** Per metric, id of the best run (higher is better except losses/errors). */
  best: Record<string, string>;
}
const LOWER_IS_BETTER = /loss|mse|mae|rmse|error/i;

export function compareRuns(ids: string[], kv: KV = defaultKV()): Comparison {
  const all = listRuns(kv);
  const runs = ids.slice(0, MAX_COMPARE).map((id) => all.find((r) => r.id === id)).filter((r): r is RunRecord => !!r);
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.metrics)))].sort();
  const best: Record<string, string> = {};
  for (const k of metricKeys) {
    const lower = LOWER_IS_BETTER.test(k);
    let bestRun: RunRecord | null = null;
    for (const r of runs) {
      const v = r.metrics[k];
      if (v === undefined) continue;
      if (!bestRun || (lower ? v < bestRun.metrics[k] : v > bestRun.metrics[k])) bestRun = r;
    }
    if (bestRun) best[k] = bestRun.id;
  }
  return { runs, metricKeys, best };
}

// ---------- Export / import of all metadata ----------
export const BackupSchema = z.object({
  app: z.literal("neuralnest"),
  version: z.literal(1),
  exportedAt: z.number(),
  runs: z.array(RunRecordSchema),
  progress: z.unknown().optional(),
});

export function exportBackup(progress: unknown, kv: KV = defaultKV()): string {
  return JSON.stringify({ app: "neuralnest", version: 1, exportedAt: Date.now(), runs: listRuns(kv), progress }, null, 2);
}

export function importBackup(json: string, kv: KV = defaultKV()): { runs: number; progress: unknown } {
  const parsed = BackupSchema.parse(JSON.parse(json));
  const existing = listRuns(kv);
  const ids = new Set(existing.map((r) => r.id));
  const merged = [...existing, ...parsed.runs.filter((r) => !ids.has(r.id))].slice(0, MAX_RUNS);
  writeRuns(merged, kv);
  return { runs: parsed.runs.length, progress: parsed.progress };
}

// ---------- Model metadata (travels with exported model files) ----------
export const ModelMetaSchema = z.object({
  app: z.literal("neuralnest"),
  version: z.literal(1),
  name: z.string().min(1).max(80),
  lab: LabTypeSchema,
  task: z.enum(["classification", "regression"]),
  createdAt: z.number(),
  inputShape: z.array(z.number().int().positive()),
  classNames: z.array(z.string().max(60)).max(20).optional(),
  classEmojis: z.array(z.string().max(16)).max(20).optional(),
  features: z.array(z.string().max(80)).max(200).optional(),
  /** Normalisation: x' = (x − shift) / scale per encoded input column. */
  norm: z.object({ method: z.enum(["none", "minmax", "zscore"]), shift: z.array(z.number()), scale: z.array(z.number()) }).optional(),
  /** Data Lab: how to turn a raw form row into the model's input vector. */
  preprocess: z.unknown().optional(),
  /** Image Teacher: the head expects MobileNet embeddings. */
  featureExtractor: z.enum(["none", "mobilenet_v2_0.5"]).default("none"),
  metrics: z.record(z.string(), z.number()).default({}),
});
export type ModelMeta = z.infer<typeof ModelMetaSchema>;

export function serializeMeta(meta: ModelMeta): string {
  return JSON.stringify(ModelMetaSchema.parse(meta), null, 2);
}
export function parseMeta(json: string): ModelMeta {
  return ModelMetaSchema.parse(JSON.parse(json));
}

/** Rough bytes used by NeuralNest keys in localStorage. */
export function localStorageUsage(kv: KV & { length?: number; key?: (i: number) => string | null } = defaultKV()): number {
  let total = 0;
  try {
    const n = (kv as Storage).length ?? 0;
    for (let i = 0; i < n; i++) {
      const k = (kv as Storage).key(i);
      if (k && k.startsWith("nn.")) total += (k.length + (kv.getItem(k)?.length ?? 0)) * 2;
    }
  } catch {
    /* ignore */
  }
  return total;
}
