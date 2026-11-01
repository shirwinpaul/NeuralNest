/**
 * Build the compact run summary sent to the tutor. Only aggregated numbers and
 * names leave the browser: no rows, pixels, images or embeddings.
 */
import { downsample } from "@/lib/ml/metrics";
import type { EpochLog } from "@/lib/ml/trainer";
import { RunSummarySchema, type RunSummary } from "./schemas";

const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
const finiteOnly = (m: Record<string, number | undefined>) =>
  Object.fromEntries(Object.entries(m).filter(([, v]) => typeof v === "number" && Number.isFinite(v)).map(([k, v]) => [k.slice(0, 40), r4(v as number)]));

export interface SummaryInput {
  lab: RunSummary["lab"];
  task: RunSummary["task"];
  architecture: string;
  params: number;
  hyperparams: Record<string, number | string | boolean>;
  dataset: RunSummary["dataset"];
  history: EpochLog[];
  metrics: Record<string, number | undefined>;
  baseline?: Record<string, number | undefined>;
  confusion?: number[][];
  notes?: string[];
}

export function buildRunSummary(i: SummaryInput): RunSummary {
  const s = {
    lab: i.lab,
    task: i.task,
    architecture: i.architecture.slice(0, 300),
    params: i.params,
    hyperparams: Object.fromEntries(Object.entries(i.hyperparams).slice(0, 20).map(([k, v]) => [k.slice(0, 40), typeof v === "string" ? v.slice(0, 40) : typeof v === "number" ? r4(v) : v])),
    dataset: {
      name: i.dataset.name.slice(0, 80),
      nTrain: i.dataset.nTrain,
      nTest: i.dataset.nTest,
      ...(i.dataset.classNames ? { classNames: i.dataset.classNames.slice(0, 10).map((c) => c.slice(0, 40)) } : {}),
      ...(i.dataset.classCounts ? { classCounts: i.dataset.classCounts.slice(0, 10) } : {}),
      ...(i.dataset.features ? { features: i.dataset.features.slice(0, 30).map((c) => c.slice(0, 60)) } : {}),
      ...(i.dataset.featureTypes ? { featureTypes: i.dataset.featureTypes.slice(0, 30) } : {}),
    },
    history: downsample(i.history, 20).map((h) => ({
      epoch: h.epoch,
      loss: Number.isFinite(h.loss) ? r4(h.loss) : null,
      ...(h.valLoss !== undefined && Number.isFinite(h.valLoss) ? { valLoss: r4(h.valLoss) } : {}),
      ...(h.acc !== undefined && Number.isFinite(h.acc) ? { acc: r4(h.acc) } : {}),
      ...(h.valAcc !== undefined && Number.isFinite(h.valAcc) ? { valAcc: r4(h.valAcc) } : {}),
    })),
    metrics: finiteOnly(i.metrics),
    ...(i.baseline ? { baseline: finiteOnly(i.baseline) } : {}),
    ...(i.confusion && i.confusion.length <= 10 ? { confusion: i.confusion.map((r) => r.slice(0, 10)) } : {}),
    ...(i.notes?.length ? { notes: i.notes.slice(0, 5).map((n) => n.slice(0, 160)) } : {}),
  };
  return RunSummarySchema.parse(s);
}
