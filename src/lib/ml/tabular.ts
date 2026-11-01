/**
 * Tabular preprocessing pipeline for the Data Lab. Fit on the TRAINING rows only
 * (no test leakage), then apply the same transform to test rows and to new rows
 * typed into the Predict form. The fitted spec is plain JSON so it can be saved
 * alongside an exported model.
 */
import { z } from "zod";
import { applyNormalizer, encodeCategory, fitCategoryEncoder, fitNormalizer, isMissing, meanOf, modeOf, type Cell, type MissingStrategy, type NormMethod } from "./data";
import { isMissingCell, parseNum, type ColumnType, type Table, type Task } from "./profile";

export interface PipelineOptions {
  target: string;
  features: string[];
  types: Record<string, ColumnType>;
  task: Task;
  norm: NormMethod;
  missing: MissingStrategy;
  maxCategories: number;
}

export const FittedPipelineSchema = z.object({
  task: z.enum(["classification", "regression"]),
  target: z.string(),
  features: z.array(
    z.object({
      name: z.string(),
      kind: z.enum(["numeric", "categorical"]),
      fill: z.union([z.number(), z.string(), z.null()]),
      categories: z.array(z.string()).optional(),
      hasOther: z.boolean().optional(),
    }),
  ),
  encodedNames: z.array(z.string()),
  norm: z.object({ method: z.enum(["none", "minmax", "zscore"]), shift: z.array(z.number()), scale: z.array(z.number()) }),
  classNames: z.array(z.string()).optional(),
  /** Regression targets are z-scored for stable training; predictions are mapped back. */
  yShift: z.number().default(0),
  yScale: z.number().default(1),
  missing: z.enum(["drop", "mean", "mode"]),
});
export type FittedPipeline = z.infer<typeof FittedPipelineSchema>;

export interface Encoded {
  X: number[][];
  y: number[]; // class index or normalised target
  rawY: string[];
  keptRows: number[]; // indices into the input rows that survived "drop"
}

function cellOf(raw: string | undefined, kind: "numeric" | "categorical"): Cell {
  if (isMissingCell(raw)) return null;
  if (kind === "numeric") {
    const v = parseNum(raw!);
    return Number.isFinite(v) ? v : null;
  }
  return raw!.trim();
}

/** Fit the pipeline on training rows. */
export function fitPipeline(t: Table, trainIdx: number[], o: PipelineOptions): FittedPipeline {
  const feats = o.features.filter((f) => o.types[f] === "numeric" || o.types[f] === "categorical");
  const colIdx = (n: string) => t.columns.indexOf(n);
  const ti = colIdx(o.target);
  const trainRows = trainIdx.map((i) => t.rows[i]).filter((r) => !isMissingCell(r[ti]));

  const features = feats.map((name) => {
    const kind = o.types[name] as "numeric" | "categorical";
    const vals = trainRows.map((r) => cellOf(r[colIdx(name)], kind));
    const fill = o.missing === "mean" && kind === "numeric" ? meanOf(vals) : o.missing === "drop" ? null : modeOf(vals);
    if (kind === "categorical") {
      const present = vals.filter((v): v is string => typeof v === "string");
      const enc = fitCategoryEncoder(present, o.maxCategories);
      return { name, kind, fill: fill ?? (o.missing === "drop" ? null : enc.categories[0] ?? ""), categories: enc.categories, hasOther: enc.hasOther };
    }
    return { name, kind, fill: fill ?? (o.missing === "drop" ? null : 0) };
  });

  const encodedNames = features.flatMap((f) =>
    f.kind === "numeric" ? [f.name] : [...f.categories!.map((c) => `${f.name}=${c}`), ...(f.hasOther ? [`${f.name}=other`] : [])],
  );

  const draft: FittedPipeline = {
    task: o.task, target: o.target, features, encodedNames,
    norm: { method: "none", shift: [], scale: [] },
    missing: o.missing, yShift: 0, yScale: 1,
  };
  if (o.task === "classification") {
    draft.classNames = [...new Set(trainRows.map((r) => r[ti].trim()))].sort();
  }
  // Fit normalisation on encoded numeric training data (one-hot columns are left as 0/1).
  const enc = encodeRows(t, trainIdx, draft, { skipNorm: true });
  const stats = fitNormalizer(enc.X, o.norm);
  let col = 0;
  for (const f of features) {
    const width = f.kind === "numeric" ? 1 : f.categories!.length + (f.hasOther ? 1 : 0);
    if (f.kind === "categorical") for (let k = 0; k < width; k++) { stats.shift[col + k] = 0; stats.scale[col + k] = 1; }
    col += width;
  }
  draft.norm = stats;
  if (o.task === "regression") {
    const ys = enc.rawY.map(parseNum);
    const m = ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length);
    const sd = Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, ys.length)) || 1;
    draft.yShift = m;
    draft.yScale = sd;
  }
  return draft;
}

/** Encode one raw row (record of column → string) into the model input vector, or null if it must be dropped. */
export function encodeFeatures(get: (name: string) => string | undefined, p: FittedPipeline, skipNorm = false): number[] | null {
  const out: number[] = [];
  for (const f of p.features) {
    let v = cellOf(get(f.name), f.kind);
    if (isMissing(v)) {
      if (p.missing === "drop" && f.fill === null) return null;
      v = f.fill;
    }
    if (f.kind === "numeric") out.push(typeof v === "number" ? v : Number(v) || 0);
    else out.push(...encodeCategory(String(v), { categories: f.categories!, hasOther: !!f.hasOther }));
  }
  return skipNorm || p.norm.shift.length === 0 ? out : applyNormalizer([out], p.norm)[0];
}

export function encodeRows(t: Table, idx: number[], p: FittedPipeline, opts: { skipNorm?: boolean } = {}): Encoded {
  const ti = t.columns.indexOf(p.target);
  const X: number[][] = [], y: number[] = [], rawY: string[] = [], keptRows: number[] = [];
  for (const i of idx) {
    const row = t.rows[i];
    const target = row[ti];
    if (isMissingCell(target)) continue; // rows without a label can't be used
    const x = encodeFeatures((n) => row[t.columns.indexOf(n)], p, opts.skipNorm);
    if (!x) continue;
    let yy: number;
    if (p.task === "classification") {
      yy = p.classNames?.indexOf(target.trim()) ?? -1;
      if (yy < 0) continue; // class unseen in training
    } else {
      yy = (parseNum(target) - p.yShift) / p.yScale;
      if (!Number.isFinite(yy)) continue;
    }
    X.push(x);
    y.push(yy);
    rawY.push(target.trim());
    keptRows.push(i);
  }
  return { X, y, rawY, keptRows };
}

export const denormY = (v: number, p: FittedPipeline) => v * p.yScale + p.yShift;

/** Map encoded column index → original feature index (for grouped permutation importance). */
export function featureGroups(p: FittedPipeline): number[][] {
  const groups: number[][] = [];
  let col = 0;
  for (const f of p.features) {
    const width = f.kind === "numeric" ? 1 : f.categories!.length + (f.hasOther ? 1 : 0);
    groups.push(Array.from({ length: width }, (_, k) => col + k));
    col += width;
  }
  return groups;
}
