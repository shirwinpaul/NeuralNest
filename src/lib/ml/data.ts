/** Framework-independent data utilities: splits, normalisation, encoding, missing values. */
import { makeRng, permutation, shuffleInPlace } from "./rng";

export interface SplitRatios { train: number; val: number; test: number }
export interface SplitIndices { train: number[]; val: number[]; test: number[] }

/**
 * Seeded train/validation/test split over row indices. When `labels` are given the
 * split is stratified: each class is split separately so class proportions match.
 */
export function splitIndices(n: number, ratios: SplitRatios, seed: number, labels?: number[]): SplitIndices {
  const total = ratios.train + ratios.val + ratios.test;
  if (total <= 0) throw new Error("Split ratios must be positive");
  const r = { train: ratios.train / total, val: ratios.val / total, test: ratios.test / total };
  const rng = makeRng(seed);
  const out: SplitIndices = { train: [], val: [], test: [] };

  const groups: number[][] = [];
  if (labels) {
    if (labels.length !== n) throw new Error("labels length must equal n");
    const byClass = new Map<number, number[]>();
    labels.forEach((c, i) => {
      if (!byClass.has(c)) byClass.set(c, []);
      byClass.get(c)!.push(i);
    });
    [...byClass.keys()].sort((a, b) => a - b).forEach((k) => groups.push(byClass.get(k)!));
  } else {
    groups.push(Array.from({ length: n }, (_, i) => i));
  }

  for (const g of groups) {
    shuffleInPlace(g, rng);
    const nTest = Math.round(g.length * r.test);
    const nVal = Math.round(g.length * r.val);
    out.test.push(...g.slice(0, nTest));
    out.val.push(...g.slice(nTest, nTest + nVal));
    out.train.push(...g.slice(nTest + nVal));
  }
  // Mix classes inside each split (still deterministic).
  shuffleInPlace(out.train, rng);
  shuffleInPlace(out.val, rng);
  shuffleInPlace(out.test, rng);
  return out;
}

export function take<T>(arr: T[], idx: number[]): T[] {
  return idx.map((i) => arr[i]);
}

export function shuffledIndices(n: number, seed: number): number[] {
  return permutation(n, makeRng(seed));
}

// ---------- Normalisation ----------
export type NormMethod = "none" | "minmax" | "zscore";
/** For each column: x' = (x - shift) / scale. */
export interface NormStats { method: NormMethod; shift: number[]; scale: number[] }

export function fitNormalizer(X: number[][], method: NormMethod): NormStats {
  const d = X[0]?.length ?? 0;
  const shift = new Array(d).fill(0);
  const scale = new Array(d).fill(1);
  if (method === "none" || X.length === 0) return { method, shift, scale };
  for (let j = 0; j < d; j++) {
    const col = X.map((r) => r[j]);
    if (method === "minmax") {
      let mn = Infinity, mx = -Infinity;
      for (const v of col) { if (v < mn) mn = v; if (v > mx) mx = v; }
      shift[j] = mn;
      scale[j] = mx - mn || 1; // constant column → leave unscaled
    } else {
      const mean = col.reduce((a, b) => a + b, 0) / col.length;
      const variance = col.reduce((a, b) => a + (b - mean) ** 2, 0) / col.length;
      shift[j] = mean;
      scale[j] = Math.sqrt(variance) || 1;
    }
  }
  return { method, shift, scale };
}

export function applyNormalizer(X: number[][], s: NormStats): number[][] {
  return X.map((r) => r.map((v, j) => (v - s.shift[j]) / s.scale[j]));
}

// ---------- Encoding ----------
export function oneHot(labels: number[], numClasses: number): number[][] {
  return labels.map((l) => {
    if (!Number.isInteger(l) || l < 0 || l >= numClasses) throw new Error(`Label ${l} out of range`);
    const row = new Array(numClasses).fill(0);
    row[l] = 1;
    return row;
  });
}

export function argmax(row: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/**
 * Category encoder with a cap: the (cap - 1) most frequent values keep their own
 * column, everything else (and unseen values) goes to "other" when capped.
 */
export interface CategoryEncoder { categories: string[]; hasOther: boolean }
export function fitCategoryEncoder(values: string[], cap: number): CategoryEncoder {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(([k]) => k);
  if (sorted.length <= cap) return { categories: sorted, hasOther: false };
  return { categories: sorted.slice(0, Math.max(1, cap - 1)), hasOther: true };
}
export function encodeCategory(v: string, enc: CategoryEncoder): number[] {
  const width = enc.categories.length + (enc.hasOther ? 1 : 0);
  const row = new Array(width).fill(0);
  const i = enc.categories.indexOf(v);
  if (i >= 0) row[i] = 1;
  else if (enc.hasOther) row[width - 1] = 1;
  return row;
}

// ---------- Missing values ----------
export type MissingStrategy = "drop" | "mean" | "mode";
export type Cell = number | string | null;

export function isMissing(v: Cell): boolean {
  return v === null || (typeof v === "number" && Number.isNaN(v)) || (typeof v === "string" && v.trim() === "");
}

/** Most frequent non-missing value (ties broken by first appearance). */
export function modeOf(values: Cell[]): Cell {
  const counts = new Map<string, { v: Cell; n: number; first: number }>();
  values.forEach((v, i) => {
    if (isMissing(v)) return;
    const k = String(v);
    const e = counts.get(k);
    if (e) e.n++;
    else counts.set(k, { v, n: 1, first: i });
  });
  let best: { v: Cell; n: number; first: number } | null = null;
  for (const e of counts.values()) if (!best || e.n > best.n || (e.n === best.n && e.first < best.first)) best = e;
  return best ? best.v : null;
}

export function meanOf(values: Cell[]): number {
  const nums = values.filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
}

/**
 * Apply a missing-value strategy to a table (rows of cells).
 * - drop: remove rows with any missing cell
 * - mean: numeric columns get the column mean, others get the mode
 * - mode: every column gets its most frequent value
 * `numeric[j]` says whether column j is numeric. Fill values are returned so the
 * same treatment can be applied to new rows at prediction time.
 */
export function handleMissing(rows: Cell[][], numeric: boolean[], strategy: MissingStrategy): { rows: Cell[][]; fill: Cell[] } {
  const d = numeric.length;
  if (strategy === "drop") return { rows: rows.filter((r) => !r.some(isMissing)), fill: new Array(d).fill(null) };
  const fill: Cell[] = [];
  for (let j = 0; j < d; j++) {
    const col = rows.map((r) => r[j]);
    fill.push(strategy === "mean" && numeric[j] ? meanOf(col) : modeOf(col));
  }
  return { rows: rows.map((r) => r.map((v, j) => (isMissing(v) ? fill[j] : v))), fill };
}
