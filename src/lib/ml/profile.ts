/** Data profiling for the Data Lab: column types, stats, class balance, correlations, warnings. */

export interface Table {
  columns: string[];
  rows: string[][]; // raw cell strings
}

export type ColumnType = "numeric" | "categorical" | "text";

export interface ColumnProfile {
  name: string;
  type: ColumnType;
  missing: number;
  unique: number;
  // numeric
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
  median?: number;
  histogram?: { x0: number; x1: number; count: number }[];
  // categorical
  top?: { value: string; count: number }[];
}

export interface Warning {
  kind: "id-column" | "identical-to-target" | "leakage-correlation" | "imbalance" | "too-few-rows" | "many-missing" | "constant";
  column?: string;
  message: string;
}

export interface Profile {
  rows: number;
  columns: ColumnProfile[];
  correlation: { columns: string[]; matrix: number[][] };
}

const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-"]);
export const isMissingCell = (s: string | undefined) => s === undefined || MISSING.has(s.trim().toLowerCase());
export const parseNum = (s: string): number => {
  const t = s.trim().replace(/,/g, "");
  return t === "" ? NaN : Number(t);
};

/**
 * numeric: ≥ 95% of present cells parse as numbers.
 * categorical: few distinct values (≤ 20, or ≤ 5% of rows when big).
 * text: many distinct strings (free text / IDs) — ignored as a feature.
 */
export function detectType(values: string[]): ColumnType {
  const present = values.filter((v) => !isMissingCell(v));
  if (present.length === 0) return "text";
  const nums = present.filter((v) => Number.isFinite(parseNum(v))).length;
  if (nums / present.length >= 0.95) return "numeric";
  const unique = new Set(present.map((v) => v.trim())).size;
  const limit = Math.max(20, Math.round(present.length * 0.05));
  return unique <= Math.min(limit, 50) ? "categorical" : "text";
}

export function column(t: Table, name: string): string[] {
  const j = t.columns.indexOf(name);
  return t.rows.map((r) => r[j] ?? "");
}

function numericStats(values: number[]) {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  const mean = s.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  return { min: s[0], max: s[n - 1], mean, std, median };
}

export function histogram(values: number[], bins = 10): { x0: number; x1: number; count: number }[] {
  if (!values.length) return [];
  let mn = Infinity, mx = -Infinity;
  for (const v of values) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mn === mx) return [{ x0: mn, x1: mx, count: values.length }];
  const w = (mx - mn) / bins;
  const out = Array.from({ length: bins }, (_, i) => ({ x0: mn + i * w, x1: mn + (i + 1) * w, count: 0 }));
  for (const v of values) out[Math.min(bins - 1, Math.floor((v - mn) / w))].count++;
  return out;
}

/** Pearson correlation over rows where both values are present. */
export function pearson(a: number[], b: number[]): number {
  const pairs: [number, number][] = [];
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i]) && Number.isFinite(b[i])) pairs.push([a[i], b[i]]);
  const n = pairs.length;
  if (n < 2) return 0;
  const ma = pairs.reduce((s, p) => s + p[0], 0) / n, mb = pairs.reduce((s, p) => s + p[1], 0) / n;
  let cov = 0, va = 0, vb = 0;
  for (const [x, y] of pairs) { cov += (x - ma) * (y - mb); va += (x - ma) ** 2; vb += (y - mb) ** 2; }
  return va === 0 || vb === 0 ? 0 : cov / Math.sqrt(va * vb);
}

export function profileTable(t: Table): Profile {
  const columns: ColumnProfile[] = t.columns.map((name) => {
    const vals = column(t, name);
    const type = detectType(vals);
    const present = vals.filter((v) => !isMissingCell(v));
    const prof: ColumnProfile = { name, type, missing: vals.length - present.length, unique: new Set(present.map((v) => v.trim())).size };
    if (type === "numeric") {
      const nums = present.map(parseNum).filter(Number.isFinite);
      Object.assign(prof, numericStats(nums), { histogram: histogram(nums) });
    } else {
      const counts = new Map<string, number>();
      present.forEach((v) => counts.set(v.trim(), (counts.get(v.trim()) ?? 0) + 1));
      prof.top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([value, count]) => ({ value, count }));
    }
    return prof;
  });
  const numCols = columns.filter((c) => c.type === "numeric").map((c) => c.name).slice(0, 20);
  const numVals = numCols.map((c) => column(t, c).map((v) => (isMissingCell(v) ? NaN : parseNum(v))));
  const matrix = numVals.map((a) => numVals.map((b) => pearson(a, b)));
  return { rows: t.rows.length, columns, correlation: { columns: numCols, matrix } };
}

export type Task = "classification" | "regression";

/** Categorical target → classification; numeric with ≤ 10 distinct integers → classification; else regression. */
export function detectTask(p: ColumnProfile, values: string[]): Task {
  if (p.type !== "numeric") return "classification";
  const nums = values.filter((v) => !isMissingCell(v)).map(parseNum);
  const allInt = nums.every((v) => Number.isInteger(v));
  return allInt && p.unique <= 10 ? "classification" : "regression";
}

/** Class counts for a target column (sorted by class name). */
export function classBalance(values: string[]): { label: string; count: number }[] {
  const m = new Map<string, number>();
  values.filter((v) => !isMissingCell(v)).forEach((v) => m.set(v.trim(), (m.get(v.trim()) ?? 0) + 1));
  return [...m.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([label, count]) => ({ label, count }));
}

/** Imbalanced if the smallest class is < 1/5 of the largest, or < 10% of rows. */
export function isImbalanced(balance: { count: number }[]): boolean {
  if (balance.length < 2) return false;
  const counts = balance.map((b) => b.count);
  const total = counts.reduce((a, b) => a + b, 0);
  const mn = Math.min(...counts), mx = Math.max(...counts);
  return mn / mx < 0.2 || mn / total < 0.1;
}

const ID_NAME = /^(id|_id|index|row|row_?id|.*_id|uuid|key)$/i;

/** Warnings about leakage-like columns, imbalance, too few rows, missing data. */
export function dataWarnings(t: Table, p: Profile, target: string | null, task?: Task): Warning[] {
  const w: Warning[] = [];
  const n = t.rows.length;
  if (n < 30) w.push({ kind: "too-few-rows", message: `Only ${n} rows — results will be unreliable. Aim for at least 100.` });
  const targetVals = target ? column(t, target) : null;
  for (const c of p.columns) {
    if (c.name === target) continue;
    const vals = column(t, c.name);
    if (c.unique === 1) w.push({ kind: "constant", column: c.name, message: `"${c.name}" has the same value in every row — it can't help.` });
    // ID-like: name looks like an id, or every value is unique and (non-numeric or a 1-step sequence)
    const sequential = c.type === "numeric" && c.unique === n && isSequence(vals.map(parseNum));
    if (ID_NAME.test(c.name) || sequential || (c.type === "text" && c.unique === n - c.missing && n > 10))
      w.push({ kind: "id-column", column: c.name, message: `"${c.name}" looks like an ID column. IDs don't describe the example and can leak the answer if data was sorted — untick it.` });
    if (c.missing / Math.max(1, n) > 0.3) w.push({ kind: "many-missing", column: c.name, message: `"${c.name}" is ${Math.round((c.missing / n) * 100)}% missing.` });
    if (targetVals) {
      if (vals.every((v, i) => v.trim() === targetVals[i].trim()))
        w.push({ kind: "identical-to-target", column: c.name, message: `"${c.name}" is identical to the target — that's leakage. Untick it.` });
      else if (c.type === "numeric" && task === "regression") {
        const r = pearson(vals.map(parseNum), targetVals.map(parseNum));
        if (Math.abs(r) > 0.98) w.push({ kind: "leakage-correlation", column: c.name, message: `"${c.name}" is almost perfectly correlated with the target (r = ${r.toFixed(3)}). Check it isn't computed from the answer.` });
      }
    }
  }
  if (target && task === "classification" && targetVals) {
    const bal = classBalance(targetVals);
    if (isImbalanced(bal)) w.push({ kind: "imbalance", column: target, message: `Classes are imbalanced (${bal.map((b) => `${b.label}: ${b.count}`).join(", ")}). Accuracy can mislead — check recall per class and the baseline.` });
    if (bal.some((b) => b.count < 10)) w.push({ kind: "too-few-rows", column: target, message: "Some classes have fewer than 10 examples." });
  }
  return w;
}

function isSequence(v: number[]): boolean {
  if (v.length < 5 || v.some((x) => !Number.isFinite(x))) return false;
  const s = [...v].sort((a, b) => a - b);
  for (let i = 1; i < s.length; i++) if (s[i] - s[i - 1] !== 1) return false;
  return true;
}
