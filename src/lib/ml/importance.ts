/**
 * Permutation feature importance (computed on the test set):
 * shuffle one feature's column(s) across rows, re-score the model, and measure how
 * much the score drops. A big drop means the model relies on that feature.
 * Columns belonging to one original feature (e.g. one-hot groups) are shuffled together.
 */
import { makeRng, permutation } from "./rng";

export interface ImportanceResult {
  feature: string;
  importance: number; // mean score drop (baseline − permuted); higher = more important
  std: number;
}

export async function permutationImportance(opts: {
  X: number[][];
  groups: number[][];
  names: string[];
  score: (X: number[][]) => Promise<number> | number; // higher is better
  repeats?: number;
  seed?: number;
}): Promise<ImportanceResult[]> {
  const { X, groups, names, score } = opts;
  const repeats = opts.repeats ?? 3;
  const rng = makeRng(opts.seed ?? 0);
  const base = await score(X);
  const out: ImportanceResult[] = [];
  for (let g = 0; g < groups.length; g++) {
    const drops: number[] = [];
    for (let r = 0; r < repeats; r++) {
      const perm = permutation(X.length, rng);
      const Xp = X.map((row, i) => {
        const copy = row.slice();
        for (const c of groups[g]) copy[c] = X[perm[i]][c];
        return copy;
      });
      drops.push(base - (await score(Xp)));
    }
    const mean = drops.reduce((a, b) => a + b, 0) / drops.length;
    const std = Math.sqrt(drops.reduce((a, b) => a + (b - mean) ** 2, 0) / drops.length);
    out.push({ feature: names[g], importance: mean, std });
  }
  return out.sort((a, b) => b.importance - a.importance);
}
