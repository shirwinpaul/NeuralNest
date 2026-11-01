/** Evaluation metrics — pure functions, everything computed in code. */

export function accuracy(yTrue: number[], yPred: number[]): number {
  if (yTrue.length !== yPred.length) throw new Error("Length mismatch");
  if (yTrue.length === 0) return 0;
  let ok = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) ok++;
  return ok / yTrue.length;
}

/** cm[trueClass][predictedClass] = count */
export function confusionMatrix(yTrue: number[], yPred: number[], numClasses: number): number[][] {
  const cm = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

export interface ClassScores { precision: number; recall: number; f1: number; support: number }
export interface ClassificationReport {
  accuracy: number;
  perClass: ClassScores[];
  macro: { precision: number; recall: number; f1: number };
  confusion: number[][];
}

const safeDiv = (a: number, b: number) => (b === 0 ? 0 : a / b);

/** Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = harmonic mean. 0/0 → 0. */
export function perClassScores(cm: number[][]): ClassScores[] {
  const k = cm.length;
  return Array.from({ length: k }, (_, c) => {
    const tp = cm[c][c];
    let fp = 0, fn = 0;
    for (let j = 0; j < k; j++) {
      if (j === c) continue;
      fp += cm[j][c];
      fn += cm[c][j];
    }
    const precision = safeDiv(tp, tp + fp);
    const recall = safeDiv(tp, tp + fn);
    const f1 = safeDiv(2 * precision * recall, precision + recall);
    return { precision, recall, f1, support: tp + fn };
  });
}

export function classificationReport(yTrue: number[], yPred: number[], numClasses: number): ClassificationReport {
  const confusion = confusionMatrix(yTrue, yPred, numClasses);
  const perClass = perClassScores(confusion);
  const mean = (f: (s: ClassScores) => number) => perClass.reduce((a, s) => a + f(s), 0) / (perClass.length || 1);
  return {
    accuracy: accuracy(yTrue, yPred),
    perClass,
    macro: { precision: mean((s) => s.precision), recall: mean((s) => s.recall), f1: mean((s) => s.f1) },
    confusion,
  };
}

/** Mean cross-entropy: −(1/n) Σ log p(true class). Probabilities are clipped to avoid log(0). */
export function logLoss(yTrue: number[], probs: number[][], eps = 1e-15): number {
  if (yTrue.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.min(1 - eps, Math.max(eps, probs[i][yTrue[i]]));
    s -= Math.log(p);
  }
  return s / yTrue.length;
}

// ---------- Regression ----------
export function mae(y: number[], yhat: number[]): number {
  return y.length ? y.reduce((a, v, i) => a + Math.abs(v - yhat[i]), 0) / y.length : 0;
}
export function mse(y: number[], yhat: number[]): number {
  return y.length ? y.reduce((a, v, i) => a + (v - yhat[i]) ** 2, 0) / y.length : 0;
}
export function rmse(y: number[], yhat: number[]): number {
  return Math.sqrt(mse(y, yhat));
}
/** R² = 1 − SS_res / SS_tot. With constant targets SS_tot = 0: perfect fit → 1, else 0. */
export function r2(y: number[], yhat: number[]): number {
  if (y.length === 0) return 0;
  const mean = y.reduce((a, b) => a + b, 0) / y.length;
  const ssTot = y.reduce((a, v) => a + (v - mean) ** 2, 0);
  const ssRes = y.reduce((a, v, i) => a + (v - yhat[i]) ** 2, 0);
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}

export interface RegressionReport { mae: number; mse: number; rmse: number; r2: number }
export function regressionReport(y: number[], yhat: number[]): RegressionReport {
  return { mae: mae(y, yhat), mse: mse(y, yhat), rmse: rmse(y, yhat), r2: r2(y, yhat) };
}

// ---------- Baselines ----------
/** Most frequent class in the training labels (ties → smallest class id). */
export function majorityClass(trainLabels: number[]): number {
  const counts = new Map<number, number>();
  for (const l of trainLabels) counts.set(l, (counts.get(l) ?? 0) + 1);
  let best = -1, bestN = -1;
  for (const [k, n] of counts) if (n > bestN || (n === bestN && k < best)) { best = k; bestN = n; }
  return best;
}
export function majorityBaseline(trainLabels: number[], nPredictions: number): number[] {
  return new Array(nPredictions).fill(majorityClass(trainLabels));
}
export function meanBaseline(trainTargets: number[], nPredictions: number): number[] {
  const m = trainTargets.length ? trainTargets.reduce((a, b) => a + b, 0) / trainTargets.length : 0;
  return new Array(nPredictions).fill(m);
}

/** Down-sample a series to at most `max` points (keeps first and last) — used for AI summaries. */
export function downsample<T>(series: T[], max = 20): T[] {
  if (series.length <= max) return series.slice();
  const out: T[] = [];
  for (let i = 0; i < max; i++) out.push(series[Math.round((i * (series.length - 1)) / (max - 1))]);
  return out;
}
