import { describe, expect, it } from "vitest";
import {
  accuracy, classificationReport, confusionMatrix, downsample, logLoss, mae, majorityBaseline, meanBaseline, mse, r2, rmse,
} from "@/lib/ml/metrics";

describe("classification metrics (hand-computed)", () => {
  // true:  0 0 0 1 1 2
  // pred:  0 0 1 1 2 2
  const yTrue = [0, 0, 0, 1, 1, 2];
  const yPred = [0, 0, 1, 1, 2, 2];

  it("accuracy", () => {
    expect(accuracy(yTrue, yPred)).toBeCloseTo(4 / 6);
  });

  it("confusion matrix", () => {
    expect(confusionMatrix(yTrue, yPred, 3)).toEqual([
      [2, 1, 0],
      [0, 1, 1],
      [0, 0, 1],
    ]);
  });

  it("precision / recall / F1 per class and macro", () => {
    const r = classificationReport(yTrue, yPred, 3);
    // class 0: TP2 FP0 FN1 → P=1, R=2/3, F1=0.8
    expect(r.perClass[0].precision).toBeCloseTo(1);
    expect(r.perClass[0].recall).toBeCloseTo(2 / 3);
    expect(r.perClass[0].f1).toBeCloseTo(0.8);
    // class 1: TP1 FP1 FN1 → P=R=F1=0.5
    expect(r.perClass[1].f1).toBeCloseTo(0.5);
    // class 2: TP1 FP1 FN0 → P=0.5 R=1 F1=2/3
    expect(r.perClass[2].precision).toBeCloseTo(0.5);
    expect(r.perClass[2].recall).toBeCloseTo(1);
    expect(r.macro.f1).toBeCloseTo((0.8 + 0.5 + 2 / 3) / 3);
    expect(r.macro.precision).toBeCloseTo((1 + 0.5 + 0.5) / 3);
    expect(r.perClass.map((p) => p.support)).toEqual([3, 2, 1]);
  });

  it("handles a class never predicted (0/0 → 0)", () => {
    const r = classificationReport([0, 1], [0, 0], 2);
    expect(r.perClass[1].precision).toBe(0);
    expect(r.perClass[1].f1).toBe(0);
  });

  it("log loss", () => {
    expect(logLoss([0, 1], [[0.8, 0.2], [0.4, 0.6]])).toBeCloseTo(-(Math.log(0.8) + Math.log(0.6)) / 2);
    expect(Number.isFinite(logLoss([0], [[0, 1]]))).toBe(true);
  });
});

describe("regression metrics (hand-computed)", () => {
  const y = [3, -0.5, 2, 7];
  const yhat = [2.5, 0.0, 2, 8];
  it("MAE, MSE, RMSE", () => {
    expect(mae(y, yhat)).toBeCloseTo(0.5);
    expect(mse(y, yhat)).toBeCloseTo(0.375);
    expect(rmse(y, yhat)).toBeCloseTo(Math.sqrt(0.375));
  });
  it("R²", () => {
    expect(r2(y, yhat)).toBeCloseTo(0.9486081, 5);
    expect(r2([1, 1], [1, 1])).toBe(1);
    expect(r2([1, 2, 3], [2, 2, 2])).toBeCloseTo(0);
  });
});

describe("baselines", () => {
  it("majority class predictor", () => {
    expect(majorityBaseline([1, 1, 0, 2, 1], 3)).toEqual([1, 1, 1]);
    expect(majorityBaseline([0, 1], 1)).toEqual([0]); // tie → smallest id
  });
  it("mean predictor", () => {
    expect(meanBaseline([1, 2, 3, 6], 2)).toEqual([3, 3]);
  });
  it("a mean baseline has R² = 0 on its own training targets", () => {
    const t = [1, 4, 2, 9];
    expect(r2(t, meanBaseline(t, t.length))).toBeCloseTo(0);
  });
});

describe("downsample", () => {
  it("keeps ≤ max points including first and last", () => {
    const s = Array.from({ length: 101 }, (_, i) => i);
    const d = downsample(s, 20);
    expect(d.length).toBe(20);
    expect(d[0]).toBe(0);
    expect(d[19]).toBe(100);
  });
});
