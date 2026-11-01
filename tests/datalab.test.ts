import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { buildModel, disposeModel } from "@/lib/ml/builder";
import { oneHot, splitIndices } from "@/lib/ml/data";
import { baselineClassification, baselineRegression, DEFAULT_NN, evaluateClassification, evaluateRegression, tabularModelConfig } from "@/lib/ml/datalab";
import { generateStudy } from "@/lib/ml/datasets/tabular";
import { profileTable, type Table } from "@/lib/ml/profile";
import { denormY, encodeRows, fitPipeline } from "@/lib/ml/tabular";
import { predictArray, TrainingRunner } from "@/lib/ml/trainer";

function iris(): Table {
  const lines = readFileSync(path.join(__dirname, "..", "public", "data", "iris.csv"), "utf8").trim().split(/\r?\n/);
  return { columns: lines[0].split(","), rows: lines.slice(1).map((l) => l.split(",")) };
}

describe("Data Lab end-to-end (CPU)", () => {
  it("Iris: logistic regression beats the majority baseline by a wide margin", async () => {
    const t = iris();
    expect(t.rows.length).toBe(150);
    const types = Object.fromEntries(profileTable(t).columns.map((c) => [c.name, c.type]));
    const names = ["setosa", "versicolor", "virginica"];
    const labels = t.rows.map((r) => names.indexOf(r[4]));
    const s = splitIndices(150, { train: 70, val: 15, test: 15 }, 42, labels);
    const p = fitPipeline(t, s.train, { target: "species", features: t.columns.slice(0, 4), types, task: "classification", norm: "zscore", missing: "mean", maxCategories: 10 });
    const tr = encodeRows(t, s.train, p), te = encodeRows(t, s.test, p);
    const base = baselineClassification(tr.y, te.y, 3);
    expect(base.metrics.accuracy).toBeLessThan(0.4);
    const model = buildModel(tabularModelConfig("linear", 4, "classification", 3, { ...DEFAULT_NN, epochs: 60 }, 42), LIMITS.data);
    const x = tf.tensor2d(tr.X), y = tf.tensor2d(oneHot(tr.y, 3)), xt = tf.tensor2d(te.X);
    await new TrainingRunner(model, x, y, { epochs: 60, batchSize: 32, seed: 42 }).run();
    const ev = evaluateClassification(te.y, await predictArray(model, xt), 3);
    expect(ev.metrics.accuracy).toBeGreaterThanOrEqual(0.85);
    expect(ev.confusion.flat().reduce((a, b) => a + b, 0)).toBe(te.y.length);
    tf.dispose([x, y, xt]);
    disposeModel(model);
  });

  it("Study hours: NN regression beats the mean baseline (R² > 0.6)", async () => {
    const t = generateStudy(300);
    const types = Object.fromEntries(profileTable(t).columns.map((c) => [c.name, c.type]));
    const s = splitIndices(300, { train: 70, val: 15, test: 15 }, 42);
    const p = fitPipeline(t, s.train, { target: "exam_score", features: ["study_hours", "sleep_hours", "practice_tests", "listens_to_music"], types, task: "regression", norm: "zscore", missing: "mean", maxCategories: 10 });
    const tr = encodeRows(t, s.train, p), te = encodeRows(t, s.test, p);
    const actual = te.y.map((v) => denormY(v, p));
    const base = baselineRegression(tr.y.map((v) => denormY(v, p)), actual);
    const model = buildModel(tabularModelConfig("nn", p.encodedNames.length, "regression", 1, { ...DEFAULT_NN, epochs: 60 }, 42), LIMITS.data);
    const x = tf.tensor2d(tr.X), y = tf.tensor2d(tr.y, [tr.y.length, 1]), xt = tf.tensor2d(te.X);
    await new TrainingRunner(model, x, y, { epochs: 60, batchSize: 32, seed: 42 }).run();
    const ev = evaluateRegression(actual, (await predictArray(model, xt)).map((r) => denormY(r[0], p)));
    expect(ev.metrics.r2).toBeGreaterThan(0.6);
    expect(ev.metrics.rmse).toBeLessThan(base.metrics.rmse);
    tf.dispose([x, y, xt]);
    disposeModel(model);
  });
});
