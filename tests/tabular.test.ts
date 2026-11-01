import { describe, expect, it } from "vitest";
import { classBalance, dataWarnings, detectTask, detectType, isImbalanced, pearson, profileTable, type Table } from "@/lib/ml/profile";
import { encodeFeatures, encodeRows, featureGroups, fitPipeline, FittedPipelineSchema } from "@/lib/ml/tabular";
import { permutationImportance } from "@/lib/ml/importance";
import { generateCricket, generateFruit, generateStudy } from "@/lib/ml/datasets/tabular";
import { makeRng } from "@/lib/ml/rng";
import { r2 } from "@/lib/ml/metrics";

describe("type detection", () => {
  it("numeric, categorical and text", () => {
    expect(detectType(["1", "2.5", "-3", "", "4e2"])).toBe("numeric");
    expect(detectType(["red", "blue", "red", "green"])).toBe("categorical");
    expect(detectType(Array.from({ length: 200 }, (_, i) => `note number ${i}`))).toBe("text");
    expect(detectType(["", "NA", "?"])).toBe("text");
  });
  it("task detection", () => {
    const t: Table = { columns: ["a"], rows: [["1"], ["0"], ["1"]] };
    expect(detectTask(profileTable(t).columns[0], ["1", "0", "1"])).toBe("classification");
    const t2: Table = { columns: ["a"], rows: [["1.5"], ["2.7"], ["9.1"]] };
    expect(detectTask(profileTable(t2).columns[0], ["1.5", "2.7", "9.1"])).toBe("regression");
  });
});

describe("profiling", () => {
  it("computes stats, missing counts and correlations", () => {
    const t: Table = { columns: ["x", "y", "c"], rows: [["1", "2", "a"], ["2", "4", "b"], ["3", "6", "a"], ["", "8", "a"]] };
    const p = profileTable(t);
    expect(p.columns[0].missing).toBe(1);
    expect(p.columns[0].mean).toBeCloseTo(2);
    expect(p.columns[2].top?.[0]).toEqual({ value: "a", count: 3 });
    expect(p.correlation.columns).toEqual(["x", "y"]);
    expect(p.correlation.matrix[0][1]).toBeCloseTo(1);
    expect(pearson([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1);
  });
});

describe("warnings", () => {
  const base = (): Table => {
    const rng = makeRng(1);
    const rows = Array.from({ length: 60 }, (_, i) => {
      const label = i < 54 ? "no" : "yes";
      return [String(i + 1), String(rng()), label, label];
    });
    return { columns: ["row_id", "x", "target", "copy_of_target"], rows };
  };
  it("flags ID columns, identical-to-target leakage and class imbalance", () => {
    const t = base();
    const kinds = dataWarnings(t, profileTable(t), "target", "classification").map((w) => `${w.kind}:${w.column ?? ""}`);
    expect(kinds).toContain("id-column:row_id");
    expect(kinds).toContain("identical-to-target:copy_of_target");
    expect(kinds).toContain("imbalance:target");
  });
  it("flags too few rows", () => {
    const t: Table = { columns: ["a", "b"], rows: [["1", "x"], ["2", "y"]] };
    expect(dataWarnings(t, profileTable(t), "b", "classification").some((w) => w.kind === "too-few-rows")).toBe(true);
  });
  it("imbalance detection", () => {
    expect(isImbalanced([{ count: 50 }, { count: 50 }])).toBe(false);
    expect(isImbalanced([{ count: 95 }, { count: 5 }])).toBe(true);
    expect(classBalance(["b", "a", "a", ""])).toEqual([{ label: "a", count: 2 }, { label: "b", count: 1 }]);
  });
  it("generated datasets don't trigger leakage warnings", () => {
    for (const [t, target, task] of [[generateStudy(), "exam_score", "regression"], [generateFruit(), "fruit", "classification"]] as const) {
      const w = dataWarnings(t, profileTable(t), target, task);
      expect(w.filter((x) => x.kind === "identical-to-target" || x.kind === "id-column")).toEqual([]);
    }
  });
});

describe("pipeline", () => {
  const t = generateCricket(200);
  const types = Object.fromEntries(profileTable(t).columns.map((c) => [c.name, c.type]));
  const opts = { target: "played", features: ["outlook", "temperature_c", "humidity_pct", "wind_kmh", "ground"], types, task: "classification" as const, norm: "zscore" as const, missing: "mean" as const, maxCategories: 10 };
  const train = Array.from({ length: 150 }, (_, i) => i);
  const test = Array.from({ length: 50 }, (_, i) => 150 + i);

  it("one-hot encodes categoricals and z-scores numerics fitted on train only", () => {
    const p = fitPipeline(t, train, opts);
    expect(p.encodedNames).toContain("outlook=sunny");
    expect(p.encodedNames).toContain("temperature_c");
    expect(p.classNames).toEqual(["no", "yes"]);
    const enc = encodeRows(t, train, p);
    const ti = p.encodedNames.indexOf("temperature_c");
    const col = enc.X.map((r) => r[ti]);
    expect(col.reduce((a, b) => a + b, 0) / col.length).toBeCloseTo(0, 5);
    // one-hot columns stay 0/1
    const oi = p.encodedNames.indexOf("outlook=sunny");
    expect(enc.X.every((r) => r[oi] === 0 || r[oi] === 1)).toBe(true);
    expect(encodeRows(t, test, p).X.length).toBe(50); // mean-fill keeps rows with missing humidity
    expect(FittedPipelineSchema.parse(JSON.parse(JSON.stringify(p)))).toEqual(p);
  });

  it("drop strategy removes rows with missing values; groups map one-hot columns", () => {
    const p = fitPipeline(t, train, { ...opts, missing: "drop" });
    const enc = encodeRows(t, [...train, ...test], p);
    const missingRows = t.rows.filter((r) => r[2] === "").length;
    expect(enc.X.length).toBe(200 - missingRows);
    const groups = featureGroups(p);
    expect(groups.length).toBe(5);
    expect(groups.flat().length).toBe(p.encodedNames.length);
  });

  it("encodes a form row (unseen category → other / zeros)", () => {
    const p = fitPipeline(t, train, opts);
    const vals: Record<string, string> = { outlook: "snowy", temperature_c: "25", humidity_pct: "", wind_kmh: "10", ground: "covered" };
    const x = encodeFeatures((n) => vals[n], p);
    expect(x).not.toBeNull();
    expect(x!.length).toBe(p.encodedNames.length);
  });

  it("regression targets are standardised and invertible", () => {
    const s = generateStudy(100);
    const st = Object.fromEntries(profileTable(s).columns.map((c) => [c.name, c.type]));
    const p = fitPipeline(s, Array.from({ length: 100 }, (_, i) => i), { target: "exam_score", features: ["study_hours", "sleep_hours"], types: st, task: "regression", norm: "minmax", missing: "mean", maxCategories: 10 });
    const enc = encodeRows(s, Array.from({ length: 100 }, (_, i) => i), p);
    expect(enc.y.reduce((a, b) => a + b, 0) / enc.y.length).toBeCloseTo(0, 5);
    expect(enc.y[0] * p.yScale + p.yShift).toBeCloseTo(Number(s.rows[0][4]), 5);
  });
});

describe("permutation importance", () => {
  it("identifies the informative feature on synthetic data", async () => {
    const rng = makeRng(2);
    const X = Array.from({ length: 300 }, () => [rng(), rng(), rng()]);
    const y = X.map((r) => 4 * r[1]); // only feature 1 matters
    const predict = (M: number[][]) => M.map((r) => 4 * r[1]); // a "perfect" model
    const res = await permutationImportance({ X, groups: [[0], [1], [2]], names: ["a", "b", "c"], score: (M) => r2(y, predict(M)), repeats: 3, seed: 1 });
    expect(res[0].feature).toBe("b");
    expect(res[0].importance).toBeGreaterThan(1);
    expect(Math.abs(res.find((r) => r.feature === "a")!.importance)).toBeLessThan(1e-9);
  });
});
