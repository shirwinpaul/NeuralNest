import { describe, expect, it } from "vitest";
import {
  applyNormalizer, encodeCategory, fitCategoryEncoder, fitNormalizer, handleMissing, modeOf, oneHot, splitIndices,
} from "@/lib/ml/data";
import { gaussian, makeRng } from "@/lib/ml/rng";

describe("seeded split", () => {
  it("is reproducible and covers every index exactly once", () => {
    const a = splitIndices(100, { train: 0.7, val: 0.15, test: 0.15 }, 7);
    const b = splitIndices(100, { train: 0.7, val: 0.15, test: 0.15 }, 7);
    expect(a).toEqual(b);
    const all = [...a.train, ...a.val, ...a.test].sort((x, y) => x - y);
    expect(all).toEqual(Array.from({ length: 100 }, (_, i) => i));
    expect(a.test.length).toBe(15);
  });
  it("different seeds give different splits", () => {
    const a = splitIndices(100, { train: 0.8, val: 0, test: 0.2 }, 1);
    const b = splitIndices(100, { train: 0.8, val: 0, test: 0.2 }, 2);
    expect(a.test).not.toEqual(b.test);
  });
  it("stratifies by label", () => {
    // 80 of class 0, 20 of class 1
    const labels = [...new Array(80).fill(0), ...new Array(20).fill(1)];
    const s = splitIndices(100, { train: 0.75, val: 0, test: 0.25 }, 3, labels);
    const testOnes = s.test.filter((i) => labels[i] === 1).length;
    expect(testOnes).toBe(5);
    expect(s.test.length).toBe(25);
  });
});

describe("rng", () => {
  it("gaussian samples have ~zero mean, unit std", () => {
    const rng = makeRng(1);
    const xs = Array.from({ length: 5000 }, () => gaussian(rng));
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
    expect(Math.abs(m)).toBeLessThan(0.05);
    expect(Math.abs(sd - 1)).toBeLessThan(0.05);
  });
});

describe("normalisation", () => {
  const X = [[0, 10], [5, 10], [10, 10]];
  it("min-max scales to [0, 1] and leaves constant columns safe", () => {
    const s = fitNormalizer(X, "minmax");
    expect(applyNormalizer(X, s)).toEqual([[0, 0], [0.5, 0], [1, 0]]);
  });
  it("z-score gives mean 0, std 1", () => {
    const s = fitNormalizer(X, "zscore");
    const z = applyNormalizer(X, s).map((r) => r[0]);
    expect(z.reduce((a, b) => a + b, 0)).toBeCloseTo(0);
    expect(Math.sqrt(z.reduce((a, b) => a + b * b, 0) / 3)).toBeCloseTo(1);
  });
});

describe("encoding", () => {
  it("one-hot", () => {
    expect(oneHot([0, 2], 3)).toEqual([[1, 0, 0], [0, 0, 1]]);
    expect(() => oneHot([3], 3)).toThrow();
  });
  it("category encoder caps rare values into 'other'", () => {
    const enc = fitCategoryEncoder(["a", "a", "a", "b", "b", "c", "d"], 3);
    expect(enc.categories).toEqual(["a", "b"]);
    expect(enc.hasOther).toBe(true);
    expect(encodeCategory("c", enc)).toEqual([0, 0, 1]);
    expect(encodeCategory("a", enc)).toEqual([1, 0, 0]);
    expect(encodeCategory("zzz", enc)).toEqual([0, 0, 1]);
  });
});

describe("missing values", () => {
  const rows = [[1, "x"], [null, "y"], [3, ""], [5, "y"]];
  const numeric = [true, false];
  it("drop removes incomplete rows", () => {
    expect(handleMissing(rows, numeric, "drop").rows).toEqual([[1, "x"], [5, "y"]]);
  });
  it("mean fills numeric with mean and categorical with mode", () => {
    const r = handleMissing(rows, numeric, "mean");
    expect(r.rows[1][0]).toBeCloseTo(3);
    expect(r.rows[2][1]).toBe("y");
  });
  it("mode fills everything with the most frequent value", () => {
    expect(modeOf([2, 2, 3, null])).toBe(2);
    const r = handleMissing([[1], [1], [2], [null]], [true], "mode");
    expect(r.rows[3][0]).toBe(1);
  });
});
