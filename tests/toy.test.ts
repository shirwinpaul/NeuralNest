import { describe, expect, it } from "vitest";
import { featurize, generateToy, TOY_DATASETS } from "@/lib/ml/datasets/toy";

describe("toy dataset generators", () => {
  for (const d of TOY_DATASETS) {
    it(`${d.id}: right size, finite values, reproducible`, () => {
      const a = generateToy(d.id, { n: 200, noise: 0.1, seed: 3 });
      const b = generateToy(d.id, { n: 200, noise: 0.1, seed: 3 });
      const c = generateToy(d.id, { n: 200, noise: 0.1, seed: 4 });
      expect(a).toHaveLength(200);
      expect(a).toEqual(b);
      expect(a).not.toEqual(c);
      for (const p of a) {
        expect(Number.isFinite(p.x1) && Number.isFinite(p.x2) && Number.isFinite(p.label)).toBe(true);
        expect(Math.abs(p.x1)).toBeLessThanOrEqual(1.2);
      }
      if (d.task === "classification") {
        const ones = a.filter((p) => p.label === 1).length;
        expect(a.every((p) => p.label === 0 || p.label === 1)).toBe(true);
        expect(ones).toBeGreaterThan(40);
        expect(ones).toBeLessThan(160);
      }
    });
  }

  it("circles: inner class is closer to the origin", () => {
    const pts = generateToy("circles", { n: 300, noise: 0, seed: 1 });
    const r = (l: number) => {
      const s = pts.filter((p) => p.label === l);
      return s.reduce((a, p) => a + Math.hypot(p.x1, p.x2), 0) / s.length;
    };
    expect(r(1)).toBeLessThan(r(0));
  });

  it("xor: label is 1 when x1·x2 > 0 (noise-free)", () => {
    const pts = generateToy("xor", { n: 100, noise: 0, seed: 1 });
    expect(pts.every((p) => (p.x1 * p.x2 > 0 ? 1 : 0) === p.label)).toBe(true);
  });

  it("featurize computes engineered features", () => {
    expect(featurize(0.5, -2, ["x1", "x2", "x1sq", "x1x2"])).toEqual([0.5, -2, 0.25, -1]);
    expect(featurize(0.5, 0, ["sinx1"])[0]).toBeCloseTo(1);
  });
});
