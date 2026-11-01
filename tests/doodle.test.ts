import { describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { buildModel, disposeModel } from "@/lib/ml/builder";
import { augment, centreOfMass, DOODLE_SIZE, inkBounds, preprocessDoodle, rgbaToInk } from "@/lib/ml/doodle";
import { saliencyMap } from "@/lib/ml/saliency";
import { makeRng } from "@/lib/ml/rng";

/** A 200×200 pad with a filled rectangle in the top-left corner. */
function pad(x0: number, y0: number, w: number, h: number, W = 200, H = 200) {
  const px = new Float32Array(W * H);
  for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) px[y * W + x] = 1;
  return px;
}

describe("doodle preprocessing", () => {
  it("returns null for an empty drawing", () => {
    expect(preprocessDoodle(new Float32Array(100 * 100), 100, 100)).toBeNull();
  });

  it("centres an off-centre drawing and scales its longest side to 20px", () => {
    const out = preprocessDoodle(pad(5, 10, 40, 80), 200, 200)!;
    expect(out.length).toBe(DOODLE_SIZE * DOODLE_SIZE);
    const b = inkBounds(out, 28, 28, 0.1)!;
    expect(b.y1 - b.y0 + 1).toBeGreaterThanOrEqual(19);
    expect(b.y1 - b.y0 + 1).toBeLessThanOrEqual(21);
    expect(b.x1 - b.x0 + 1).toBeGreaterThanOrEqual(9);
    expect(b.x1 - b.x0 + 1).toBeLessThanOrEqual(11);
    const { cx, cy } = centreOfMass(out, 28, 28);
    expect(Math.abs(cx - 13.5)).toBeLessThanOrEqual(1);
    expect(Math.abs(cy - 13.5)).toBeLessThanOrEqual(1);
    expect(Math.max(...out)).toBeCloseTo(1);
  });

  it("is translation invariant: the same shape anywhere gives the same 28×28 image", () => {
    const a = preprocessDoodle(pad(5, 5, 50, 30), 200, 200)!;
    const b = preprocessDoodle(pad(120, 150, 50, 30), 200, 200)!;
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += Math.abs(a[i] - b[i]);
    expect(diff).toBeLessThan(1e-6);
  });

  it("rgbaToInk maps dark ink on white to 1 and paper to 0", () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]);
    const ink = rgbaToInk(rgba);
    expect(ink[0]).toBeCloseTo(1);
    expect(ink[1]).toBeCloseTo(0);
  });

  it("augmentation keeps size, stays in range and is seeded", () => {
    const img = preprocessDoodle(pad(50, 50, 60, 60), 200, 200)!;
    const a = augment(img, makeRng(1)), b = augment(img, makeRng(1));
    expect(a.length).toBe(784);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(Math.max(...a)).toBeLessThanOrEqual(1.0001);
    expect(a.reduce((s, v) => s + v, 0)).toBeGreaterThan(0);
  });
});

describe("saliency", () => {
  it("returns a normalised 28×28 map without leaking tensors", () => {
    const model = buildModel({
      inputShape: [28, 28, 1],
      layers: [{ type: "conv2d", filters: 4, kernelSize: 3, activation: "relu", padding: "same" }, { type: "maxPooling2d", poolSize: 2 }, { type: "flatten" }, { type: "dense", units: 3, activation: "softmax" }],
      optimizer: "adam", learningRate: 0.001, batchSize: 16, epochs: 1, loss: "categoricalCrossentropy",
    }, LIMITS.doodle);
    const before = tf.memory().numTensors;
    const img = preprocessDoodle(pad(50, 50, 60, 60), 200, 200)!;
    const s = saliencyMap(model, img, [28, 28, 1], 1);
    expect(s.length).toBe(784);
    expect(Math.max(...s)).toBeCloseTo(1);
    expect(Math.min(...s)).toBeGreaterThanOrEqual(0);
    expect(tf.memory().numTensors).toBe(before);
    disposeModel(model);
  });
});
