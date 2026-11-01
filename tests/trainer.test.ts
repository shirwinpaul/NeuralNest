import { describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { buildModel, disposeModel } from "@/lib/ml/builder";
import { predictArray, TrainingRunner, type RunnerState } from "@/lib/ml/trainer";
import { gaussian, makeRng } from "@/lib/ml/rng";

function xorData() {
  // 4 corners repeated with small seeded jitter
  const rng = makeRng(5);
  const X: number[][] = [], Y: number[][] = [];
  for (let i = 0; i < 80; i++) {
    const a = i % 2, b = Math.floor(i / 2) % 2;
    X.push([a + gaussian(rng, 0, 0.05), b + gaussian(rng, 0, 0.05)]);
    Y.push([a ^ b]);
  }
  return { X, Y };
}

const xorConfig = {
  inputShape: [2],
  layers: [
    { type: "dense", units: 8, activation: "tanh" },
    { type: "dense", units: 1, activation: "sigmoid" },
  ],
  optimizer: "adam",
  learningRate: 0.05,
  batchSize: 16,
  epochs: 150,
  loss: "binaryCrossentropy",
  seed: 1,
} as const;

describe("training runner", () => {
  it("learns XOR to high accuracy within 150 epochs (seeded)", async () => {
    const { X, Y } = xorData();
    const model = buildModel(xorConfig, LIMITS.playground);
    const x = tf.tensor2d(X), y = tf.tensor2d(Y);
    const r = new TrainingRunner(model, x, y, { epochs: 150, batchSize: 16, seed: 1 });
    const hist = await r.run();
    expect(r.state).toBe("done");
    expect(hist.length).toBe(150);
    expect(hist[149].loss).toBeLessThan(hist[0].loss);
    const probs = await predictArray(model, x);
    const acc = probs.filter((p, i) => (p[0] > 0.5 ? 1 : 0) === Y[i][0]).length / Y.length;
    expect(acc).toBeGreaterThanOrEqual(0.95);
    tf.dispose([x, y]);
    disposeModel(model);
  });

  it("linear regression recovers slope and intercept (y = 3x − 2)", async () => {
    const rng = makeRng(9);
    const xs = Array.from({ length: 200 }, () => rng() * 4 - 2);
    const ys = xs.map((v) => 3 * v - 2 + gaussian(rng, 0, 0.05));
    const model = buildModel({
      inputShape: [1], layers: [{ type: "dense", units: 1, activation: "linear" }],
      optimizer: "adam", learningRate: 0.1, batchSize: 32, epochs: 150, loss: "meanSquaredError", seed: 3,
    }, LIMITS.data);
    const x = tf.tensor2d(xs, [200, 1]), y = tf.tensor2d(ys, [200, 1]);
    await new TrainingRunner(model, x, y, { epochs: 150, batchSize: 32, seed: 3 }).run();
    const [w, b] = model.getWeights().map((t) => t.dataSync()[0]);
    expect(w).toBeCloseTo(3, 1);
    expect(b).toBeCloseTo(-2, 1);
    tf.dispose([x, y]);
    disposeModel(model);
  });

  it("supports stop, step, pause/resume, early stopping and frees tensors", async () => {
    const { X, Y } = xorData();
    const before = tf.memory().numTensors;
    const model = buildModel({ ...xorConfig, epochs: 100 }, LIMITS.playground);
    const x = tf.tensor2d(X), y = tf.tensor2d(Y);
    const states: RunnerState[] = [];

    // step: exactly one epoch, then paused
    const r = new TrainingRunner(model, x, y, { epochs: 100, batchSize: 16, onStateChange: (s) => states.push(s) });
    const p = r.step();
    await waitFor(() => r.state === "paused");
    expect(r.epoch).toBe(1);
    // resume, then pause after a few epochs
    r.run();
    await waitFor(() => r.epoch >= 3);
    r.pause();
    await waitFor(() => r.state === "paused");
    const pausedAt = r.epoch;
    await new Promise((res) => setTimeout(res, 50));
    expect(r.epoch).toBeLessThanOrEqual(pausedAt + 1);
    r.stop();
    await p;
    expect(r.state).toBe("stopped");
    expect(r.stopReason).toBe("user");
    expect(states).toContain("paused");

    // early stopping on a validation set that can't improve (patience 2, huge minDelta)
    const r2 = new TrainingRunner(model, x, y, { epochs: 50, batchSize: 16, validation: { x, y }, earlyStopping: { patience: 2, minDelta: 1e9 } });
    await r2.run();
    expect(r2.stopReason).toBe("early-stopping");
    expect(r2.epoch).toBe(3);
    expect(r2.history[0].valLoss).toBeDefined();

    tf.dispose([x, y]);
    disposeModel(model);
    expect(tf.memory().numTensors).toBe(before);
  });
});

async function waitFor(cond: () => boolean, timeout = 20000) {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > timeout) throw new Error("timeout");
    await new Promise((r) => setTimeout(r, 5));
  }
}
