import { describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { analyzeLayers, buildModel, countParams, disposeModel, describeArchitecture, validateConfig, type ModelConfigInput } from "@/lib/ml/builder";

const mlp: ModelConfigInput = {
  inputShape: [2],
  layers: [
    { type: "dense", units: 4, activation: "relu" },
    { type: "dense", units: 3, activation: "tanh" },
    { type: "dense", units: 1, activation: "sigmoid" },
  ],
  optimizer: "adam",
  learningRate: 0.03,
  batchSize: 10,
  epochs: 50,
  loss: "binaryCrossentropy",
};

describe("model builder", () => {
  it("counts parameters of a known MLP: (2·4+4)+(4·3+3)+(3·1+1)=31", () => {
    expect(countParams(validateConfigOk(mlp))).toBe(31);
  });

  it("built model has the same parameter count as the analytic one", () => {
    const m = buildModel(mlp, LIMITS.playground);
    expect(m.countParams()).toBe(31);
    disposeModel(m);
  });

  it("counts CNN parameters and matches TF.js", () => {
    const cnn: ModelConfigInput = {
      inputShape: [28, 28, 1],
      layers: [
        { type: "conv2d", filters: 8, kernelSize: 3, activation: "relu", padding: "same" },
        { type: "maxPooling2d", poolSize: 2 },
        { type: "conv2d", filters: 16, kernelSize: 3, activation: "relu", padding: "same" },
        { type: "maxPooling2d", poolSize: 2 },
        { type: "flatten" },
        { type: "dropout", rate: 0.25 },
        { type: "dense", units: 32, activation: "relu" },
        { type: "dense", units: 3, activation: "softmax" },
      ],
      optimizer: "adam",
      learningRate: 0.001,
      batchSize: 16,
      epochs: 10,
      loss: "categoricalCrossentropy",
    };
    // conv1 3·3·1·8+8=80, conv2 3·3·8·16+16=1168, dense 784·32+32=25120, out 32·3+3=99
    const a = analyzeLayers(validateConfigOk(cnn, "doodle"));
    expect(a.layers[3].shape).toEqual([7, 7, 16]);
    expect(a.totalParams).toBe(80 + 1168 + 25120 + 99);
    const m = buildModel(cnn, LIMITS.doodle);
    expect(m.countParams()).toBe(a.totalParams);
    disposeModel(m);
  });

  it("rejects invalid configs without throwing", () => {
    const r = validateConfig({ ...mlp, layers: [] }, LIMITS.playground);
    expect(r.ok).toBe(false);
    expect(validateConfig({ ...mlp, optimizer: "magic" }, LIMITS.playground).ok).toBe(false);
  });

  it("rejects over-limit configs", () => {
    const tooWide = { ...mlp, layers: [{ type: "dense", units: 64, activation: "relu" }, mlp.layers[2]] };
    const r = validateConfig(tooWide, LIMITS.playground);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors.join(" ")).toMatch(/limit/);
    expect(validateConfig({ ...mlp, epochs: 99999 }, LIMITS.playground).ok).toBe(false);
    expect(validateConfig({ ...mlp, batchSize: 7 }, LIMITS.playground).ok).toBe(false);
    expect(validateConfig({ ...mlp, learningRate: 50 }, LIMITS.playground).ok).toBe(false);
    expect(() => buildModel(tooWide, LIMITS.playground)).toThrow();
  });

  it("rejects mismatched loss/output and dense-after-conv without flatten", () => {
    expect(validateConfig({ ...mlp, loss: "categoricalCrossentropy" }, LIMITS.playground).ok).toBe(false);
    const bad = { ...mlp, inputShape: [8, 8, 1], layers: [{ type: "conv2d", filters: 4, kernelSize: 3, activation: "relu" }, { type: "dense", units: 2, activation: "softmax" }], loss: "categoricalCrossentropy" };
    expect(validateConfig(bad, LIMITS.doodle).ok).toBe(false);
  });

  it("describes architectures compactly", () => {
    expect(describeArchitecture(validateConfigOk(mlp))).toBe("2 → 4 relu → 3 tanh → 1 sigmoid");
  });

  it("does not leak tensors when building and disposing", () => {
    const before = tf.memory().numTensors;
    const m = buildModel(mlp, LIMITS.playground);
    disposeModel(m);
    expect(tf.memory().numTensors).toBe(before);
  });
});

function validateConfigOk(c: ModelConfigInput, lab: keyof typeof LIMITS = "playground") {
  const r = validateConfig(c, LIMITS[lab]);
  if (!r.ok) throw new Error(r.errors.join("; "));
  return r.config;
}
