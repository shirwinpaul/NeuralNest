import { describe, expect, it } from "vitest";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { buildModel, disposeModel } from "@/lib/ml/builder";
import { modelToZip, zipToModel } from "@/lib/ml/modelio";
import type { ModelMeta } from "@/lib/ml/storage";

const meta: ModelMeta = {
  app: "neuralnest", version: 1, name: "xor", lab: "playground", task: "classification", createdAt: 1, inputShape: [2],
  classNames: ["0", "1"], features: ["x1", "x2"], featureExtractor: "none", metrics: { testAccuracy: 1 },
};

describe("model export/import", () => {
  it("round-trips weights, predictions and metadata through a zip", async () => {
    const model = buildModel({
      inputShape: [2], layers: [{ type: "dense", units: 4, activation: "tanh" }, { type: "dense", units: 1, activation: "sigmoid" }],
      optimizer: "adam", learningRate: 0.01, batchSize: 4, epochs: 1, loss: "binaryCrossentropy", seed: 5,
    }, LIMITS.playground);
    const zip = await modelToZip(model, meta);
    expect(zip.byteLength).toBeGreaterThan(100);
    const { model: back, meta: meta2 } = await zipToModel(zip);
    expect(meta2).toEqual(meta);
    const x = tf.tensor2d([[0.1, 0.9], [-0.5, 0.3]]);
    const a = (model.predict(x) as tf.Tensor).dataSync(), b = (back.predict(x) as tf.Tensor).dataSync();
    expect(Array.from(b)).toEqual(Array.from(a));
    x.dispose();
    disposeModel(model);
    back.dispose();
  });

  it("rejects zips without NeuralNest metadata", async () => {
    const JSZip = (await import("jszip")).default;
    const z = new JSZip();
    z.file("model.json", "{}");
    await expect(zipToModel(await z.generateAsync({ type: "uint8array" }))).rejects.toThrow(/must contain/);
  });
});
