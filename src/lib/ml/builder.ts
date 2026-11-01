/** Builds TF.js models from a validated (Zod) config, enforcing per-lab limits. */
import * as tf from "@tensorflow/tfjs";
import { z } from "zod";
import type { LabLimits } from "@/config/limits";
import { deriveSeed } from "./rng";

export const ActivationSchema = z.enum(["relu", "sigmoid", "tanh", "softmax", "linear"]);
export type Activation = z.infer<typeof ActivationSchema>;

export const LayerSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("dense"), units: z.number().int().min(1), activation: ActivationSchema }),
  z.object({
    type: z.literal("conv2d"),
    filters: z.number().int().min(1),
    kernelSize: z.number().int().min(1).max(7),
    activation: ActivationSchema,
    padding: z.enum(["valid", "same"]).default("same"),
  }),
  z.object({ type: z.literal("maxPooling2d"), poolSize: z.number().int().min(2).max(4) }),
  z.object({ type: z.literal("flatten") }),
  z.object({ type: z.literal("dropout"), rate: z.number().min(0).max(0.9) }),
]);
export type LayerConfig = z.infer<typeof LayerSchema>;

export const LossSchema = z.enum(["categoricalCrossentropy", "binaryCrossentropy", "meanSquaredError"]);

export const ModelConfigSchema = z.object({
  inputShape: z.array(z.number().int().min(1)).min(1).max(3),
  layers: z.array(LayerSchema).min(1),
  optimizer: z.enum(["sgd", "adam"]),
  learningRate: z.number().positive(),
  batchSize: z.number().int().positive(),
  epochs: z.number().int().positive(),
  l2: z.number().min(0).default(0),
  loss: LossSchema,
  seed: z.number().int().default(42),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;
export type ModelConfigInput = z.input<typeof ModelConfigSchema>;

export interface ShapeInfo { shape: number[]; params: number }

/**
 * Walk the layers computing output shapes and parameter counts analytically,
 * so the UI can show the parameter count before anything is built.
 * dense:  in·units + units           conv2d: k·k·inC·filters + filters
 * pool/flatten/dropout: 0 parameters.
 */
export function analyzeLayers(config: Pick<ModelConfig, "inputShape" | "layers">): { layers: ShapeInfo[]; totalParams: number; errors: string[] } {
  const errors: string[] = [];
  let shape = [...config.inputShape];
  const out: ShapeInfo[] = [];
  config.layers.forEach((l, i) => {
    let params = 0;
    switch (l.type) {
      case "dense": {
        if (shape.length !== 1) errors.push(`Layer ${i + 1}: add a Flatten layer before Dense.`);
        const inDim = shape.reduce((a, b) => a * b, 1);
        params = inDim * l.units + l.units;
        shape = [l.units];
        break;
      }
      case "conv2d": {
        if (shape.length !== 3) {
          errors.push(`Layer ${i + 1}: Conv2D needs image-shaped input [height, width, channels].`);
          break;
        }
        const [h, w, c] = shape;
        params = l.kernelSize * l.kernelSize * c * l.filters + l.filters;
        const oh = l.padding === "same" ? h : h - l.kernelSize + 1;
        const ow = l.padding === "same" ? w : w - l.kernelSize + 1;
        if (oh < 1 || ow < 1) errors.push(`Layer ${i + 1}: image is too small for this kernel.`);
        shape = [oh, ow, l.filters];
        break;
      }
      case "maxPooling2d": {
        if (shape.length !== 3) {
          errors.push(`Layer ${i + 1}: MaxPooling needs image-shaped input.`);
          break;
        }
        const [h, w, c] = shape;
        // stride = pool size, 'valid' padding
        const oh = Math.floor((h - l.poolSize) / l.poolSize) + 1;
        const ow = Math.floor((w - l.poolSize) / l.poolSize) + 1;
        if (oh < 1 || ow < 1) errors.push(`Layer ${i + 1}: image is too small to pool.`);
        shape = [oh, ow, c];
        break;
      }
      case "flatten":
        shape = [shape.reduce((a, b) => a * b, 1)];
        break;
      case "dropout":
        break;
    }
    out.push({ shape: [...shape], params });
  });
  return { layers: out, totalParams: out.reduce((a, l) => a + l.params, 0), errors };
}

export function countParams(config: Pick<ModelConfig, "inputShape" | "layers">): number {
  return analyzeLayers(config).totalParams;
}

export type ValidationResult = { ok: true; config: ModelConfig; params: number } | { ok: false; errors: string[] };

/** Schema + limit + structural checks. Never throws. */
export function validateConfig(input: unknown, limits: LabLimits): ValidationResult {
  const parsed = ModelConfigSchema.safeParse(input);
  if (!parsed.success) return { ok: false, errors: parsed.error.issues.map((i) => `${i.path.join(".") || "config"}: ${i.message}`) };
  const c = parsed.data;
  const errors: string[] = [];
  const weighted = c.layers.filter((l) => l.type === "dense" || l.type === "conv2d").length;
  if (weighted > limits.maxLayers) errors.push(`Too many layers (${weighted}); the limit is ${limits.maxLayers}.`);
  c.layers.forEach((l, i) => {
    // The output layer may need up to 10 units (one per class) even in tiny labs.
    const unitCap = i === c.layers.length - 1 ? Math.max(limits.maxUnits, 10) : limits.maxUnits;
    if (l.type === "dense" && l.units > unitCap) errors.push(`Dense layer has ${l.units} units; the limit is ${unitCap}.`);
    if (l.type === "conv2d" && l.filters > limits.maxFilters) errors.push(`Conv layer has ${l.filters} filters; the limit is ${limits.maxFilters}.`);
    if (l.type === "dropout" && l.rate > limits.maxDropout) errors.push(`Dropout ${l.rate} is above the limit ${limits.maxDropout}.`);
  });
  if (c.epochs > limits.maxEpochs) errors.push(`Epochs ${c.epochs} is above the limit ${limits.maxEpochs}.`);
  if (!limits.batchSizes.includes(c.batchSize)) errors.push(`Batch size must be one of ${limits.batchSizes.join(", ")}.`);
  if (c.learningRate < limits.minLearningRate || c.learningRate > limits.maxLearningRate)
    errors.push(`Learning rate must be between ${limits.minLearningRate} and ${limits.maxLearningRate}.`);
  if (c.l2 > limits.maxL2) errors.push(`L2 ${c.l2} is above the limit ${limits.maxL2}.`);
  const a = analyzeLayers(c);
  errors.push(...a.errors);
  if (a.totalParams > limits.maxParams) errors.push(`Model has ${a.totalParams.toLocaleString()} parameters; the limit is ${limits.maxParams.toLocaleString()}.`);
  const last = c.layers[c.layers.length - 1];
  if (last.type !== "dense") errors.push("The last layer must be a Dense output layer.");
  if (c.loss === "categoricalCrossentropy" && last.type === "dense" && last.activation !== "softmax") errors.push("Categorical cross-entropy needs a softmax output.");
  if (c.loss === "binaryCrossentropy" && last.type === "dense" && last.activation !== "sigmoid") errors.push("Binary cross-entropy needs a sigmoid output.");
  return errors.length ? { ok: false, errors } : { ok: true, config: c, params: a.totalParams };
}

/** Clamp a (possibly AI-suggested) numeric value into the allowed range. */
export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
export function nearestBatch(v: number, allowed: readonly number[]): number {
  return allowed.reduce((best, b) => (Math.abs(b - v) < Math.abs(best - v) ? b : best), allowed[0]);
}

/** Build + compile a model. Throws with a readable message if the config is invalid. */
export function buildModel(input: unknown, limits: LabLimits): tf.Sequential {
  const v = validateConfig(input, limits);
  if (!v.ok) throw new Error(v.errors.join(" "));
  const c = v.config;
  const model = tf.sequential();
  const reg = c.l2 > 0 ? tf.regularizers.l2({ l2: c.l2 }) : undefined;
  c.layers.forEach((l, i) => {
    const first = i === 0 ? { inputShape: c.inputShape } : {};
    const kernelInitializer = tf.initializers.glorotUniform({ seed: deriveSeed(c.seed, i) });
    switch (l.type) {
      case "dense":
        model.add(tf.layers.dense({ ...first, units: l.units, activation: l.activation, kernelInitializer, kernelRegularizer: reg }));
        break;
      case "conv2d":
        model.add(tf.layers.conv2d({ ...first, filters: l.filters, kernelSize: l.kernelSize, padding: l.padding, activation: l.activation, kernelInitializer, kernelRegularizer: reg }));
        break;
      case "maxPooling2d":
        model.add(tf.layers.maxPooling2d({ ...first, poolSize: l.poolSize, strides: l.poolSize }));
        break;
      case "flatten":
        model.add(tf.layers.flatten(first));
        break;
      case "dropout":
        model.add(tf.layers.dropout({ ...first, rate: l.rate, seed: deriveSeed(c.seed, 1000 + i) }));
        break;
    }
  });
  compileModel(model, c);
  return model;
}

export function compileModel(model: tf.LayersModel, c: Pick<ModelConfig, "optimizer" | "learningRate" | "loss">): void {
  const optimizer = c.optimizer === "adam" ? tf.train.adam(c.learningRate) : tf.train.sgd(c.learningRate);
  model.compile({ optimizer, loss: c.loss, metrics: c.loss === "meanSquaredError" ? [] : ["accuracy"] });
}

/** Dispose a model and its optimizer state (TF.js keeps optimizer variables otherwise). */
export function disposeModel(model: tf.LayersModel | null | undefined): void {
  if (!model) return;
  try {
    model.optimizer?.dispose();
  } catch {
    /* already disposed */
  }
  try {
    model.dispose();
  } catch {
    /* already disposed */
  }
}

/** Compact human-readable architecture string, e.g. "2 → 4 relu → 1 sigmoid". */
export function describeArchitecture(c: Pick<ModelConfig, "inputShape" | "layers">): string {
  const parts = [c.inputShape.join("×")];
  for (const l of c.layers) {
    if (l.type === "dense") parts.push(`${l.units} ${l.activation}`);
    else if (l.type === "conv2d") parts.push(`conv${l.kernelSize}×${l.filters}`);
    else if (l.type === "maxPooling2d") parts.push(`pool${l.poolSize}`);
    else if (l.type === "flatten") parts.push("flatten");
    else parts.push(`dropout ${l.rate}`);
  }
  return parts.join(" → ");
}
