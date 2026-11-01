/** Data Lab model helpers: configs for linear / NN models and evaluation of predictions. */
import type { Activation, ModelConfigInput } from "./builder";
import { argmax } from "./data";
import { classificationReport, logLoss, majorityClass, meanBaseline, regressionReport } from "./metrics";
import type { Task } from "./profile";

export type TabModelKind = "baseline" | "linear" | "nn";
export const MODEL_LABEL: Record<TabModelKind, string> = { baseline: "Baseline", linear: "Linear / Logistic", nn: "Neural network" };

export interface NNSpec {
  hidden: number[];
  activation: Activation;
  epochs: number;
  learningRate: number;
  batchSize: number;
  l2: number;
  dropout: number;
}
export const DEFAULT_NN: NNSpec = { hidden: [16, 8], activation: "relu", epochs: 80, learningRate: 0.01, batchSize: 32, l2: 0, dropout: 0 };

/** Linear = a single dense layer (softmax for classes → multinomial logistic regression). */
export function tabularModelConfig(kind: "linear" | "nn", inputDim: number, task: Task, nClasses: number, nn: NNSpec, seed: number): ModelConfigInput {
  const out = task === "classification"
    ? { type: "dense" as const, units: Math.max(2, nClasses), activation: "softmax" as const }
    : { type: "dense" as const, units: 1, activation: "linear" as const };
  const hidden = kind === "nn"
    ? nn.hidden.flatMap((u) => [
        { type: "dense" as const, units: u, activation: nn.activation },
        ...(nn.dropout > 0 ? [{ type: "dropout" as const, rate: nn.dropout }] : []),
      ])
    : [];
  return {
    inputShape: [inputDim],
    layers: [...hidden, out],
    optimizer: "adam",
    learningRate: kind === "linear" ? Math.max(nn.learningRate, 0.01) : nn.learningRate,
    batchSize: nn.batchSize,
    epochs: nn.epochs,
    l2: nn.l2,
    loss: task === "classification" ? "categoricalCrossentropy" : "meanSquaredError",
    seed,
  };
}

export interface ClassEval {
  task: "classification";
  predictions: number[];
  probs: number[][];
  confusion: number[][];
  perClass: { precision: number; recall: number; f1: number; support: number }[];
  metrics: Record<string, number>;
}
export interface RegEval {
  task: "regression";
  predictions: number[];
  actual: number[];
  metrics: Record<string, number>;
}
export type Eval = ClassEval | RegEval;

export function evaluateClassification(yTrue: number[], probs: number[][], nClasses: number): ClassEval {
  const predictions = probs.map(argmax);
  const rep = classificationReport(yTrue, predictions, nClasses);
  return {
    task: "classification",
    predictions,
    probs,
    confusion: rep.confusion,
    perClass: rep.perClass,
    metrics: { accuracy: rep.accuracy, macroPrecision: rep.macro.precision, macroRecall: rep.macro.recall, macroF1: rep.macro.f1, logLoss: logLoss(yTrue, probs) },
  };
}

export function evaluateRegression(actual: number[], predictions: number[]): RegEval {
  const r = regressionReport(actual, predictions);
  return { task: "regression", predictions, actual, metrics: { mae: r.mae, mse: r.mse, rmse: r.rmse, r2: r.r2 } };
}

/** Baseline: majority class (with its training frequency as "probability") or the training mean. */
export function baselineClassification(trainY: number[], testY: number[], nClasses: number): ClassEval {
  const maj = majorityClass(trainY);
  const freq = Array.from({ length: nClasses }, (_, c) => trainY.filter((y) => y === c).length / Math.max(1, trainY.length));
  const probs = testY.map(() => freq.map((f, c) => (c === maj ? Math.max(f, 1e-6) : f)));
  const ev = evaluateClassification(testY, probs, nClasses);
  ev.predictions = testY.map(() => maj);
  return ev;
}
export function baselineRegression(trainActual: number[], testActual: number[]): RegEval {
  return evaluateRegression(testActual, meanBaseline(trainActual, testActual.length));
}
