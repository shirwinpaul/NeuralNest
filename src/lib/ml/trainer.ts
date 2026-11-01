/**
 * Training runner around `model.fit`: streams per-epoch / per-batch metrics,
 * yields to the browser so the UI stays smooth, and supports
 * Pause / Resume / Step / Stop plus optional early stopping.
 */
import * as tf from "@tensorflow/tfjs";
import { makeRng, permutation } from "./rng";

export interface EpochLog {
  epoch: number; // 1-based
  loss: number;
  acc?: number;
  valLoss?: number;
  valAcc?: number;
}

export type RunnerState = "idle" | "running" | "paused" | "stopped" | "done";
export type StopReason = "completed" | "user" | "early-stopping" | "error";

export interface EarlyStopping {
  patience: number;
  minDelta?: number;
}

export interface TrainOptions {
  epochs: number;
  batchSize: number;
  seed?: number;
  shuffle?: boolean;
  validation?: { x: tf.Tensor; y: tf.Tensor };
  earlyStopping?: EarlyStopping;
  onEpochEnd?: (log: EpochLog) => void | Promise<void>;
  onBatchEnd?: (batch: number, loss: number) => void;
  onStateChange?: (state: RunnerState) => void;
  /** Yield to the browser every N batches (keeps the page responsive). */
  yieldEvery?: number;
}

const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

export class TrainingRunner {
  readonly history: EpochLog[] = [];
  state: RunnerState = "idle";
  stopReason: StopReason | null = null;
  error: Error | null = null;
  startedAt = 0;
  elapsedMs = 0;

  private resumeFn: (() => void) | null = null;
  private pauseRequested = false;
  private stopRequested = false;
  private stepMode = false;
  private running: Promise<EpochLog[]> | null = null;
  private best = Infinity;
  private sinceBest = 0;
  private rng;

  constructor(
    private model: tf.LayersModel,
    private x: tf.Tensor,
    private y: tf.Tensor,
    private opts: TrainOptions,
  ) {
    this.rng = makeRng(opts.seed ?? 42);
  }

  get epoch(): number {
    return this.history.length;
  }

  private setState(s: RunnerState) {
    this.state = s;
    this.opts.onStateChange?.(s);
  }

  /** Start (or continue) training until all epochs are done or it is stopped. */
  run(): Promise<EpochLog[]> {
    if (this.state === "paused") {
      this.resume();
      return this.running ?? Promise.resolve(this.history);
    }
    if (this.running) return this.running;
    if (this.state === "done" || this.state === "stopped") return Promise.resolve(this.history);
    this.running = this.loop().finally(() => {
      this.running = null;
    });
    return this.running;
  }

  /** Train exactly one more epoch, then pause. */
  step(): Promise<EpochLog[]> {
    this.stepMode = true;
    return this.run();
  }

  pause() {
    if (this.state === "running") this.pauseRequested = true;
  }

  resume() {
    this.pauseRequested = false;
    if (this.resumeFn) {
      const r = this.resumeFn;
      this.resumeFn = null;
      this.setState("running");
      r();
    }
  }

  stop() {
    this.stopRequested = true;
    this.model.stopTraining = true;
    this.resume(); // release a paused loop so it can exit
  }

  private async waitIfPaused() {
    if (!this.pauseRequested || this.stopRequested) return;
    this.setState("paused");
    this.elapsedMs += performance.now() - this.startedAt;
    await new Promise<void>((res) => (this.resumeFn = res));
    this.startedAt = performance.now();
  }

  private async loop(): Promise<EpochLog[]> {
    this.setState("running");
    this.startedAt = performance.now();
    const n = this.x.shape[0];
    try {
      while (this.history.length < this.opts.epochs && !this.stopRequested) {
        await this.waitIfPaused();
        if (this.stopRequested) break;
        const log = await this.trainOneEpoch(n);
        this.history.push(log);
        await this.opts.onEpochEnd?.(log);
        if (this.checkEarlyStop(log)) {
          this.stopReason = "early-stopping";
          break;
        }
        if (this.stepMode) {
          this.stepMode = false;
          this.pauseRequested = true;
        }
      }
      if (!this.stopReason) this.stopReason = this.stopRequested ? "user" : "completed";
      this.elapsedMs += performance.now() - this.startedAt;
      this.setState(this.stopReason === "completed" ? "done" : "stopped");
    } catch (e) {
      this.error = e instanceof Error ? e : new Error(String(e));
      this.stopReason = "error";
      this.setState("stopped");
    }
    return this.history;
  }

  private async trainOneEpoch(n: number): Promise<EpochLog> {
    let xs = this.x, ys = this.y;
    let disposeAfter: tf.Tensor[] = [];
    if (this.opts.shuffle !== false && n > 1) {
      // Seeded shuffle each epoch → reproducible runs.
      const idx = tf.tensor1d(permutation(n, this.rng), "int32");
      xs = tf.gather(this.x, idx);
      ys = tf.gather(this.y, idx);
      disposeAfter = [idx, xs, ys];
    }
    const yieldEvery = this.opts.yieldEvery ?? 4;
    try {
      const h = await this.model.fit(xs, ys, {
        epochs: 1,
        batchSize: this.opts.batchSize,
        shuffle: false,
        verbose: 0,
        validationData: this.opts.validation ? [this.opts.validation.x, this.opts.validation.y] : undefined,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            this.opts.onBatchEnd?.(batch, num(logs?.loss) ?? NaN);
            if (batch % yieldEvery === 0) await tf.nextFrame();
            await this.waitIfPaused();
          },
        },
      });
      const last = (k: string) => {
        const arr = h.history[k];
        return arr ? num(arr[arr.length - 1]) : undefined;
      };
      return {
        epoch: this.history.length + 1,
        loss: last("loss") ?? NaN,
        acc: last("acc") ?? last("accuracy"),
        valLoss: last("val_loss"),
        valAcc: last("val_acc") ?? last("val_accuracy"),
      };
    } finally {
      disposeAfter.forEach((t) => t.dispose());
    }
  }

  private checkEarlyStop(log: EpochLog): boolean {
    const es = this.opts.earlyStopping;
    if (!es) return false;
    const monitored = log.valLoss ?? log.loss;
    if (monitored < this.best - (es.minDelta ?? 0)) {
      this.best = monitored;
      this.sinceBest = 0;
      return false;
    }
    this.sinceBest++;
    return this.sinceBest >= es.patience;
  }
}

/** Predict in batches inside tidy; returns plain JS arrays and frees all tensors. */
export async function predictArray(model: tf.LayersModel, x: tf.Tensor, batchSize = 256): Promise<number[][]> {
  const out = tf.tidy(() => model.predict(x, { batchSize }) as tf.Tensor);
  try {
    const arr = (await out.array()) as number[][] | number[];
    return (arr as unknown[]).map((r) => (Array.isArray(r) ? (r as number[]) : [r as number]));
  } finally {
    out.dispose();
  }
}
