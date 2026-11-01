"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { LIMITS } from "@/config/limits";
import { CHALLENGES, type PlaygroundSnapshot } from "@/data/challenges";
import { resolvePreset, type PlaygroundPreset } from "@/data/presets";
import { useEngine } from "@/hooks/useEngine";
import { buildModel, countParams, disposeModel, describeArchitecture, type Activation, type ModelConfigInput } from "@/lib/ml/builder";
import { splitIndices, take } from "@/lib/ml/data";
import { FEATURES, featurize, generateToy, TOY_DATASETS, type FeatureId, type ToyPoint } from "@/lib/ml/datasets/toy";
import { TrainingRunner, type EpochLog, type RunnerState } from "@/lib/ml/trainer";
import { loadProgress, setFlag, unlockBadge } from "@/lib/progress";
import { activeBackend } from "@/lib/ml/engine";
import { saveRun } from "@/lib/ml/storage";
import { EnginePanel } from "../EnginePanel";
import { NetworkDiagram, denseKernels } from "../NetworkDiagram";
import { LineChartPanel } from "@/components/charts/LineChartPanel";
import { Celebrate } from "@/components/ui/Celebrate";
import { ModelSaveBar } from "../ModelSaveBar";
import { TutorPanel } from "@/components/tutor/TutorPanel";
import { buildRunSummary } from "@/lib/ai/summary";
import { nearestBatch } from "@/lib/ml/builder";
import { DecisionBoundary, gridCoords } from "./DecisionBoundary";

const VIS_EVERY = 5;
const MAX_RUN_EPOCHS = 1000;

interface Live { trainAcc?: number; testAcc?: number; trainMse?: number; testMse?: number }

export default function Playground({ presetId }: { presetId: string | null }) {
  const { engine, stats } = useEngine();
  const [cfg, setCfg] = useState<PlaygroundPreset>(() => resolvePreset(presetId));
  const [seed, setSeed] = useState(1);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [state, setState] = useState<RunnerState>("idle");
  const [grid, setGrid] = useState<Float32Array | null>(null);
  const [curve, setCurve] = useState<{ x: number; y: number }[] | null>(null);
  const [live, setLive] = useState<Live>({});
  const [weights, setWeights] = useState<number[][][] | null>(null);
  const [showTest, setShowTest] = useState(true);
  const [celebrate, setCelebrate] = useState<{ emoji: string; name: string } | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  const [saved, setSaved] = useState(false);

  const modelRef = useRef<tf.Sequential | null>(null);
  const runnerRef = useRef<TrainingRunner | null>(null);
  const tensorsRef = useRef<tf.Tensor[]>([]);
  const task = TOY_DATASETS.find((d) => d.id === cfg.kind)!.task;

  // ---- data ----
  const points = useMemo(() => generateToy(cfg.kind, { n: cfg.n, noise: cfg.noise, seed }), [cfg.kind, cfg.n, cfg.noise, seed]);
  const split = useMemo(() => {
    const labels = task === "classification" ? points.map((p) => p.label) : undefined;
    const s = splitIndices(points.length, { train: cfg.trainPct, val: 0, test: 100 - cfg.trainPct }, seed, labels);
    return { train: take(points, s.train), test: take(points, s.test) };
  }, [points, cfg.trainPct, seed, task]);

  const modelConfig = useMemo<ModelConfigInput>(() => ({
    inputShape: [cfg.features.length],
    layers: [
      ...cfg.hidden.map((u) => ({ type: "dense" as const, units: u, activation: cfg.activation })),
      task === "classification" ? { type: "dense" as const, units: 1, activation: "sigmoid" as const } : { type: "dense" as const, units: 1, activation: "linear" as const },
    ],
    optimizer: cfg.optimizer,
    learningRate: cfg.learningRate,
    batchSize: cfg.batchSize,
    epochs: MAX_RUN_EPOCHS,
    l2: cfg.l2,
    loss: task === "classification" ? "binaryCrossentropy" : "meanSquaredError",
    seed,
  }), [cfg, task, seed]);
  const params = useMemo(() => countParams({ inputShape: [cfg.features.length], layers: modelConfig.layers as never }), [cfg.features.length, modelConfig.layers]);

  // ---- lifecycle ----
  const teardown = useCallback(() => {
    runnerRef.current?.stop();
    runnerRef.current = null;
    disposeModel(modelRef.current);
    modelRef.current = null;
    tf.dispose(tensorsRef.current);
    tensorsRef.current = [];
  }, []);

  const reset = useCallback(() => {
    teardown();
    setHistory([]);
    setGrid(null);
    setCurve(null);
    setLive({});
    setWeights(null);
    setState("idle");
    setError(null);
    setSaved(false);
  }, [teardown]);

  useEffect(() => teardown, [teardown]);

  const toXY = useCallback((pts: ToyPoint[]) => {
    const X = pts.map((p) => featurize(p.x1, p.x2, cfg.features));
    return { x: tf.tensor2d(X, [pts.length, cfg.features.length]), y: tf.tensor2d(pts.map((p) => [p.label]), [pts.length, 1]) };
  }, [cfg.features]);

  const checkChallenges = useCallback((snap: PlaygroundSnapshot) => {
    const flags = loadProgress().flags;
    for (const ch of CHALLENGES) {
      const res = ch.check(snap, flags);
      res.setFlags?.forEach((f) => setFlag(f));
      if (res.hint) setHint(res.hint);
      if (res.solved) {
        setSolved((s) => ({ ...s, [ch.id]: true }));
        if (unlockBadge(ch.badge.id)) setCelebrate(ch.badge);
      }
    }
  }, []);

  /** Recompute heatmap, exact train/test metrics and weights from the current model. */
  const refreshVisuals = useCallback(async (log?: EpochLog) => {
    const m = modelRef.current;
    const [trainX, , testX, , gridX] = tensorsRef.current;
    if (!m || !gridX) return;
    const outs = tf.tidy(() => [m.predict(gridX), m.predict(trainX), m.predict(testX)] as tf.Tensor[]);
    const [g, ptr, pte] = await Promise.all(outs.map((o) => o.data()));
    tf.dispose(outs);
    const metric = (pred: ArrayLike<number>, pts: ToyPoint[]) =>
      task === "classification"
        ? pts.filter((p, i) => (pred[i] >= 0.5 ? 1 : 0) === p.label).length / Math.max(1, pts.length)
        : pts.reduce((a, p, i) => a + (pred[i] - p.label) ** 2, 0) / Math.max(1, pts.length);
    const l: Live = task === "classification"
      ? { trainAcc: metric(ptr, split.train), testAcc: metric(pte, split.test) }
      : { trainMse: metric(ptr, split.train), testMse: metric(pte, split.test) };
    setLive(l);
    if (task === "classification") setGrid(Float32Array.from(g));
    else setCurve(Array.from(g, (y, i) => ({ x: -1.25 + (i * 2.5) / 99, y })));
    if (params <= 400) setWeights(denseKernels(m));
    if (log) {
      checkChallenges({
        dataset: cfg.kind, hiddenLayers: cfg.hidden, numPoints: cfg.n, l2: cfg.l2, epoch: log.epoch, features: cfg.features,
        trainAcc: l.trainAcc, testAcc: l.testAcc, trainLoss: l.trainMse ?? log.loss, testLoss: l.testMse ?? log.valLoss,
      });
    }
  }, [task, split, params, cfg, checkChallenges]);

  const ensureRunner = useCallback((): TrainingRunner | null => {
    if (runnerRef.current) return runnerRef.current;
    try {
      const model = buildModel(modelConfig, LIMITS.playground);
      const tr = toXY(split.train), te = toXY(split.test);
      const gridPts = task === "classification" ? gridCoords() : Array.from({ length: 100 }, (_, i) => [-1.25 + (i * 2.5) / 99, 0] as [number, number]);
      const gridX = tf.tensor2d(gridPts.map(([a, b]) => featurize(a, b, cfg.features)), [gridPts.length, cfg.features.length]);
      tensorsRef.current = [tr.x, tr.y, te.x, te.y, gridX];
      modelRef.current = model;
      const runner = new TrainingRunner(model, tr.x, tr.y, {
        epochs: MAX_RUN_EPOCHS,
        batchSize: cfg.batchSize,
        seed,
        validation: split.test.length ? { x: te.x, y: te.y } : undefined,
        onStateChange: (st) => {
          setState(st);
          if (st === "paused" || st === "done" || st === "stopped") void refreshVisuals();
        },
        onEpochEnd: async (log) => {
          setHistory((h) => [...h, log]);
          if (log.epoch === 1 || log.epoch % VIS_EVERY === 0) await refreshVisuals(log);
          if (log.epoch === 1) unlockBadge("first-run");
        },
      });
      runnerRef.current = runner;
      return runner;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not build the model.");
      return null;
    }
  }, [modelConfig, toXY, split, task, cfg.features, cfg.batchSize, seed, refreshVisuals]);

  const onPlay = async () => {
    setError(null);
    const r = ensureRunner();
    if (!r) return;
    await r.run();
    if (r.error) setError(friendlyError(r.error));
    await refreshVisuals();
  };
  const onPause = () => runnerRef.current?.pause();
  const onStep = async () => {
    const r = ensureRunner();
    if (!r) return;
    if (r.state === "running") return;
    void r.step();
  };

  const onSave = () => {
    const r = runnerRef.current;
    if (!r || !history.length) return;
    saveRun({
      lab: "playground",
      name: `${TOY_DATASETS.find((d) => d.id === cfg.kind)!.name} · ${cfg.hidden.length} hidden`,
      durationMs: Math.round(r.elapsedMs),
      backend: activeBackend(),
      architecture: describeArchitecture({ inputShape: [cfg.features.length], layers: modelConfig.layers as never }),
      params,
      hyperparams: { learningRate: cfg.learningRate, batchSize: cfg.batchSize, l2: cfg.l2, optimizer: cfg.optimizer, activation: cfg.activation, noise: cfg.noise, epochs: history.length },
      dataset: { name: cfg.kind, nTrain: split.train.length, nTest: split.test.length, features: cfg.features },
      history,
      metrics: Object.fromEntries(Object.entries({
        trainAccuracy: live.trainAcc, testAccuracy: live.testAcc, trainMse: live.trainMse, testMse: live.testMse,
        trainLoss: history[history.length - 1]?.loss, testLoss: history[history.length - 1]?.valLoss,
      }).filter(([, v]) => v !== undefined && Number.isFinite(v))) as Record<string, number>,
    });
    setSaved(true);
  };

  // Any change to data or architecture invalidates the current model.
  const update = (patch: Partial<PlaygroundPreset>) => {
    reset();
    setCfg((c) => ({ ...c, ...patch }));
  };
  const toggleFeature = (f: FeatureId) =>
    update({ features: cfg.features.includes(f) ? (cfg.features.length > 1 ? cfg.features.filter((x) => x !== f) : cfg.features) : [...cfg.features, f] });

  const last = history[history.length - 1];
  const chartData = history.map((h) => ({ epoch: h.epoch, train: h.loss, test: h.valLoss }));
  const running = state === "running";
  const neurons = cfg.hidden.reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-4">
      <Celebrate badge={celebrate} onDone={() => setCelebrate(null)} />
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">🎯 Neural Playground</h1>
          <p className="text-muted">Pick a puzzle, design a network, press ▶ and watch it learn — live.</p>
        </div>
        <EnginePanel engine={engine} stats={stats} />
      </div>

      {error && <p role="alert" className="panel border-amber p-3 text-amber">⚠️ {error}</p>}

      <div className="grid gap-4 lg:grid-cols-[280px_1fr_320px]">
        {/* ---------- Data ---------- */}
        <section className="panel space-y-4 p-4" aria-labelledby="pg-data">
          <h2 id="pg-data" className="font-semibold">1 · Data</h2>
          <div className="grid grid-cols-2 gap-2">
            {TOY_DATASETS.map((d) => (
              <button key={d.id} aria-pressed={cfg.kind === d.id} onClick={() => update({ kind: d.id })}
                className={`btn btn-sm justify-start ${cfg.kind === d.id ? "btn-primary" : ""}`}>{d.emoji} {d.name.replace(" (regression)", "")}</button>
            ))}
          </div>
          <Slider label={`Noise: ${cfg.noise.toFixed(2)}`} min={0} max={0.5} step={0.01} value={cfg.noise} onChange={(v) => update({ noise: v })} />
          <Slider label={`Points: ${cfg.n}`} min={20} max={600} step={10} value={cfg.n} onChange={(v) => update({ n: v })} />
          <Slider label={`Train / test: ${cfg.trainPct}% / ${100 - cfg.trainPct}%`} min={20} max={90} step={5} value={cfg.trainPct} onChange={(v) => update({ trainPct: v })} />
          <button className="btn btn-sm w-full" onClick={() => { reset(); setSeed((s) => s + 1); }}>🎲 Regenerate (seed {seed})</button>
          <div>
            <p className="label">Input features</p>
            <div className="flex flex-wrap gap-1.5">
              {FEATURES.map((f) => (
                <button key={f.id} aria-pressed={cfg.features.includes(f.id)} onClick={() => toggleFeature(f.id)}
                  className={`btn btn-sm mono ${cfg.features.includes(f.id) ? "btn-primary" : ""}`}>{f.label}</button>
              ))}
            </div>
          </div>
        </section>

        {/* ---------- Visual ---------- */}
        <section className="space-y-3" aria-labelledby="pg-vis">
          <h2 id="pg-vis" className="sr-only">Training visualisation</h2>
          <div className="panel flex flex-wrap items-center gap-2 p-3">
            {running ? (
              <button className="btn btn-primary" onClick={onPause} aria-label="Pause training">⏸ Pause</button>
            ) : (
              <button className="btn btn-primary" onClick={onPlay} disabled={engine.status !== "ready" || state === "done"} aria-label="Start or resume training">▶ {state === "paused" ? "Resume" : "Train"}</button>
            )}
            <button className="btn" onClick={onStep} disabled={running || engine.status !== "ready"} aria-label="Train one epoch">⏭ Step</button>
            <button className="btn" onClick={reset} aria-label="Reset the model">↺ Reset</button>
            <button className="btn" onClick={onSave} disabled={running || !history.length || saved}>{saved ? "✓ Saved" : "💾 Save run"}</button>
            <span className="mono ml-auto text-sm" aria-live="polite">
              epoch <b className="text-cyan">{history.length}</b> · {state}
            </span>
          </div>
          {history.length > 0 && !running && (
            <ModelSaveBar getModel={() => modelRef.current} meta={() => ({
              app: "neuralnest", version: 1, name: `Playground ${cfg.kind}`, lab: "playground", task, createdAt: Date.now(),
              inputShape: [cfg.features.length], features: cfg.features, classNames: task === "classification" ? ["orange (0)", "blue (1)"] : undefined,
              featureExtractor: "none", metrics: Object.fromEntries(Object.entries({ testAccuracy: live.testAcc, testMse: live.testMse }).filter(([, v]) => v !== undefined)) as Record<string, number>,
            })} />
          )}
          <div className="flex justify-center">
            <DecisionBoundary grid={grid} curve={curve} train={split.train} test={split.test} task={task} showTest={showTest} />
          </div>
          <label className="flex items-center justify-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={showTest} onChange={(e) => setShowTest(e.target.checked)} /> Show test points (rings)
          </label>
          <div className="grid grid-cols-2 gap-2 text-center sm:grid-cols-4" aria-live="polite">
            {task === "classification" ? (
              <>
                <Stat label="Train accuracy" value={pct(live.trainAcc)} />
                <Stat label="Test accuracy" value={pct(live.testAcc)} accent />
              </>
            ) : (
              <>
                <Stat label="Train MSE" value={num(live.trainMse)} />
                <Stat label="Test MSE" value={num(live.testMse)} accent />
              </>
            )}
            <Stat label="Train loss" value={num(last?.loss)} />
            <Stat label="Test loss" value={num(last?.valLoss)} />
          </div>
          <LineChartPanel title="Loss over time (lower is better)" data={chartData} xKey="epoch"
            series={[{ key: "train", label: "Training loss", color: "var(--cyan)" }, { key: "test", label: "Test loss", color: "var(--amber)" }]} />
        </section>

        {/* ---------- Network ---------- */}
        <section className="panel space-y-4 p-4" aria-labelledby="pg-net">
          <h2 id="pg-net" className="font-semibold">2 · Network</h2>
          <NetworkDiagram sizes={[cfg.features.length, ...cfg.hidden, 1]} weights={weights} training={running}
            labels={["in", ...cfg.hidden.map((_, i) => `h${i + 1}`), "out"]} />
          <p className="mono text-xs text-muted">{describeArchitecture({ inputShape: [cfg.features.length], layers: modelConfig.layers as never })}</p>
          <p className="text-sm">Parameters: <b className="mono text-cyan">{params}</b> · hidden neurons: <b className="mono">{neurons}</b></p>
          <div className="space-y-2">
            {cfg.hidden.map((u, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-16 text-xs text-muted">Layer {i + 1}</span>
                <button className="btn btn-sm" aria-label={`Fewer neurons in layer ${i + 1}`} disabled={u <= 1} onClick={() => update({ hidden: cfg.hidden.map((x, j) => (j === i ? x - 1 : x)) })}>−</button>
                <span className="mono w-6 text-center">{u}</span>
                <button className="btn btn-sm" aria-label={`More neurons in layer ${i + 1}`} disabled={u >= LIMITS.playground.maxUnits} onClick={() => update({ hidden: cfg.hidden.map((x, j) => (j === i ? x + 1 : x)) })}>+</button>
              </div>
            ))}
            <div className="flex gap-2">
              <button className="btn btn-sm" disabled={cfg.hidden.length >= 4} onClick={() => update({ hidden: [...cfg.hidden, 4] })}>+ Layer</button>
              <button className="btn btn-sm" disabled={cfg.hidden.length === 0} onClick={() => update({ hidden: cfg.hidden.slice(0, -1) })}>− Layer</button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Select label="Activation" value={cfg.activation} options={["relu", "tanh", "sigmoid", "linear"]} onChange={(v) => update({ activation: v as Activation })} />
            <Select label="Optimizer" value={cfg.optimizer} options={["adam", "sgd"]} onChange={(v) => update({ optimizer: v as "adam" | "sgd" })} />
            <Select label="Learning rate" value={String(cfg.learningRate)} options={["0.0005", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "2.5"]} onChange={(v) => update({ learningRate: Number(v) })} />
            <Select label="Batch size" value={String(cfg.batchSize)} options={["1", "4", "10", "16", "32"]} onChange={(v) => update({ batchSize: Number(v) })} />
            <Select label="L2 regularisation" value={String(cfg.l2)} options={["0", "0.0001", "0.001", "0.003", "0.01", "0.03", "0.1"]} onChange={(v) => update({ l2: Number(v) })} />
          </div>
        </section>
      </div>

      <TutorPanel lab="playground" getSummary={() => history.length ? buildRunSummary({
        lab: "playground", task, params,
        architecture: describeArchitecture({ inputShape: [cfg.features.length], layers: modelConfig.layers as never }),
        hyperparams: { learningRate: cfg.learningRate, batchSize: cfg.batchSize, l2: cfg.l2, optimizer: cfg.optimizer, activation: cfg.activation, noise: cfg.noise, epochs: history.length },
        dataset: { name: cfg.kind, nTrain: split.train.length, nTest: split.test.length, features: cfg.features, ...(task === "classification" ? { classNames: ["orange (0)", "blue (1)"], classCounts: [0, 1].map((c) => split.train.filter((p) => p.label === c).length) } : {}) },
        history, metrics: { trainAccuracy: live.trainAcc, testAccuracy: live.testAcc, trainMse: live.trainMse, testMse: live.testMse, trainLoss: last?.loss, testLoss: last?.valLoss },
      }) : null} onApplyPatch={(p) => update({
        ...(p.learningRate !== undefined ? { learningRate: p.learningRate } : {}),
        ...(p.batchSize !== undefined ? { batchSize: nearestBatch(p.batchSize, [1, 4, 10, 16, 32]) } : {}),
        ...(p.l2 !== undefined ? { l2: p.l2 } : {}),
        ...(p.hiddenLayers ? { hidden: p.hiddenLayers } : {}),
        ...(p.activation ? { activation: p.activation } : {}),
        ...(p.optimizer ? { optimizer: p.optimizer } : {}),
      })} />

      {/* ---------- Challenges ---------- */}
      <section className="panel p-4" aria-labelledby="pg-ch">
        <h2 id="pg-ch" className="mb-2 font-semibold">🏆 Challenges <span className="text-xs font-normal text-muted">(checked automatically from your real test metrics)</span></h2>
        {hint && <p className="mb-2 text-sm text-amber" role="status">💡 {hint}</p>}
        <ul className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
          {CHALLENGES.map((c) => (
            <li key={c.id} className={`rounded-xl border p-3 text-sm ${solved[c.id] ? "border-good" : "border-border"}`}>
              <p className="font-semibold">{c.badge.emoji} {c.title} {solved[c.id] && <span className="text-good">✓ solved</span>}</p>
              <p className="mt-1 text-muted">{c.description}</p>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function friendlyError(e: Error): string {
  if (/memory|OOM|allocate/i.test(e.message)) return "Your device ran out of memory. Try fewer points or a smaller network, then press Reset.";
  return "Training stopped unexpectedly. Press Reset and try again.";
}
const pct = (v?: number) => (v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
const num = (v?: number) => (v === undefined || Number.isNaN(v) ? "–" : v.toFixed(4));

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="panel px-2 py-2">
      <p className="text-[11px] text-muted">{label}</p>
      <p className={`mono text-lg font-semibold ${accent ? "text-cyan" : ""}`}>{value}</p>
    </div>
  );
}

function Slider({ label, min, max, step, value, onChange }: { label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
        {[...new Set([...options, value])].map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );
}

