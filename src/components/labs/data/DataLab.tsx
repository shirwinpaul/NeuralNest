"use client";
import Papa from "papaparse";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { DATA_LIMITS, LIMITS } from "@/config/limits";
import { useEngine } from "@/hooks/useEngine";
import { buildModel, countParams, describeArchitecture, disposeModel, validateConfig, type Activation } from "@/lib/ml/builder";
import { oneHot, splitIndices } from "@/lib/ml/data";
import {
  baselineClassification, baselineRegression, DEFAULT_NN, evaluateClassification, evaluateRegression, MODEL_LABEL, tabularModelConfig,
  type Eval, type NNSpec, type TabModelKind,
} from "@/lib/ml/datalab";
import { SAMPLE_DATASETS, loadSample } from "@/lib/ml/datasets/tabular";
import { activeBackend } from "@/lib/ml/engine";
import { permutationImportance, type ImportanceResult } from "@/lib/ml/importance";
import { accuracy, r2 } from "@/lib/ml/metrics";
import { classBalance, column, dataWarnings, detectTask, profileTable, type ColumnType, type Table, type Task } from "@/lib/ml/profile";
import { saveRun } from "@/lib/ml/storage";
import { denormY, encodeFeatures, encodeRows, featureGroups, fitPipeline, type FittedPipeline } from "@/lib/ml/tabular";
import { predictArray, TrainingRunner, type EpochLog } from "@/lib/ml/trainer";
import { unlockBadge } from "@/lib/progress";
import { downloadText, toCSV } from "@/lib/download";
import { EnginePanel } from "../EnginePanel";
import { LineChartPanel } from "@/components/charts/LineChartPanel";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import { HBarPanel, ScatterPanel } from "@/components/charts/Charts";
import { ProfileView } from "./ProfileView";
import { PredictForm, type PredictResult } from "./PredictForm";
import { ModelCard } from "./ModelCard";
import { ModelSaveBar } from "../ModelSaveBar";
import { TutorPanel } from "@/components/tutor/TutorPanel";
import { buildRunSummary } from "@/lib/ai/summary";

const SEED = 42;
const PRESET_SAMPLE: Record<string, string> = { study: "study", iris: "iris", fruit: "fruit", cricket: "cricket" };

interface ModelResult {
  kind: TabModelKind;
  eval: Eval;
  history: EpochLog[];
  durationMs: number;
  architecture: string;
  params: number;
}

interface Prepared {
  pipeline: FittedPipeline;
  split: { train: number[]; val: number[]; test: number[] };
  test: ReturnType<typeof encodeRows>;
  counts: { train: number; val: number; test: number };
}

export default function DataLab({ presetId }: { presetId: string | null }) {
  const { engine, stats } = useEngine();
  const [table, setTable] = useState<Table | null>(null);
  const [source, setSource] = useState<{ name: string; credit: string }>({ name: "", credit: "" });
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [taskOverride, setTaskOverride] = useState<Task | null>(null);
  const [features, setFeatures] = useState<string[]>([]);
  const [norm, setNorm] = useState<"none" | "minmax" | "zscore">("zscore");
  const [missing, setMissing] = useState<"drop" | "mean" | "mode">("mean");
  const [maxCats, setMaxCats] = useState(10);
  const [nn, setNn] = useState<NNSpec>(DEFAULT_NN);
  const [earlyStop, setEarlyStop] = useState(true);
  const [training, setTraining] = useState<TabModelKind | null>(null);
  const [liveHistory, setLiveHistory] = useState<Record<string, EpochLog[]>>({});
  const [results, setResults] = useState<Partial<Record<TabModelKind, ModelResult>>>({});
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [selected, setSelected] = useState<TabModelKind>("nn");
  const [importance, setImportance] = useState<ImportanceResult[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [trainErr, setTrainErr] = useState<string | null>(null);

  const modelsRef = useRef<Partial<Record<"linear" | "nn", tf.Sequential>>>({});
  const runnerRef = useRef<TrainingRunner | null>(null);

  const disposeAll = useCallback(() => {
    runnerRef.current?.stop();
    Object.values(modelsRef.current).forEach((m) => disposeModel(m));
    modelsRef.current = {};
  }, []);
  useEffect(() => disposeAll, [disposeAll]);

  // ---------- loading ----------
  const resetResults = useCallback(() => {
    disposeAll();
    setResults({});
    setPrepared(null);
    setImportance(null);
    setLiveHistory({});
    setTrainErr(null);
  }, [disposeAll]);

  const applyTable = useCallback((t: Table, name: string, credit: string, preferTarget?: string) => {
    resetResults();
    const p = profileTable(t);
    const tgt = preferTarget && t.columns.includes(preferTarget) ? preferTarget : t.columns[t.columns.length - 1];
    setTable(t);
    setSource({ name, credit });
    setTarget(tgt);
    setTaskOverride(null);
    const warnCols = new Set(dataWarnings(t, p, tgt).filter((w) => w.kind === "id-column" || w.kind === "identical-to-target").map((w) => w.column));
    setFeatures(p.columns.filter((c) => c.name !== tgt && c.type !== "text" && !warnCols.has(c.name)).map((c) => c.name));
  }, [resetResults]);

  const loadSampleById = useCallback(async (id: string) => {
    setLoadErr(null);
    try {
      const s = SAMPLE_DATASETS.find((d) => d.id === id)!;
      applyTable(await loadSample(id), s.name, s.source, s.target);
    } catch {
      setLoadErr("Couldn't load that sample. Check your connection and try again.");
    }
  }, [applyTable]);

  useEffect(() => {
    const id = presetId ? PRESET_SAMPLE[presetId] : null;
    if (!id) return;
    // Load the preset sample once the page has mounted (async, outside the render).
    const t = setTimeout(() => void loadSampleById(id), 0);
    return () => clearTimeout(t);
  }, [presetId, loadSampleById]);

  const onFile = (file: File) => {
    setLoadErr(null);
    if (file.size > DATA_LIMITS.csvMaxBytes) return setLoadErr(`That file is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is 5 MB.`);
    Papa.parse<string[]>(file, {
      skipEmptyLines: true,
      complete: (res) => {
        const data = res.data.filter((r) => Array.isArray(r) && r.some((c) => String(c).trim() !== ""));
        if (data.length < 2) return setLoadErr("That CSV has no data rows (the first row must be the column names).");
        const columns = data[0].map((c, i) => String(c).trim() || `column_${i + 1}`);
        if (columns.length > DATA_LIMITS.csvMaxCols) return setLoadErr(`Too many columns (${columns.length}); the limit is ${DATA_LIMITS.csvMaxCols}.`);
        if (new Set(columns).size !== columns.length) return setLoadErr("Two columns have the same name — please rename one.");
        let rows = data.slice(1).map((r) => columns.map((_, j) => String(r[j] ?? "")));
        const truncated = rows.length > DATA_LIMITS.csvMaxRows;
        if (truncated) rows = rows.slice(0, DATA_LIMITS.csvMaxRows);
        applyTable({ columns, rows }, file.name.slice(0, 80), "Your own file (never leaves this browser)");
        if (truncated) setLoadErr(`Only the first ${DATA_LIMITS.csvMaxRows.toLocaleString()} rows were loaded.`);
      },
      error: () => setLoadErr("Couldn't read that CSV file. Is it a plain .csv saved from a spreadsheet?"),
    });
  };

  // ---------- profile ----------
  const profile = useMemo(() => (table ? profileTable(table) : null), [table]);
  const types = useMemo<Record<string, ColumnType>>(() => Object.fromEntries((profile?.columns ?? []).map((c) => [c.name, c.type])), [profile]);
  const detectedTask: Task | null = useMemo(() => {
    if (!table || !profile || !target) return null;
    return detectTask(profile.columns.find((c) => c.name === target)!, column(table, target));
  }, [table, profile, target]);
  const task = taskOverride ?? detectedTask ?? "classification";
  const warnings = useMemo(() => (table && profile ? dataWarnings(table, profile, target, task) : []), [table, profile, target, task]);
  const balance = useMemo(() => (table && target && task === "classification" ? classBalance(column(table, target)) : null), [table, target, task]);
  const regressionTargetOk = task !== "regression" || types[target ?? ""] === "numeric";
  const tooManyClasses = task === "classification" && (balance?.length ?? 0) > 20;

  const changeSetup = (fn: () => void) => {
    resetResults();
    fn();
  };

  // ---------- training ----------
  const prepare = (): Prepared | null => {
    if (!table || !target) return null;
    const ti = table.columns.indexOf(target);
    const labels = task === "classification" ? (() => {
      const names = [...new Set(table.rows.map((r) => r[ti].trim()))].sort();
      return table.rows.map((r) => names.indexOf(r[ti].trim()));
    })() : undefined;
    const split = splitIndices(table.rows.length, { train: 70, val: 15, test: 15 }, SEED, labels);
    const pipeline = fitPipeline(table, split.train, { target, features, types, task, norm, missing, maxCategories: maxCats });
    const test = encodeRows(table, split.test, pipeline);
    return { pipeline, split, test, counts: { train: 0, val: 0, test: test.X.length } };
  };

  const trainAll = async () => {
    setTrainErr(null);
    resetResults();
    const prep = prepare();
    if (!prep) return;
    const { pipeline, split, test } = prep;
    const tr = encodeRows(table!, split.train, pipeline);
    const va = encodeRows(table!, split.val, pipeline);
    prep.counts = { train: tr.X.length, val: va.X.length, test: test.X.length };
    if (tr.X.length < 10 || test.X.length < 3) {
      setTrainErr("Not enough usable rows after preprocessing. Try ‘fill with mean/mode’ instead of ‘drop’, or use more data.");
      return;
    }
    if (pipeline.encodedNames.length === 0) {
      setTrainErr("Pick at least one numeric or categorical feature.");
      return;
    }
    setPrepared(prep);
    const nClasses = pipeline.classNames?.length ?? 1;
    const out: Partial<Record<TabModelKind, ModelResult>> = {};
    const actual = (e: typeof test) => e.y.map((v) => denormY(v, pipeline));

    // Baseline
    const t0 = performance.now();
    out.baseline = {
      kind: "baseline",
      eval: task === "classification" ? baselineClassification(tr.y, test.y, nClasses) : baselineRegression(actual(tr), actual(test)),
      history: [], durationMs: performance.now() - t0,
      architecture: task === "classification" ? "always predict the most common class" : "always predict the training mean",
      params: 0,
    };
    setResults({ ...out });

    const mk = (X: number[][]) => tf.tensor2d(X, [X.length, pipeline.encodedNames.length]);
    const mkY = (y: number[]) => (task === "classification" ? tf.tensor2d(oneHot(y, nClasses)) : tf.tensor2d(y, [y.length, 1]));
    const xs = { tr: mk(tr.X), va: mk(va.X.length ? va.X : tr.X.slice(0, 1)), te: mk(test.X) };
    const ys = { tr: mkY(tr.y), va: mkY(va.y.length ? va.y : tr.y.slice(0, 1)) };
    try {
      for (const kind of ["linear", "nn"] as const) {
        setTraining(kind);
        const cfg = tabularModelConfig(kind, pipeline.encodedNames.length, task, nClasses, nn, SEED);
        const v = validateConfig(cfg, LIMITS.data);
        if (!v.ok) throw new Error(v.errors.join(" "));
        const model = buildModel(cfg, LIMITS.data);
        modelsRef.current[kind] = model;
        const start = performance.now();
        const runner = new TrainingRunner(model, xs.tr, ys.tr, {
          epochs: nn.epochs, batchSize: nn.batchSize, seed: SEED,
          validation: va.X.length ? { x: xs.va, y: ys.va } : undefined,
          earlyStopping: earlyStop ? { patience: 15, minDelta: 1e-4 } : undefined,
          onEpochEnd: (log) => setLiveHistory((h) => ({ ...h, [kind]: [...(h[kind] ?? []), log] })),
        });
        runnerRef.current = runner;
        const history = await runner.run();
        if (runner.error) throw runner.error;
        const probs = await predictArray(model, xs.te);
        const ev = task === "classification" ? evaluateClassification(test.y, probs, nClasses) : evaluateRegression(actual(test), probs.map((p) => denormY(p[0], pipeline)));
        out[kind] = { kind, eval: ev, history, durationMs: performance.now() - start, architecture: describeArchitecture(v.config), params: countParams(v.config) };
        setResults({ ...out });
      }
      unlockBadge("first-run");
      unlockBadge("data-scientist");
      for (const k of ["baseline", "linear", "nn"] as const) {
        const r = out[k]!;
        saveRun({
          lab: "data", name: `${source.name} · ${MODEL_LABEL[k]}`, durationMs: Math.round(r.durationMs), backend: activeBackend(),
          architecture: r.architecture, params: r.params,
          hyperparams: k === "baseline" ? { model: "baseline" } : { learningRate: nn.learningRate, batchSize: nn.batchSize, epochs: r.history.length, l2: nn.l2, dropout: nn.dropout, norm, missing },
          dataset: { name: source.name, nTrain: tr.X.length, nTest: test.X.length, classNames: pipeline.classNames?.slice(0, 20), features: pipeline.features.map((f) => f.name).slice(0, 60) },
          history: r.history, metrics: prefixed(r.eval.metrics),
        });
      }
    } catch (e) {
      setTrainErr(/memory|allocate/i.test(String(e)) ? "Ran out of memory — try fewer features, a smaller network or fewer rows." : `Training failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      tf.dispose([xs.tr, xs.va, xs.te, ys.tr, ys.va]);
      setTraining(null);
      runnerRef.current = null;
    }
  };

  // ---------- importance & predict ----------
  const computeImportance = async () => {
    const m = selected !== "baseline" ? modelsRef.current[selected] : undefined;
    if (!m || !prepared) return;
    setBusy("Computing importance…");
    const { pipeline, test } = prepared;
    const groups = featureGroups(pipeline);
    const y = task === "classification" ? test.y : test.y.map((v) => denormY(v, pipeline));
    const score = async (X: number[][]) => {
      const t = tf.tensor2d(X, [X.length, pipeline.encodedNames.length]);
      const p = await predictArray(m, t);
      t.dispose();
      return task === "classification" ? accuracy(y, p.map((r) => r.indexOf(Math.max(...r)))) : r2(y, p.map((r) => denormY(r[0], pipeline)));
    };
    try {
      setImportance(await permutationImportance({ X: test.X, groups, names: pipeline.features.map((f) => f.name), score, repeats: 3, seed: SEED }));
    } finally {
      setBusy(null);
    }
  };

  const predictOne = async (vals: Record<string, string>): Promise<PredictResult | null> => {
    const m = selected !== "baseline" ? modelsRef.current[selected] : modelsRef.current.nn;
    if (!m || !prepared) return null;
    const x = encodeFeatures((n) => vals[n], prepared.pipeline);
    if (!x) return null;
    const t = tf.tensor2d([x]);
    const [p] = await predictArray(m, t);
    t.dispose();
    return task === "classification" ? { kind: "class", probs: p, classNames: prepared.pipeline.classNames! } : { kind: "value", value: denormY(p[0], prepared.pipeline) };
  };

  const exportPredictions = () => {
    const r = results[selected];
    if (!r || !prepared || !table) return;
    const { pipeline, test } = prepared;
    const header = [...pipeline.features.map((f) => f.name), `actual_${pipeline.target}`, `predicted_${pipeline.target}`, ...(task === "classification" ? ["confidence"] : [])];
    const rows = test.keptRows.map((ri, k) => {
      const raw = table.rows[ri];
      const feats = pipeline.features.map((f) => raw[table.columns.indexOf(f.name)]);
      if (r.eval.task === "classification") {
        const pred = r.eval.predictions[k];
        return [...feats, test.rawY[k], pipeline.classNames![pred], r.eval.probs[k][pred].toFixed(4)];
      }
      return [...feats, test.rawY[k], r.eval.predictions[k].toFixed(4)];
    });
    downloadText(`predictions-${selected}.csv`, toCSV(header, rows), "text/csv");
  };

  // ---------- render ----------
  const sel = results[selected];
  const formDefaults = useMemo(() => {
    if (!prepared || !table) return {};
    const first = table.rows[prepared.test.keptRows[0] ?? 0];
    return Object.fromEntries(prepared.pipeline.features.map((f) => [f.name, f.kind === "categorical" ? f.categories![0] ?? "" : first?.[table.columns.indexOf(f.name)] ?? ""]));
  }, [prepared, table]);
  const curveSeries = (["linear", "nn"] as const).flatMap((k) => [
    { key: `${k}_train`, label: `${MODEL_LABEL[k]} train`, color: k === "nn" ? "var(--cyan)" : "var(--violet)" },
    { key: `${k}_val`, label: `${MODEL_LABEL[k]} validation`, color: k === "nn" ? "var(--amber)" : "#CC79A7" },
  ]);
  const curveData = useMemo(() => {
    const len = Math.max(0, ...Object.values(liveHistory).map((h) => h.length));
    return Array.from({ length: len }, (_, i) => {
      const row: Record<string, number | undefined> = { epoch: i + 1 };
      for (const k of ["linear", "nn"]) {
        row[`${k}_train`] = liveHistory[k]?.[i]?.loss;
        row[`${k}_val`] = liveHistory[k]?.[i]?.valLoss;
      }
      return row;
    });
  }, [liveHistory]);
  const metricRows = task === "classification"
    ? [["accuracy", "Accuracy", pct], ["macroPrecision", "Precision (macro)", pct], ["macroRecall", "Recall (macro)", pct], ["macroF1", "F1 (macro)", pct], ["logLoss", "Log loss", n3]] as const
    : [["mae", "MAE", n3], ["rmse", "RMSE", n3], ["mse", "MSE", n3], ["r2", "R²", n3]] as const;
  const nnParams = useMemo(() => {
    try {
      const dim = prepared?.pipeline.encodedNames.length ?? features.length;
      return countParams(tabularModelConfig("nn", Math.max(1, dim), task, balance?.length ?? 1, nn, SEED) as never);
    } catch {
      return 0;
    }
  }, [prepared, features.length, task, balance, nn]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">📊 Data Lab</h1>
          <p className="text-muted">Load a table, explore it, then compare a baseline, a linear model and a neural network — on the same seeded split.</p>
        </div>
        <EnginePanel engine={engine} stats={stats} />
      </div>

      {/* 1. Load */}
      <section className="panel space-y-3 p-4" aria-labelledby="dl-load">
        <h2 id="dl-load" className="font-semibold">1 · Load data</h2>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_DATASETS.map((s) => (
            <button key={s.id} className={`btn btn-sm ${source.name === s.name ? "btn-primary" : ""}`} onClick={() => loadSampleById(s.id)} title={s.description}>{s.emoji} {s.name}</button>
          ))}
          <label className="btn btn-sm">
            📁 Upload CSV (≤ 5 MB)
            <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          </label>
        </div>
        <p className="text-xs text-muted">🔒 Your file is read and processed only in this browser. The AI tutor never sees rows — only column names, types and summary numbers.</p>
        {loadErr && <p role="alert" className="text-sm text-amber">⚠️ {loadErr}</p>}
        {source.name && <p className="text-sm">Loaded <b>{source.name}</b> <span className="text-xs text-muted">— {source.credit}</span></p>}
      </section>

      {table && profile && (
        <>
          {/* 2. Profile */}
          <section className="panel space-y-3 p-4" aria-labelledby="dl-profile">
            <h2 id="dl-profile" className="font-semibold">2 · Data profile</h2>
            <ProfileView profile={profile} warnings={warnings} balance={balance} target={target} />
          </section>

          {/* 3. Setup */}
          <section className="panel space-y-4 p-4" aria-labelledby="dl-setup">
            <h2 id="dl-setup" className="font-semibold">3 · Set up the experiment</h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <label className="block">
                <span className="label">Target column (what to predict)</span>
                <select className="select" value={target ?? ""} onChange={(e) => changeSetup(() => { setTarget(e.target.value); setTaskOverride(null); setFeatures((f) => f.filter((x) => x !== e.target.value)); })}>
                  {table.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label className="block">
                <span className="label">Task (auto-detected: {detectedTask})</span>
                <select className="select" value={task} onChange={(e) => changeSetup(() => setTaskOverride(e.target.value as Task))}>
                  <option value="classification">Classification (categories)</option>
                  <option value="regression">Regression (numbers)</option>
                </select>
              </label>
              <label className="block">
                <span className="label">Normalisation</span>
                <select className="select" value={norm} onChange={(e) => changeSetup(() => setNorm(e.target.value as typeof norm))}>
                  <option value="zscore">z-score (mean 0, sd 1)</option>
                  <option value="minmax">min-max (0 … 1)</option>
                  <option value="none">none</option>
                </select>
              </label>
              <label className="block">
                <span className="label">Missing values</span>
                <select className="select" value={missing} onChange={(e) => changeSetup(() => setMissing(e.target.value as typeof missing))}>
                  <option value="mean">fill numbers with mean, categories with mode</option>
                  <option value="mode">fill with most common value</option>
                  <option value="drop">drop rows with missing values</option>
                </select>
              </label>
            </div>
            {!regressionTargetOk && <p className="text-sm text-amber" role="alert">Regression needs a numeric target column.</p>}
            {tooManyClasses && <p className="text-sm text-amber" role="alert">That target has more than 20 categories — pick another target or use regression.</p>}
            <fieldset>
              <legend className="label">Features to use</legend>
              <div className="flex flex-wrap gap-2">
                {profile.columns.filter((c) => c.name !== target).map((c) => (
                  <label key={c.name} className={`chip cursor-pointer ${c.type === "text" ? "opacity-50" : ""}`}>
                    <input type="checkbox" disabled={c.type === "text"} checked={features.includes(c.name)}
                      onChange={(e) => changeSetup(() => setFeatures((f) => (e.target.checked ? [...f, c.name] : f.filter((x) => x !== c.name))))} />
                    {c.name} <span className="text-[10px]">({c.type})</span>
                  </label>
                ))}
              </div>
              <label className="mt-2 flex items-center gap-2 text-xs text-muted">
                One-hot categories cap:
                <input type="number" min={2} max={DATA_LIMITS.maxCategories} value={maxCats} className="input w-16" onChange={(e) => changeSetup(() => setMaxCats(Math.min(DATA_LIMITS.maxCategories, Math.max(2, Number(e.target.value) || 2))))} />
                (rarer values grouped as “other”)
              </label>
            </fieldset>
            <fieldset className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <legend className="label">Neural network (the linear model uses the same epochs, batch and L2)</legend>
              <label className="block"><span className="label">Hidden layers</span>
                <input className="input mono" value={nn.hidden.join(",")} onChange={(e) => {
                  const h = e.target.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((v) => v > 0).slice(0, 4).map((v) => Math.min(LIMITS.data.maxUnits, v));
                  changeSetup(() => setNn({ ...nn, hidden: h }));
                }} aria-describedby="hl-help" /></label>
              <label className="block"><span className="label">Activation</span>
                <select className="select" value={nn.activation} onChange={(e) => changeSetup(() => setNn({ ...nn, activation: e.target.value as Activation }))}>{["relu", "tanh", "sigmoid"].map((a) => <option key={a}>{a}</option>)}</select></label>
              <label className="block"><span className="label">Epochs (max)</span>
                <input type="number" className="input mono" min={1} max={LIMITS.data.maxEpochs} value={nn.epochs} onChange={(e) => changeSetup(() => setNn({ ...nn, epochs: Math.min(LIMITS.data.maxEpochs, Math.max(1, Number(e.target.value) || 1)) }))} /></label>
              <label className="block"><span className="label">Learning rate</span>
                <select className="select" value={nn.learningRate} onChange={(e) => changeSetup(() => setNn({ ...nn, learningRate: Number(e.target.value) }))}>{[0.001, 0.003, 0.01, 0.03, 0.1].map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
              <label className="block"><span className="label">Batch size</span>
                <select className="select" value={nn.batchSize} onChange={(e) => changeSetup(() => setNn({ ...nn, batchSize: Number(e.target.value) }))}>{[8, 16, 32, 64, 128].map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
              <label className="block"><span className="label">Dropout</span>
                <select className="select" value={nn.dropout} onChange={(e) => changeSetup(() => setNn({ ...nn, dropout: Number(e.target.value) }))}>{[0, 0.1, 0.2, 0.3, 0.5].map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
              <label className="block"><span className="label">L2</span>
                <select className="select" value={nn.l2} onChange={(e) => changeSetup(() => setNn({ ...nn, l2: Number(e.target.value) }))}>{[0, 0.0001, 0.001, 0.01].map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
              <label className="flex items-end gap-2 pb-2 text-sm"><input type="checkbox" checked={earlyStop} onChange={(e) => setEarlyStop(e.target.checked)} /> Early stopping</label>
            </fieldset>
            <p id="hl-help" className="text-xs text-muted">Hidden layers: comma-separated sizes, e.g. 16,8 (max 4 layers, {LIMITS.data.maxUnits} units). NN parameters ≈ <b className="mono">{nnParams.toLocaleString()}</b></p>
            <button className="btn btn-primary" onClick={trainAll} disabled={!!training || engine.status !== "ready" || !features.length || !regressionTargetOk || tooManyClasses}>
              {training ? `Training ${MODEL_LABEL[training]}…` : "🚀 Train & compare all 3 models"}
            </button>
            {trainErr && <p role="alert" className="text-sm text-amber">⚠️ {trainErr}</p>}
            <p className="sr-only" aria-live="polite">{training ? `Training ${MODEL_LABEL[training]}` : Object.keys(results).length ? "Training finished" : ""}</p>
          </section>

          {(curveData.length > 0 || Object.keys(results).length > 0) && (
            <section className="panel space-y-4 p-4" aria-labelledby="dl-results">
              <h2 id="dl-results" className="font-semibold">4 · Results on the test set</h2>
              {prepared && <p className="text-xs text-muted">Seeded split (seed {SEED}{task === "classification" ? ", stratified" : ""}): {prepared.counts.train} train · {prepared.counts.val} validation · {prepared.counts.test} test rows. Preprocessing was fitted on training rows only.</p>}
              <LineChartPanel title="Learning curves (loss per epoch)" data={curveData} xKey="epoch" series={curveSeries} />
              <div className="overflow-x-auto">
                <table className="mono w-full text-sm">
                  <thead><tr className="text-left text-xs text-muted"><th className="p-1">Model</th>{metricRows.map(([, l]) => <th key={l} className="p-1 text-right">{l}</th>)}<th className="p-1 text-right">Params</th><th className="p-1 text-right">Time</th></tr></thead>
                  <tbody>
                    {(["baseline", "linear", "nn"] as const).map((k) => {
                      const r = results[k];
                      return (
                        <tr key={k} className={`border-t border-border ${selected === k ? "bg-cyan/10" : ""}`}>
                          <td className="p-1"><label className="flex items-center gap-2"><input type="radio" name="sel-model" checked={selected === k} onChange={() => { setSelected(k); setImportance(null); }} disabled={!r} />{MODEL_LABEL[k]}</label></td>
                          {metricRows.map(([key, l, fmt]) => <td key={l} className="p-1 text-right">{r ? fmt(r.eval.metrics[key]) : "…"}</td>)}
                          <td className="p-1 text-right">{r?.params ?? "–"}</td>
                          <td className="p-1 text-right">{r ? `${(r.durationMs / 1000).toFixed(1)}s` : ""}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {sel && prepared && (
                <div className="grid gap-4 lg:grid-cols-2">
                  {sel.eval.task === "classification" ? (
                    <div className="panel p-3">
                      <p className="mb-2 text-sm font-semibold">Confusion matrix — {MODEL_LABEL[selected]}</p>
                      <ConfusionMatrix matrix={sel.eval.confusion} labels={prepared.pipeline.classNames!} />
                      <table className="mono mt-3 w-full text-xs">
                        <thead><tr className="text-muted"><th className="text-left">class</th><th className="text-right">precision</th><th className="text-right">recall</th><th className="text-right">F1</th><th className="text-right">support</th></tr></thead>
                        <tbody>{sel.eval.perClass.map((c, i) => <tr key={i}><td>{prepared.pipeline.classNames![i]}</td><td className="text-right">{pct(c.precision)}</td><td className="text-right">{pct(c.recall)}</td><td className="text-right">{pct(c.f1)}</td><td className="text-right">{c.support}</td></tr>)}</tbody>
                      </table>
                    </div>
                  ) : (
                    <>
                      <ScatterPanel title={`Predicted vs actual — ${MODEL_LABEL[selected]}`} xLabel="actual" yLabel="predicted" diagonal data={sel.eval.actual.map((a, i) => ({ x: a, y: (sel.eval as { predictions: number[] }).predictions[i] }))} />
                      <ScatterPanel title="Residuals (actual − predicted)" xLabel="predicted" yLabel="residual" zeroLine data={sel.eval.actual.map((a, i) => ({ x: (sel.eval as { predictions: number[] }).predictions[i], y: a - (sel.eval as { predictions: number[] }).predictions[i] }))} />
                    </>
                  )}
                  {selected !== "baseline" && (
                    <div className="space-y-2">
                      <button className="btn btn-sm" onClick={computeImportance} disabled={!!busy}>{busy ?? "🔍 Permutation feature importance"}</button>
                      {importance && (
                        <HBarPanel title={`Feature importance — ${MODEL_LABEL[selected]}`} valueLabel={task === "classification" ? "accuracy drop" : "R² drop"}
                          data={importance.map((r) => ({ name: r.feature, value: r.importance, err: r.std }))}
                          note={`How much the test ${task === "classification" ? "accuracy" : "R²"} drops when a feature's values are shuffled (3 repeats). Bigger = the model relies on it more.`} />
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          <TutorPanel lab="data" getSummary={() => {
            const r = results[selected];
            if (!r || !prepared || !profile) return null;
            const pl = prepared.pipeline;
            return buildRunSummary({
              lab: "data", task, architecture: `${MODEL_LABEL[selected]}: ${r.architecture}`, params: r.params,
              hyperparams: { model: selected, learningRate: nn.learningRate, batchSize: nn.batchSize, epochs: r.history.length, l2: nn.l2, dropout: nn.dropout, normalisation: norm, missing },
              dataset: {
                name: source.name, nTrain: prepared.counts.train, nTest: prepared.counts.test, classNames: pl.classNames?.slice(0, 10),
                classCounts: balance?.slice(0, 10).map((b) => b.count),
                features: pl.features.map((f) => f.name), featureTypes: pl.features.map((f) => f.kind),
              },
              history: r.history, metrics: r.eval.metrics, baseline: results.baseline?.eval.metrics,
              confusion: r.eval.task === "classification" ? r.eval.confusion : undefined,
              notes: warnings.map((w) => w.message).slice(0, 5),
            });
          }} onApplyPatch={(p) => changeSetup(() => setNn((cur) => ({
            ...cur,
            ...(p.learningRate !== undefined ? { learningRate: p.learningRate } : {}),
            ...(p.batchSize !== undefined ? { batchSize: p.batchSize } : {}),
            ...(p.epochs !== undefined ? { epochs: p.epochs } : {}),
            ...(p.l2 !== undefined ? { l2: p.l2 } : {}),
            ...(p.dropout !== undefined ? { dropout: p.dropout } : {}),
            ...(p.hiddenLayers ? { hidden: p.hiddenLayers } : {}),
            ...(p.activation && p.activation !== "linear" ? { activation: p.activation } : {}),
          })))} />

          {prepared && results.nn && (
            <>
              <section className="panel space-y-3 p-4" aria-labelledby="dl-predict">
                <h2 id="dl-predict" className="font-semibold">5 · Predict a new example <span className="text-xs font-normal text-muted">(using {MODEL_LABEL[selected === "baseline" ? "nn" : selected]})</span></h2>
                <PredictForm key={selected} pipeline={prepared.pipeline} defaults={formDefaults} onPredict={predictOne} />
              </section>
              <section className="panel space-y-3 p-4" aria-labelledby="dl-export">
                <h2 id="dl-export" className="font-semibold">6 · Export</h2>
                <div className="flex flex-wrap gap-2">
                  <button className="btn btn-sm" onClick={exportPredictions} disabled={!sel}>⬇️ Test predictions (CSV)</button>
                </div>
                <ModelSaveBar getModel={() => modelsRef.current[selected === "baseline" ? "nn" : selected] ?? null} meta={() => {
                  const k = selected === "baseline" ? "nn" : selected;
                  return {
                    app: "neuralnest", version: 1, name: `${source.name} ${MODEL_LABEL[k]}`.slice(0, 80), lab: "data", task, createdAt: Date.now(),
                    inputShape: [prepared.pipeline.encodedNames.length], features: prepared.pipeline.encodedNames.slice(0, 200), classNames: prepared.pipeline.classNames?.slice(0, 20),
                    norm: prepared.pipeline.norm, preprocess: prepared.pipeline, featureExtractor: "none", metrics: prefixed(results[k]?.eval.metrics ?? {}),
                  };
                }} />
                <ModelCard data={{
                  modelName: `${source.name} — ${MODEL_LABEL[selected]}`,
                  datasetName: source.name, datasetSource: source.credit, task, target: target!,
                  features: prepared.pipeline.features.map((f) => f.name), rows: prepared.counts,
                  architecture: sel?.architecture ?? "",
                  metrics: sel ? metricRows.map(([k, l, fmt]) => ({ label: l, value: fmt(sel.eval.metrics[k]) })) : [],
                  baseline: results.baseline ? metricRows.slice(0, 1).map(([k, l, fmt]) => `${l} ${fmt(results.baseline!.eval.metrics[k])}`).join("") + ` (${results.baseline.architecture})` : "",
                  warnings: warnings.map((w) => w.message),
                }} />
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}

const pct = (v?: number) => (v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
const n3 = (v?: number) => (v === undefined ? "–" : v.toFixed(3));
function prefixed(m: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(m).filter(([, v]) => Number.isFinite(v)).map(([k, v]) => [`test${k[0].toUpperCase()}${k.slice(1)}`, v]));
}
