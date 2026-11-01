"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { DATA_LIMITS, LIMITS } from "@/config/limits";
import { classColor } from "@/config/palette";
import { useEngine } from "@/hooks/useEngine";
import { buildModel, countParams, describeArchitecture, disposeModel, validateConfig, type ModelConfigInput } from "@/lib/ml/builder";
import { argmax, oneHot, splitIndices, take } from "@/lib/ml/data";
import { augment } from "@/lib/ml/doodle";
import { loadMnist } from "@/lib/ml/datasets/mnist";
import { activeBackend } from "@/lib/ml/engine";
import { classificationReport } from "@/lib/ml/metrics";
import { makeRng } from "@/lib/ml/rng";
import { saliencyMap } from "@/lib/ml/saliency";
import { saveRun } from "@/lib/ml/storage";
import { predictArray, TrainingRunner, type EpochLog, type RunnerState } from "@/lib/ml/trainer";
import { unlockBadge } from "@/lib/progress";
import { EnginePanel } from "../EnginePanel";
import { LineChartPanel } from "@/components/charts/LineChartPanel";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import { DrawingPad, Thumb, type DrawingPadHandle } from "./DrawingPad";
import { ModelSaveBar } from "../ModelSaveBar";
import { TutorPanel } from "@/components/tutor/TutorPanel";
import { buildRunSummary } from "@/lib/ai/summary";

interface DClass { name: string; emoji: string }
interface Sample { img: Float32Array; label: number; id: number }
interface Result { acc: number; confusion: number[][]; wrong: { sample: Sample; pred: number; conf: number }[]; testN: number }

const DEFAULT_CLASSES: DClass[] = [{ name: "sun", emoji: "☀️" }, { name: "house", emoji: "🏠" }, { name: "fish", emoji: "🐟" }];
const DIGITS: DClass[] = Array.from({ length: 10 }, (_, i) => ({ name: String(i), emoji: `${i}️⃣` }));
let nextId = 1;

function cnnConfig(nClasses: number, epochs: number, lr: number): ModelConfigInput {
  return {
    inputShape: [28, 28, 1],
    layers: [
      { type: "conv2d", filters: 8, kernelSize: 3, activation: "relu", padding: "same" },
      { type: "maxPooling2d", poolSize: 2 },
      { type: "conv2d", filters: 16, kernelSize: 3, activation: "relu", padding: "same" },
      { type: "maxPooling2d", poolSize: 2 },
      { type: "flatten" },
      { type: "dropout", rate: 0.25 },
      { type: "dense", units: 32, activation: "relu" },
      { type: "dense", units: nClasses, activation: "softmax" },
    ],
    optimizer: "adam", learningRate: lr, batchSize: 16, epochs, loss: "categoricalCrossentropy", seed: 7,
  };
}

export default function DoodleLab({ presetId }: { presetId: string | null }) {
  const { engine, stats } = useEngine();
  const [classes, setClasses] = useState<DClass[]>(DEFAULT_CLASSES);
  const [mode, setMode] = useState<"custom" | "digits">("custom");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [active, setActive] = useState(0);
  const [augmentOn, setAugmentOn] = useState(true);
  const [epochs, setEpochs] = useState(15);
  const [lr, setLr] = useState(0.002);
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [state, setState] = useState<RunnerState>("idle");
  const [result, setResult] = useState<Result | null>(null);
  const [live, setLive] = useState<number[] | null>(null);
  const [liveImg, setLiveImg] = useState<Float32Array | null>(null);
  const [saliency, setSaliency] = useState<Float32Array | null>(null);
  const [focus, setFocus] = useState<{ t: number; p: number } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [loadingPack, setLoadingPack] = useState(false);

  const padRef = useRef<DrawingPadHandle>(null);
  const livePadRef = useRef<DrawingPadHandle>(null);
  const modelRef = useRef<tf.Sequential | null>(null);
  const runnerRef = useRef<TrainingRunner | null>(null);
  const [trainedClasses, setTrainedClasses] = useState<DClass[]>([]);

  useEffect(() => () => {
    runnerRef.current?.stop();
    disposeModel(modelRef.current);
  }, []);

  const counts = useMemo(() => classes.map((_, i) => samples.filter((s) => s.label === i).length), [classes, samples]);
  const cfg = useMemo(() => cnnConfig(classes.length, epochs, lr), [classes.length, epochs, lr]);
  const params = useMemo(() => countParams(cfg as never), [cfg]);
  const minPerClass = mode === "digits" ? 5 : 15;
  const ready = counts.every((c) => c >= minPerClass) && classes.length >= 2;
  const running = state === "running";

  const addDrawing = () => {
    const img = padRef.current?.capture();
    if (!img) return setMsg("Draw something first!");
    if (counts[active] >= (mode === "digits" ? 500 : DATA_LIMITS.doodleMaxPerClass)) return setMsg("That class is full — delete some drawings or pick another class.");
    setSamples((s) => [...s, { img, label: active, id: nextId++ }]);
    padRef.current?.clear();
    setMsg(null);
  };

  const loadDigits = useCallback(async (perClass: number) => {
    setLoadingPack(true);
    setMsg(null);
    try {
      const { images, labels } = await loadMnist(perClass);
      setMode("digits");
      setClasses(DIGITS);
      setSamples(images.map((img, i) => ({ img, label: labels[i], id: nextId++ })));
      setActive(0);
      setResult(null);
      setHistory([]);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "Couldn't load the starter pack.");
    } finally {
      setLoadingPack(false);
    }
  }, []);

  useEffect(() => {
    if (presetId !== "digits") return;
    const t = setTimeout(() => void loadDigits(30), 0);
    return () => clearTimeout(t);
  }, [presetId, loadDigits]);

  const backToCustom = () => {
    setMode("custom");
    setClasses(DEFAULT_CLASSES);
    setSamples([]);
    setResult(null);
    setHistory([]);
  };

  const train = async () => {
    setMsg(null);
    setResult(null);
    setHistory([]);
    setLive(null);
    setSaliency(null);
    runnerRef.current?.stop();
    disposeModel(modelRef.current);
    const v = validateConfig(cfg, LIMITS.doodle);
    if (!v.ok) return setMsg(v.errors.join(" "));

    const split = splitIndices(samples.length, { train: 80, val: 0, test: 20 }, 7, samples.map((s) => s.label));
    const trainS = take(samples, split.train), testS = take(samples, split.test);
    const rng = makeRng(7);
    const trainImgs = augmentOn ? trainS.flatMap((s) => [s.img, augment(s.img, rng), augment(s.img, rng)]) : trainS.map((s) => s.img);
    const trainLabels = augmentOn ? trainS.flatMap((s) => [s.label, s.label, s.label]) : trainS.map((s) => s.label);
    const stack = (imgs: Float32Array[]) => {
      const buf = new Float32Array(imgs.length * 784);
      imgs.forEach((im, i) => buf.set(im, i * 784));
      return tf.tensor4d(buf, [imgs.length, 28, 28, 1]);
    };
    const n = classes.length;
    const xtr = stack(trainImgs), ytr = tf.tensor2d(oneHot(trainLabels, n));
    const xte = stack(testS.map((s) => s.img)), yte = tf.tensor2d(oneHot(testS.map((s) => s.label), n));
    let model: tf.Sequential | null = null;
    try {
      model = buildModel(v.config, LIMITS.doodle);
      modelRef.current = model;
      setTrainedClasses(classes);
      const runner = new TrainingRunner(model, xtr, ytr, {
        epochs, batchSize: 16, seed: 7, validation: testS.length ? { x: xte, y: yte } : undefined,
        onStateChange: setState,
        onEpochEnd: (log) => setHistory((h) => [...h, log]),
        yieldEvery: 2,
      });
      runnerRef.current = runner;
      const t0 = performance.now();
      const hist = await runner.run();
      if (runner.error) throw runner.error;
      if (runner.stopReason === "user" && !hist.length) return;
      const probs = await predictArray(model, xte, 64);
      const preds = probs.map(argmax);
      const rep = classificationReport(testS.map((s) => s.label), preds, n);
      const wrong = testS.map((s, i) => ({ sample: s, pred: preds[i], conf: probs[i][preds[i]] })).filter((w) => w.pred !== w.sample.label);
      setResult({ acc: rep.accuracy, confusion: rep.confusion, wrong, testN: testS.length });
      unlockBadge("first-run");
      unlockBadge("doodle-artist");
      saveRun({
        lab: "doodle", name: mode === "digits" ? "Digits 0–9 (MNIST subset)" : classes.map((c) => c.emoji).join(" "),
        durationMs: Math.round(performance.now() - t0), backend: activeBackend(), architecture: describeArchitecture(v.config), params,
        hyperparams: { learningRate: lr, batchSize: 16, epochs: hist.length, augmentation: augmentOn },
        dataset: { name: mode === "digits" ? "mnist-subset" : "my-doodles", nTrain: trainS.length, nTest: testS.length, classNames: classes.map((c) => c.name) },
        history: hist, metrics: { testAccuracy: rep.accuracy, testMacroF1: rep.macro.f1, ...(hist.at(-1)?.valLoss !== undefined ? { testLoss: hist.at(-1)!.valLoss! } : {}) },
      });
    } catch (e) {
      setMsg(/memory|allocate/i.test(String(e)) ? "Ran out of memory — try fewer drawings, turn off augmentation, or close other tabs." : "Training stopped unexpectedly. Try again.");
    } finally {
      tf.dispose([xtr, ytr, xte, yte]);
      runnerRef.current = null;
    }
  };

  const predictLive = useCallback(async () => {
    const m = modelRef.current;
    const img = livePadRef.current?.capture();
    if (!m || !img) return;
    const x = tf.tensor4d(img, [1, 28, 28, 1]);
    const [p] = await predictArray(m, x);
    x.dispose();
    setLive(p);
    setLiveImg(img);
    setSaliency(null);
  }, []);

  const showSaliency = () => {
    const m = modelRef.current;
    if (!m || !liveImg || !live) return;
    setSaliency(saliencyMap(m, liveImg, [28, 28, 1], argmax(live)));
  };

  const shownWrong = result?.wrong.filter((w) => !focus || (w.sample.label === focus.t && w.pred === focus.p)) ?? [];
  const tc = trainedClasses;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">✏️ Doodle Classifier</h1>
          <p className="text-muted">Invent 2–5 classes, draw examples, and train a convolutional neural network from scratch.</p>
        </div>
        <EnginePanel engine={engine} stats={stats} />
      </div>
      <p className="text-xs text-muted sm:hidden">💻 Best on a laptop or tablet — drawing and training work on phones too, just slower.</p>
      {msg && <p role="alert" className="panel p-3 text-sm text-amber">⚠️ {msg}</p>}

      <section className="panel space-y-3 p-4" aria-labelledby="dd-classes">
        <h2 id="dd-classes" className="font-semibold">1 · Classes</h2>
        {mode === "custom" ? (
          <div className="flex flex-wrap items-end gap-2">
            {classes.map((c, i) => (
              <div key={i} className="flex items-end gap-1 rounded-xl border p-2" style={{ borderColor: classColor(i) }}>
                <label className="block w-14"><span className="label">Emoji</span><input className="input text-center" maxLength={4} value={c.emoji} onChange={(e) => { setClasses(classes.map((x, j) => (j === i ? { ...x, emoji: e.target.value } : x))); setResult(null); }} /></label>
                <label className="block w-28"><span className="label">Name</span><input className="input" maxLength={20} value={c.name} onChange={(e) => { setClasses(classes.map((x, j) => (j === i ? { ...x, name: e.target.value } : x))); setResult(null); }} /></label>
                <button className="btn btn-sm" aria-label={`Remove class ${c.name}`} disabled={classes.length <= DATA_LIMITS.doodleMinClasses}
                  onClick={() => { setClasses(classes.filter((_, j) => j !== i)); setSamples((s) => s.filter((x) => x.label !== i).map((x) => ({ ...x, label: x.label > i ? x.label - 1 : x.label }))); setActive(0); setResult(null); }}>✕</button>
              </div>
            ))}
            <button className="btn btn-sm" disabled={classes.length >= DATA_LIMITS.doodleMaxClasses} onClick={() => setClasses([...classes, { name: `class ${classes.length + 1}`, emoji: "⭐" }])}>+ Class</button>
          </div>
        ) : (
          <p className="text-sm">Using the <b>Digits 0–9</b> starter pack ({samples.length} images). <button className="btn btn-sm ml-2" onClick={backToCustom}>← Back to my own classes</button></p>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <span>Starter pack:</span>
          {[30, 100, 300].map((n) => <button key={n} className="btn btn-sm" disabled={loadingPack} onClick={() => loadDigits(n)}>Digits 0–9 · {n}/digit</button>)}
          <span>Small subset of MNIST (CC BY-SA 3.0 — Yann LeCun et al.)</span>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[auto_1fr]" aria-labelledby="dd-draw">
        <div className="panel space-y-3 p-4">
          <h2 id="dd-draw" className="font-semibold">2 · Draw examples</h2>
          <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Class to add drawings to">
            {classes.map((c, i) => (
              <button key={i} role="radio" aria-checked={active === i} onClick={() => setActive(i)} className={`btn btn-sm ${active === i ? "btn-primary" : ""}`}>{c.emoji} {c.name}</button>
            ))}
          </div>
          <DrawingPad ref={padRef} label={`Drawing pad for ${classes[active]?.name}`} />
          <div className="flex gap-2">
            <button className="btn btn-primary" onClick={addDrawing}>➕ Add to {classes[active]?.emoji} {classes[active]?.name}</button>
            <button className="btn" onClick={() => padRef.current?.clear()}>🧽 Clear</button>
          </div>
          <p className="text-xs text-muted">Aim for 15–50 drawings per class. Each drawing is auto-centred and shrunk to 28×28 pixels, like MNIST.</p>
        </div>
        <div className="panel space-y-3 p-4">
          <h3 className="font-semibold">Gallery</h3>
          {classes.map((c, i) => (
            <div key={i}>
              <p className="text-sm">{c.emoji} <b>{c.name}</b> <span className={`mono text-xs ${counts[i] >= minPerClass ? "text-good" : "text-amber"}`}>{counts[i]} {counts[i] < minPerClass && `(need ${minPerClass})`}</span></p>
              <div className="mt-1 flex max-h-28 flex-wrap gap-1 overflow-y-auto">
                {samples.filter((s) => s.label === i).slice(0, 60).map((s) => (
                  <button key={s.id} className="group relative" aria-label={`Delete this ${c.name} drawing`} onClick={() => setSamples((all) => all.filter((x) => x.id !== s.id))}>
                    <Thumb img={s.img} size={40} />
                    <span className="absolute inset-0 hidden items-center justify-center rounded bg-bad/70 text-xs group-hover:flex group-focus:flex">✕</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel space-y-3 p-4" aria-labelledby="dd-train">
        <h2 id="dd-train" className="font-semibold">3 · Train the CNN</h2>
        <p className="mono text-xs text-muted">{describeArchitecture(cfg as never)} · {params.toLocaleString()} parameters</p>
        <div className="flex flex-wrap items-end gap-3">
          <label className="block w-28"><span className="label">Epochs</span><input type="number" className="input mono" min={1} max={LIMITS.doodle.maxEpochs} value={epochs} onChange={(e) => setEpochs(Math.min(LIMITS.doodle.maxEpochs, Math.max(1, Number(e.target.value) || 1)))} /></label>
          <label className="block w-32"><span className="label">Learning rate</span><select className="select" value={lr} onChange={(e) => setLr(Number(e.target.value))}>{[...new Set([0.0005, 0.001, 0.002, 0.005, 0.01, lr])].map((v) => <option key={v} value={v}>{v}</option>)}</select></label>
          <label className="flex items-center gap-2 pb-2 text-sm"><input type="checkbox" checked={augmentOn} onChange={(e) => setAugmentOn(e.target.checked)} /> Data augmentation (small shifts & rotations, ×3 data)</label>
          {running ? (
            <button className="btn" onClick={() => runnerRef.current?.stop()}>⏹ Stop</button>
          ) : (
            <button className="btn btn-primary" onClick={train} disabled={!ready || engine.status !== "ready"}>🚀 Train</button>
          )}
          <span className="mono text-sm" aria-live="polite">epoch {history.length}/{epochs} · {state}</span>
        </div>
        {!ready && <p className="text-xs text-amber">Each class needs at least {minPerClass} drawings before training.</p>}
        <div className="grid gap-4 md:grid-cols-2">
          <LineChartPanel title="Loss" data={history.map((h) => ({ epoch: h.epoch, train: h.loss, test: h.valLoss }))} xKey="epoch" series={[{ key: "train", label: "Training", color: "var(--cyan)" }, { key: "test", label: "Test", color: "var(--amber)" }]} />
          <LineChartPanel title="Accuracy" yDomain={[0, 1]} data={history.map((h) => ({ epoch: h.epoch, train: h.acc, test: h.valAcc }))} xKey="epoch" series={[{ key: "train", label: "Training", color: "var(--cyan)" }, { key: "test", label: "Test", color: "var(--amber)" }]} />
        </div>
      </section>

      {result && (
        <section className="panel space-y-4 p-4" aria-labelledby="dd-results">
          <h2 id="dd-results" className="font-semibold">4 · Results</h2>
          <ModelSaveBar getModel={() => modelRef.current} meta={() => ({
            app: "neuralnest", version: 1, name: mode === "digits" ? "Digits CNN" : `Doodles ${tc.map((c) => c.name).join(" ")}`.slice(0, 80), lab: "doodle", task: "classification",
            createdAt: Date.now(), inputShape: [28, 28, 1], classNames: tc.map((c) => c.name.slice(0, 60)), classEmojis: tc.map((c) => c.emoji.slice(0, 16)),
            featureExtractor: "none", metrics: { testAccuracy: result.acc },
          })} />
          <p className="text-lg" aria-live="polite">Test accuracy: <b className="mono text-cyan">{(result.acc * 100).toFixed(1)}%</b> <span className="text-sm text-muted">on {result.testN} drawings the network never trained on</span></p>
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <p className="label">Confusion matrix (click a red cell to filter mistakes)</p>
              <ConfusionMatrix matrix={result.confusion} labels={tc.map((c) => `${c.emoji} ${c.name}`)} onCellClick={(t, p) => setFocus({ t, p })} />
            </div>
            <div>
              <p className="label">Where it gets confused {focus && <button className="btn btn-sm ml-2" onClick={() => setFocus(null)}>show all</button>}</p>
              {shownWrong.length === 0 ? <p className="text-sm text-good">No mistakes here 🎉</p> : (
                <ul className="flex flex-wrap gap-2">
                  {shownWrong.slice(0, 40).map((w) => (
                    <li key={w.sample.id} className="text-center text-[10px]">
                      <Thumb img={w.sample.img} size={52} label={`A ${tc[w.sample.label].name} predicted as ${tc[w.pred].name}`} />
                      <span>{tc[w.sample.label].emoji}→{tc[w.pred].emoji} {(w.conf * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}

      {result && (
        <section className="panel grid gap-4 p-4 md:grid-cols-[auto_1fr]" aria-labelledby="dd-live">
          <div className="space-y-2">
            <h2 id="dd-live" className="font-semibold">5 · Live prediction</h2>
            <DrawingPad ref={livePadRef} onStroke={predictLive} label="Live prediction pad" />
            <button className="btn" onClick={() => { livePadRef.current?.clear(); setLive(null); setSaliency(null); setLiveImg(null); }}>🧽 Clear</button>
          </div>
          <div className="space-y-3" aria-live="polite">
            {live ? (
              <>
                <ul className="space-y-1">
                  {tc.map((c, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <span className="w-24 truncate">{c.emoji} {c.name}</span>
                      <span className="h-4 rounded transition-all" style={{ width: `${Math.max(1, live[i] * 70)}%`, background: classColor(i) }} />
                      <span className="mono">{(live[i] * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
                <div className="flex items-start gap-4">
                  <div className="text-center text-xs"><Thumb img={liveImg!} size={112} label="What the network sees (28×28)" /><p>what it sees</p></div>
                  {saliency && <div className="text-center text-xs"><Thumb img={liveImg!} overlay={saliency} size={112} label="Saliency map" /><p>what it&apos;s looking at</p></div>}
                </div>
                <button className="btn btn-sm" onClick={showSaliency}>👀 What is the AI looking at?</button>
                <p className="text-xs text-muted">The saliency map highlights pixels where a tiny change would most change the score for “{tc[argmax(live)].name}” (gradient of the score with respect to each pixel).</p>
              </>
            ) : <p className="text-sm text-muted">Draw on the pad — predictions update after every stroke.</p>}
          </div>
        </section>
      )}
      <TutorPanel lab="doodle" getSummary={() => result ? buildRunSummary({
        lab: "doodle", task: "classification", architecture: describeArchitecture(cfg as never), params,
        hyperparams: { learningRate: lr, batchSize: 16, epochs: history.length, augmentation: augmentOn },
        dataset: { name: mode === "digits" ? "MNIST digits subset" : "my doodles", nTrain: samples.length - result.testN, nTest: result.testN, classNames: tc.map((c) => c.name), classCounts: tc.map((_, i) => samples.filter((s) => s.label === i).length) },
        history, metrics: { testAccuracy: result.acc, trainLoss: history.at(-1)?.loss, testLoss: history.at(-1)?.valLoss, trainAccuracy: history.at(-1)?.acc },
        confusion: result.confusion,
      }) : null} onApplyPatch={(p) => { if (p.learningRate !== undefined) setLr(p.learningRate); if (p.epochs !== undefined) setEpochs(p.epochs); }} />
    </div>
  );
}
