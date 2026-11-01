"use client";
/* eslint-disable @next/next/no-img-element -- thumbnails are local data: URLs */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { DATA_LIMITS, LIMITS } from "@/config/limits";
import { MOBILENET } from "@/config/models";
import { classColor } from "@/config/palette";
import { useEngine } from "@/hooks/useEngine";
import { buildModel, countParams, describeArchitecture, disposeModel, type ModelConfigInput } from "@/lib/ml/builder";
import { argmax, oneHot, splitIndices, take } from "@/lib/ml/data";
import { activeBackend } from "@/lib/ml/engine";
import { fileToBitmap, loadFeatureExtractor, SMALL, smallPixels, stackFlat, thumbnail, toSquareCanvas, urlToImage } from "@/lib/ml/image";
import { accuracy } from "@/lib/ml/metrics";
import { saveRun } from "@/lib/ml/storage";
import { predictArray, TrainingRunner, type EpochLog } from "@/lib/ml/trainer";
import { unlockBadge } from "@/lib/progress";
import { EnginePanel } from "../EnginePanel";
import { LineChartPanel } from "@/components/charts/LineChartPanel";
import { ModelSaveBar } from "../ModelSaveBar";
import { TutorPanel } from "@/components/tutor/TutorPanel";
import { buildRunSummary } from "@/lib/ai/summary";

interface IClass { name: string; emoji: string }
interface Example { id: number; label: number; emb: Float32Array; px: Float32Array; thumb: string; set: "train" | "new" }
interface Trained { acc: number; history: EpochLog[]; ms: number; params: number; arch: string }

let nextId = 1;
const HEAD_EPOCHS = 30;

const headConfig = (dim: number, n: number): ModelConfigInput => ({
  inputShape: [dim],
  layers: [{ type: "dense", units: 64, activation: "relu" }, { type: "dropout", rate: 0.2 }, { type: "dense", units: n, activation: "softmax" }],
  optimizer: "adam", learningRate: 0.001, batchSize: 16, epochs: HEAD_EPOCHS, loss: "categoricalCrossentropy", seed: 11,
});
const scratchConfig = (n: number): ModelConfigInput => ({
  inputShape: [SMALL, SMALL, 3],
  layers: [
    { type: "conv2d", filters: 8, kernelSize: 3, activation: "relu", padding: "same" }, { type: "maxPooling2d", poolSize: 2 },
    { type: "conv2d", filters: 16, kernelSize: 3, activation: "relu", padding: "same" }, { type: "maxPooling2d", poolSize: 2 },
    { type: "flatten" }, { type: "dense", units: 32, activation: "relu" }, { type: "dense", units: n, activation: "softmax" },
  ],
  optimizer: "adam", learningRate: 0.002, batchSize: 16, epochs: HEAD_EPOCHS, loss: "categoricalCrossentropy", seed: 11,
});

export default function ImageTeacher({ presetId }: { presetId: string | null }) {
  const { engine, stats } = useEngine();
  const [classes, setClasses] = useState<IClass[]>([{ name: "class 1", emoji: "🙂" }, { name: "class 2", emoji: "✋" }]);
  const [examples, setExamples] = useState<Example[]>([]);
  const [active, setActive] = useState(0);
  const [mnState, setMnState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [cam, setCam] = useState<"off" | "starting" | "on" | "denied" | "unavailable">("off");
  const [capturing, setCapturing] = useState(false);
  const [collectInto, setCollectInto] = useState<"train" | "new">("train");
  const [history, setHistory] = useState<EpochLog[]>([]);
  const [transfer, setTransfer] = useState<Trained | null>(null);
  const [scratch, setScratch] = useState<Trained | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [live, setLive] = useState<number[] | null>(null);
  const [photoPred, setPhotoPred] = useState<{ thumb: string; probs: number[] } | null>(null);
  const [biasResult, setBiasResult] = useState<{ a: number; b: number } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [embDim, setEmbDim] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const headRef = useRef<tf.Sequential | null>(null);
  const extractorRef = useRef<Awaited<ReturnType<typeof loadFeatureExtractor>> | null>(null);
  const captureTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [trainedClasses, setTrainedClasses] = useState<IClass[]>([]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCam((c) => (c === "on" || c === "starting" ? "off" : c));
  }, []);
  useEffect(() => () => {
    stopCamera();
    if (captureTimer.current) clearInterval(captureTimer.current);
    disposeModel(headRef.current);
  }, [stopCamera]);

  const ensureExtractor = useCallback(async () => {
    if (extractorRef.current) return extractorRef.current;
    setMnState("loading");
    try {
      extractorRef.current = await loadFeatureExtractor();
      setEmbDim(extractorRef.current.dim);
      setMnState("ready");
      return extractorRef.current;
    } catch {
      setMnState("error");
      setMsg("Couldn't download MobileNet. Check your internet connection and try again.");
      return null;
    }
  }, []);

  const addFromCanvas = useCallback((c: HTMLCanvasElement, label: number, set: "train" | "new") => {
    const ex = extractorRef.current;
    if (!ex) return;
    const e: Example = { id: nextId++, label, emb: ex.embed(c), px: smallPixels(c), thumb: thumbnail(c), set };
    setExamples((all) => [...all, e]);
  }, []);

  const countsFor = (set: "train" | "new") => classes.map((_, i) => examples.filter((e) => e.label === i && e.set === set).length);
  const counts = countsFor("train");
  const newCounts = countsFor("new");

  // ---------- camera ----------
  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return setCam("unavailable");
    setCam("starting");
    if (!(await ensureExtractor())) return setCam("off");
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: "user" }, audio: false });
      streamRef.current = s;
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
      }
      setCam("on");
    } catch (e) {
      setCam((e as Error).name === "NotAllowedError" ? "denied" : "unavailable");
    }
  };

  const startBurst = () => {
    if (cam !== "on" || !videoRef.current) return;
    setCapturing(true);
    const max = DATA_LIMITS.imageMaxPerClass;
    const grab = () => {
      const v = videoRef.current;
      if (!v || v.readyState < 2) return;
      const c = toSquareCanvas(v, MOBILENET.inputSize, true);
      const ex = extractorRef.current!;
      const e: Example = { id: nextId++, label: active, emb: ex.embed(c), px: smallPixels(c), thumb: thumbnail(c), set: collectInto };
      setExamples((all) => (all.filter((x) => x.label === active && x.set === collectInto).length >= max ? all : [...all, e]));
    };
    grab();
    captureTimer.current = setInterval(grab, 150);
  };
  const stopBurst = () => {
    if (captureTimer.current) clearInterval(captureTimer.current);
    captureTimer.current = null;
    setCapturing(false);
  };

  // ---------- uploads & samples ----------
  const onUpload = async (files: FileList, label: number, set: "train" | "new") => {
    if (!(await ensureExtractor())) return;
    setBusy("Reading photos…");
    let bad = 0;
    for (const f of Array.from(files).slice(0, DATA_LIMITS.imageMaxPerClass)) {
      try {
        const bmp = await fileToBitmap(f);
        addFromCanvas(toSquareCanvas(bmp), label, set);
        bmp.close();
        await tf.nextFrame();
      } catch {
        bad++;
      }
    }
    setBusy(null);
    if (bad) setMsg(`${bad} file(s) couldn't be read as images and were skipped.`);
  };

  const loadSamples = useCallback(async (setting: "setting-a" | "setting-b") => {
    if (!(await ensureExtractor())) return;
    setBusy("Loading sample shapes…");
    try {
      const idx = (await (await fetch("/samples/index.json")).json()) as Record<string, Record<string, string[]>>;
      if (setting === "setting-a") {
        setClasses([{ name: "circle", emoji: "⚪" }, { name: "square", emoji: "⬛" }]);
        setExamples([]);
        setTransfer(null);
        setScratch(null);
        setBiasResult(null);
      }
      const set = setting === "setting-a" ? "train" : "new";
      for (const [label, shape] of [[0, "circle"], [1, "square"]] as const) {
        for (const url of idx[setting][shape]) {
          addFromCanvas(toSquareCanvas(await urlToImage(url)), label, set);
          await tf.nextFrame();
        }
      }
    } catch {
      setMsg("Couldn't load the sample images.");
    } finally {
      setBusy(null);
    }
  }, [ensureExtractor, addFromCanvas]);

  useEffect(() => {
    if (presetId !== "bias") return;
    const t = setTimeout(() => void loadSamples("setting-a"), 0);
    return () => clearTimeout(t);
  }, [presetId, loadSamples]);

  // ---------- training ----------
  const trainSet = useMemo(() => examples.filter((e) => e.set === "train"), [examples]);
  const ready = counts.every((c) => c >= 5) && classes.length >= 2;

  const fit = async (cfg: ModelConfigInput, xs: { tr: tf.Tensor; te: tf.Tensor }, ys: { tr: tf.Tensor; te: tf.Tensor }, teY: number[], onLog?: (l: EpochLog) => void) => {
    const model = buildModel(cfg, LIMITS.image);
    const t0 = performance.now();
    const r = new TrainingRunner(model, xs.tr, ys.tr, { epochs: HEAD_EPOCHS, batchSize: 16, seed: 11, validation: { x: xs.te, y: ys.te }, onEpochEnd: onLog, yieldEvery: 2 });
    const history = await r.run();
    if (r.error) throw r.error;
    const preds = (await predictArray(model, xs.te)).map(argmax);
    return { model, res: { acc: accuracy(teY, preds), history, ms: performance.now() - t0, params: model.countParams(), arch: describeArchitecture(cfg as never) } as Trained };
  };

  const splitTrain = () => {
    const s = splitIndices(trainSet.length, { train: 80, val: 0, test: 20 }, 11, trainSet.map((e) => e.label));
    return { tr: take(trainSet, s.train), te: take(trainSet, s.test) };
  };

  const trainTransfer = async () => {
    const ex = await ensureExtractor();
    if (!ex) return;
    setMsg(null);
    setBusy("Training the transfer-learning head…");
    setHistory([]);
    setBiasResult(null);
    disposeModel(headRef.current);
    headRef.current = null;
    const { tr, te } = splitTrain();
    const n = classes.length;
    const xs = { tr: stackFlat(tr.map((e) => e.emb), [ex.dim]), te: stackFlat(te.map((e) => e.emb), [ex.dim]) };
    const ys = { tr: tf.tensor2d(oneHot(tr.map((e) => e.label), n)), te: tf.tensor2d(oneHot(te.map((e) => e.label), n)) };
    try {
      const { model, res } = await fit(headConfig(ex.dim, n), xs, ys, te.map((e) => e.label), (l) => setHistory((h) => [...h, l]));
      headRef.current = model as tf.Sequential;
      setTrainedClasses(classes);
      setTransfer(res);
      unlockBadge("first-run");
      unlockBadge("transfer-master");
      saveRun({
        lab: "image", name: `Transfer: ${classes.map((c) => c.emoji).join(" ")}`, durationMs: Math.round(res.ms), backend: activeBackend(),
        architecture: `MobileNet v2 (frozen) → ${res.arch}`, params: res.params, hyperparams: { learningRate: 0.001, batchSize: 16, epochs: HEAD_EPOCHS, method: "transfer" },
        dataset: { name: "my-images", nTrain: tr.length, nTest: te.length, classNames: classes.map((c) => c.name) },
        history: res.history, metrics: { testAccuracy: res.acc, ...(res.history.at(-1)?.valLoss !== undefined ? { testLoss: res.history.at(-1)!.valLoss! } : {}) },
      });
    } catch (e) {
      setMsg(/memory|allocate/i.test(String(e)) ? "Ran out of memory — try fewer photos or close other tabs." : "Training failed. Try again.");
    } finally {
      tf.dispose([xs.tr, xs.te, ys.tr, ys.te]);
      setBusy(null);
    }
  };

  const trainScratch = async () => {
    setBusy("Training a tiny CNN from scratch (32×32 pixels)…");
    const { tr, te } = splitTrain();
    const n = classes.length;
    const xs = { tr: stackFlat(tr.map((e) => e.px), [SMALL, SMALL, 3]), te: stackFlat(te.map((e) => e.px), [SMALL, SMALL, 3]) };
    const ys = { tr: tf.tensor2d(oneHot(tr.map((e) => e.label), n)), te: tf.tensor2d(oneHot(te.map((e) => e.label), n)) };
    try {
      const { model, res } = await fit(scratchConfig(n), xs, ys, te.map((e) => e.label));
      disposeModel(model);
      setScratch(res);
      saveRun({
        lab: "image", name: `From scratch: ${classes.map((c) => c.emoji).join(" ")}`, durationMs: Math.round(res.ms), backend: activeBackend(),
        architecture: res.arch, params: res.params, hyperparams: { learningRate: 0.002, batchSize: 16, epochs: HEAD_EPOCHS, method: "scratch" },
        dataset: { name: "my-images", nTrain: tr.length, nTest: te.length, classNames: classes.map((c) => c.name) },
        history: res.history, metrics: { testAccuracy: res.acc },
      });
    } catch {
      setMsg("The from-scratch model couldn't train (maybe low memory).");
    } finally {
      tf.dispose([xs.tr, xs.te, ys.tr, ys.te]);
      setBusy(null);
    }
  };

  // ---------- prediction ----------
  const predictEmb = useCallback(async (emb: Float32Array) => {
    const m = headRef.current;
    if (!m) return null;
    const x = tf.tensor2d(emb, [1, emb.length]);
    const [p] = await predictArray(m, x);
    x.dispose();
    return p;
  }, []);

  useEffect(() => {
    if (cam !== "on" || !transfer || capturing) return;
    let alive = true;
    const id = setInterval(async () => {
      const v = videoRef.current, ex = extractorRef.current;
      if (!alive || !v || !ex || v.readyState < 2) return;
      const p = await predictEmb(ex.embed(toSquareCanvas(v, MOBILENET.inputSize, true)));
      if (alive && p) setLive(p);
    }, 250);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [cam, transfer, capturing, predictEmb]);

  const testPhoto = async (f: File) => {
    const ex = await ensureExtractor();
    if (!ex) return;
    try {
      const bmp = await fileToBitmap(f);
      const c = toSquareCanvas(bmp);
      bmp.close();
      const p = await predictEmb(ex.embed(c));
      if (p) setPhotoPred({ thumb: thumbnail(c, 96), probs: p });
    } catch {
      setMsg("That file couldn't be read as an image.");
    }
  };

  const runBiasTest = async () => {
    const newSet = examples.filter((e) => e.set === "new");
    if (!headRef.current || !transfer || newSet.length === 0) return;
    const preds: number[] = [];
    for (const e of newSet) {
      const p = await predictEmb(e.emb);
      preds.push(p ? argmax(p) : -1);
    }
    setBiasResult({ a: transfer.acc, b: accuracy(newSet.map((e) => e.label), preds) });
    unlockBadge("bias-buster");
  };

  const headParams = embDim ? countParams(headConfig(embDim, classes.length) as never) : null;
  const tc = trainedClasses;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">📷 Image Teacher</h1>
          <p className="text-muted">Teach the computer to recognise things with your webcam or photos, using transfer learning.</p>
        </div>
        <EnginePanel engine={engine} stats={stats} />
      </div>
      <p className="text-xs text-muted sm:hidden">💻 Best on a laptop — it works on phones too, just slower.</p>
      {msg && <p role="alert" className="panel p-3 text-sm text-amber">⚠️ {msg} <button className="btn btn-sm ml-2" onClick={() => setMsg(null)}>OK</button></p>}
      {busy && <p role="status" className="panel pulse p-3 text-sm">⏳ {busy}</p>}

      <section className="panel space-y-2 p-4 text-sm" aria-labelledby="it-about">
        <h2 id="it-about" className="font-semibold">How this works</h2>
        <p>We use <b>{MOBILENET.name}</b> ({MOBILENET.licence}), a network already trained on over a million photos, as a frozen <em>feature extractor</em>. It turns each image into {embDim ?? 1280} numbers, and we train only a small new “head” on top. The model downloads once (about {MOBILENET.approxDownloadMB} MB) when you first add images.</p>
        <p className="text-muted">🔒 Webcam frames and photos are processed in this browser and kept in memory only — never uploaded, never sent to the AI tutor.</p>
        {mnState === "loading" && <p className="pulse text-cyan">Downloading MobileNet (~{MOBILENET.approxDownloadMB} MB)…</p>}
        {mnState === "idle" && <button className="btn btn-sm" onClick={ensureExtractor} disabled={engine.status !== "ready"}>⬇️ Load MobileNet now</button>}
      </section>

      <section className="panel space-y-3 p-4" aria-labelledby="it-classes">
        <h2 id="it-classes" className="font-semibold">1 · Classes & examples</h2>
        <div className="flex flex-wrap items-end gap-2">
          {classes.map((c, i) => (
            <div key={i} className="flex items-end gap-1 rounded-xl border p-2" style={{ borderColor: classColor(i) }}>
              <label className="block w-14"><span className="label">Emoji</span><input className="input text-center" maxLength={4} value={c.emoji} onChange={(e) => setClasses(classes.map((x, j) => (j === i ? { ...x, emoji: e.target.value } : x)))} /></label>
              <label className="block w-28"><span className="label">Name</span><input className="input" maxLength={20} value={c.name} onChange={(e) => setClasses(classes.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} /></label>
              <button className="btn btn-sm" aria-label={`Remove ${c.name}`} disabled={classes.length <= 2} onClick={() => { setClasses(classes.filter((_, j) => j !== i)); setExamples((ex) => ex.filter((e) => e.label !== i).map((e) => ({ ...e, label: e.label > i ? e.label - 1 : e.label }))); setActive(0); }}>✕</button>
            </div>
          ))}
          <button className="btn btn-sm" disabled={classes.length >= DATA_LIMITS.imageMaxClasses} onClick={() => setClasses([...classes, { name: `class ${classes.length + 1}`, emoji: "⭐" }])}>+ Class</button>
          <button className="btn btn-sm" onClick={() => loadSamples("setting-a")} disabled={!!busy || engine.status !== "ready"}>🧪 Load sample shapes (no camera needed)</button>
        </div>

        <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
          <div className="space-y-2">
            <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-border bg-bg-2">
              <video ref={videoRef} muted playsInline className={`h-full w-full -scale-x-100 object-cover ${cam === "on" ? "" : "hidden"}`} aria-label="Webcam preview" />
              {cam !== "on" && (
                <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center text-sm">
                  {cam === "denied" && <p className="text-amber">Camera permission was blocked. You can allow it in the browser&apos;s address bar — or just upload photos instead.</p>}
                  {cam === "unavailable" && <p className="text-amber">No camera found. Upload photos instead — everything else works the same.</p>}
                  {(cam === "off" || cam === "starting") && <p className="text-muted">The camera is only used while this lab is open. Your browser will ask for permission. Frames never leave your device.</p>}
                  <button className="btn btn-primary btn-sm" onClick={startCamera} disabled={cam === "starting" || engine.status !== "ready"}>{cam === "starting" ? "Starting…" : "📷 Turn on camera"}</button>
                </div>
              )}
            </div>
            {cam === "on" && (
              <div className="flex flex-wrap gap-2">
                <button className="btn btn-primary select-none" onPointerDown={startBurst} onPointerUp={stopBurst} onPointerLeave={stopBurst}
                  onKeyDown={(e) => { if ((e.key === " " || e.key === "Enter") && !capturing) { e.preventDefault(); startBurst(); } }} onKeyUp={stopBurst}>
                  {capturing ? "📸 Capturing…" : `Hold to capture → ${classes[active]?.emoji} ${classes[active]?.name}`}
                </button>
                <button className="btn btn-sm" onClick={stopCamera}>Turn off camera</button>
              </div>
            )}
            <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Class to add images to">
              {classes.map((c, i) => <button key={i} role="radio" aria-checked={active === i} className={`btn btn-sm ${active === i ? "btn-primary" : ""}`} onClick={() => setActive(i)}>{c.emoji} {c.name}</button>)}
            </div>
            <label className="btn btn-sm w-full">
              📁 Upload photos to {classes[active]?.emoji} {classes[active]?.name}
              <input type="file" accept="image/*" multiple className="sr-only" onChange={(e) => { if (e.target.files) void onUpload(e.target.files, active, collectInto); e.target.value = ""; }} />
            </label>
            <p className="text-xs text-muted">Collecting into: <b>{collectInto === "train" ? "training set" : "new-setting test set (bias activity)"}</b></p>
          </div>
          <div className="space-y-2">
            {classes.map((c, i) => (
              <div key={i}>
                <p className="text-sm">{c.emoji} <b>{c.name}</b> <span className="mono text-xs" aria-live="polite">{counts[i]} training{newCounts[i] ? ` · ${newCounts[i]} new-setting` : ""}</span></p>
                <div className="mt-1 flex max-h-24 flex-wrap gap-1 overflow-y-auto">
                  {examples.filter((e) => e.label === i).slice(-48).map((e) => (
                    <button key={e.id} className={`relative rounded ${e.set === "new" ? "ring-2 ring-amber" : ""}`} aria-label={`Delete this ${c.name} image`} onClick={() => setExamples((all) => all.filter((x) => x.id !== e.id))}>
                      <img src={e.thumb} alt="" width={40} height={40} className="rounded" />
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <p className="text-xs text-muted">Tip: 15–40 varied examples per class work well. Click a thumbnail to delete it. Amber rings = new-setting test images.</p>
          </div>
        </div>
      </section>

      <section className="panel space-y-3 p-4" aria-labelledby="it-train">
        <h2 id="it-train" className="font-semibold">2 · Train</h2>
        <p className="mono text-xs text-muted">Head: MobileNet embedding → 64 relu → dropout 0.2 → {classes.length} softmax {headParams ? `· ${headParams.toLocaleString()} trainable parameters` : ""}</p>
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-primary" onClick={trainTransfer} disabled={!ready || !!busy}>🎁 Train with transfer learning</button>
          <button className="btn" onClick={trainScratch} disabled={!ready || !!busy}>🧱 Compare: tiny CNN from scratch</button>
        </div>
        {!ready && <p className="text-xs text-amber">Each class needs at least 5 training images.</p>}
        <LineChartPanel title="Transfer-learning head: loss" data={history.map((h) => ({ epoch: h.epoch, train: h.loss, test: h.valLoss }))} xKey="epoch"
          series={[{ key: "train", label: "Training", color: "var(--cyan)" }, { key: "test", label: "Test", color: "var(--amber)" }]} />
        {(transfer || scratch) && (
          <table className="mono w-full text-sm">
            <thead><tr className="text-left text-xs text-muted"><th>Method</th><th className="text-right">Test accuracy</th><th className="text-right">Trainable params</th><th className="text-right">Time</th></tr></thead>
            <tbody>
              {transfer && <tr><td>🎁 Transfer learning</td><td className="text-right text-cyan">{(transfer.acc * 100).toFixed(1)}%</td><td className="text-right">{transfer.params.toLocaleString()}</td><td className="text-right">{(transfer.ms / 1000).toFixed(1)}s</td></tr>}
              {scratch && <tr><td>🧱 From scratch</td><td className="text-right">{(scratch.acc * 100).toFixed(1)}%</td><td className="text-right">{scratch.params.toLocaleString()}</td><td className="text-right">{(scratch.ms / 1000).toFixed(1)}s</td></tr>}
            </tbody>
          </table>
        )}
        {transfer && embDim && (
          <ModelSaveBar getModel={() => headRef.current} meta={() => ({
            app: "neuralnest", version: 1, name: `Image Teacher ${tc.map((c) => c.name).join(" ")}`.slice(0, 80), lab: "image", task: "classification",
            createdAt: Date.now(), inputShape: [embDim], classNames: tc.map((c) => c.name.slice(0, 60)), classEmojis: tc.map((c) => c.emoji.slice(0, 16)),
            featureExtractor: "mobilenet_v2_0.5", metrics: { testAccuracy: transfer.acc },
          })} />
        )}
        {transfer && scratch && <p className="text-sm text-muted">With few examples, the transfer model usually wins: MobileNet already knows edges, textures and shapes, so only a small head must be learned. <Link href="/academy/transfer-learning" className="text-cyan underline">Why?</Link></p>}
      </section>

      {transfer && (
        <section className="panel grid gap-4 p-4 md:grid-cols-2" aria-labelledby="it-predict">
          <div className="space-y-2">
            <h2 id="it-predict" className="font-semibold">3 · Test it</h2>
            {cam === "on" ? <p className="text-sm text-muted">Point the camera at something — live predictions below.</p> : <p className="text-sm text-muted">Turn on the camera for live predictions, or test with a new photo.</p>}
            <label className="btn btn-sm">🖼️ Test with a new photo<input type="file" accept="image/*" className="sr-only" onChange={(e) => e.target.files?.[0] && testPhoto(e.target.files[0])} /></label>
            {photoPred && <img src={photoPred.thumb} alt="Your test photo" width={96} height={96} className="rounded" />}
          </div>
          <div aria-live="polite">
            {(live || photoPred) && (
              <ul className="space-y-1">
                {tc.map((c, i) => {
                  const p = (cam === "on" && live ? live : photoPred?.probs)?.[i] ?? 0;
                  return (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <span className="w-24 truncate">{c.emoji} {c.name}</span>
                      <span className="h-4 rounded transition-all" style={{ width: `${Math.max(1, p * 70)}%`, background: classColor(i) }} />
                      <span className="mono">{(p * 100).toFixed(1)}%</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </section>
      )}

      <section className="panel space-y-3 p-4" aria-labelledby="it-bias">
        <h2 id="it-bias" className="font-semibold">⚖️ Bias activity: does it still work somewhere new?</h2>
        <ol className="list-decimal space-y-1 pl-5 text-sm">
          <li>Train using examples from <b>one setting only</b> — one background, one lighting, one person. (Or click “Load sample shapes”: circles are always on yellow, squares always on blue.)</li>
          <li>Collect a <b>new-setting test set</b> in a different place or lighting:
            <span className="ml-2 inline-flex gap-1">
              <button className={`btn btn-sm ${collectInto === "new" ? "btn-primary" : ""}`} onClick={() => setCollectInto(collectInto === "new" ? "train" : "new")} aria-pressed={collectInto === "new"}>{collectInto === "new" ? "✓ Collecting new-setting images" : "Collect new-setting images"}</button>
              <button className="btn btn-sm" onClick={() => loadSamples("setting-b")} disabled={!!busy || classes[0]?.name !== "circle"}>Load sample setting B (backgrounds swapped)</button>
            </span>
          </li>
          <li>Compare accuracy: <button className="btn btn-sm" onClick={runBiasTest} disabled={!transfer || !newCounts.some((c) => c > 0)}>Run the new-setting test</button></li>
        </ol>
        {biasResult && (
          <div className="rounded-xl border border-amber/60 p-3" role="status">
            <p className="text-sm">Original setting test accuracy: <b className="mono">{(biasResult.a * 100).toFixed(1)}%</b> → new setting: <b className="mono text-amber">{(biasResult.b * 100).toFixed(1)}%</b> ({biasResult.b < biasResult.a ? `${((biasResult.a - biasResult.b) * 100).toFixed(1)} points lower` : "no drop this time"})</p>
            <p className="mt-1 text-sm text-muted">If accuracy dropped, the model probably learned the <em>setting</em> (background, lighting, who is in the picture) instead of the thing itself. Real AI systems fail like this when their training data doesn&apos;t represent everyone and everywhere they&apos;re used. <Link href="/academy/bias-fairness" className="text-cyan underline">Read the Academy lesson →</Link></p>
          </div>
        )}
      </section>
      <TutorPanel lab="image" getSummary={() => transfer ? buildRunSummary({
        lab: "image", task: "classification", architecture: `MobileNet v2 0.5 (frozen) → ${transfer.arch}`, params: transfer.params,
        hyperparams: { learningRate: 0.001, batchSize: 16, epochs: HEAD_EPOCHS, method: "transfer learning" },
        dataset: { name: "my images", nTrain: trainSet.length, nTest: Math.round(trainSet.length * 0.2), classNames: tc.map((c) => c.name), classCounts: counts.slice(0, 10) },
        history: transfer.history,
        metrics: { transferTestAccuracy: transfer.acc, scratchTestAccuracy: scratch?.acc, newSettingAccuracy: biasResult?.b },
        notes: biasResult ? ["Bias activity: the model was also tested on images from a different setting."] : undefined,
      }) : null} />
    </div>
  );
}
