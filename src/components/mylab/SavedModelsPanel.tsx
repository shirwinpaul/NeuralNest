"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { classColor } from "@/config/palette";
import { useEngine } from "@/hooks/useEngine";
import { featurize, type FeatureId } from "@/lib/ml/datasets/toy";
import { fileToBitmap, loadFeatureExtractor, toSquareCanvas } from "@/lib/ml/image";
import { deleteModelFromBrowser, listSavedModels, loadModelFromBrowser, saveModelToBrowser, zipToModel, type SavedModel } from "@/lib/ml/modelio";
import { encodeFeatures, FittedPipelineSchema, denormY } from "@/lib/ml/tabular";
import { predictArray } from "@/lib/ml/trainer";
import { PredictForm, type PredictResult } from "@/components/labs/data/PredictForm";
import { DrawingPad, type DrawingPadHandle } from "@/components/labs/doodle/DrawingPad";

const LAB_EMOJI: Record<string, string> = { playground: "🎯", data: "📊", doodle: "✏️", image: "📷" };

export default function SavedModelsPanel() {
  const { engine } = useEngine();
  const [models, setModels] = useState<SavedModel[]>([]);
  const [open, setOpen] = useState<{ rec: SavedModel; model: tf.LayersModel } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = useCallback(() => setModels(listSavedModels()), []);
  useEffect(() => {
    const t = setTimeout(refresh, 0);
    window.addEventListener("nn-models", refresh);
    return () => {
      clearTimeout(t);
      window.removeEventListener("nn-models", refresh);
    };
  }, [refresh]);
  useEffect(() => () => {
    open?.model.dispose();
  }, [open]);

  const load = async (rec: SavedModel) => {
    setMsg(null);
    try {
      // The previous model is disposed by the effect cleanup when `open` changes.
      setOpen({ rec, model: await loadModelFromBrowser(rec.id) });
    } catch {
      setMsg("Couldn't load that model — it may have been cleared from browser storage.");
    }
  };

  const onImport = async (f: File) => {
    setMsg(null);
    try {
      const { model, meta } = await zipToModel(await f.arrayBuffer());
      await saveModelToBrowser(model, meta);
      model.dispose();
      setMsg(`Imported “${meta.name}” ✓`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "That file isn't a NeuralNest model zip.");
    }
  };

  return (
    <section className="panel space-y-3 p-4" aria-labelledby="ml-models">
      <h2 id="ml-models" className="font-semibold">🧠 My Models <span className="text-xs font-normal text-muted">(saved in this browser)</span></h2>
      {msg && <p role="status" className="text-sm">{msg}</p>}
      {models.length === 0 ? <p className="text-sm text-muted">No saved models yet. Train one and press “💾 Save to My Models”.</p> : (
        <ul className="space-y-1 text-sm">
          {models.map((m) => (
            <li key={m.id} className="flex flex-wrap items-center gap-2 border-t border-border pt-1">
              <span>{LAB_EMOJI[m.meta.lab]} {m.meta.name}</span>
              <span className="mono text-xs text-muted">{new Date(m.savedAt).toLocaleDateString()}</span>
              <span className="ml-auto flex gap-1">
                <button className="btn btn-sm" onClick={() => load(m)} disabled={engine.status !== "ready"}>Load &amp; predict</button>
                <button className="btn btn-sm" aria-label={`Delete ${m.meta.name}`} onClick={async () => { if (open?.rec.id === m.id) setOpen(null); await deleteModelFromBrowser(m.id); }}>🗑</button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <label className="btn btn-sm">⬆️ Import model (.zip)<input type="file" accept=".zip,application/zip" className="sr-only" onChange={(e) => { if (e.target.files?.[0]) void onImport(e.target.files[0]); e.target.value = ""; }} /></label>
      {open && <Predictor key={open.rec.id} rec={open.rec} model={open.model} />}
    </section>
  );
}

function Bars({ probs, names }: { probs: number[]; names: string[] }) {
  return (
    <ul className="space-y-1" aria-live="polite">
      {names.map((n, i) => (
        <li key={i} className="flex items-center gap-2 text-sm">
          <span className="w-24 truncate">{n}</span>
          <span className="h-3 rounded" style={{ width: `${Math.max(1, (probs[i] ?? 0) * 60)}%`, background: classColor(i) }} />
          <span className="mono">{((probs[i] ?? 0) * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

function Predictor({ rec, model }: { rec: SavedModel; model: tf.LayersModel }) {
  const meta = rec.meta;
  const names = (meta.classNames ?? []).map((n, i) => `${meta.classEmojis?.[i] ?? ""} ${n}`.trim());
  const [probs, setProbs] = useState<number[] | null>(null);
  const [x, setX] = useState<[number, number]>([0.3, -0.2]);
  const [err, setErr] = useState<string | null>(null);
  const pad = useRef<DrawingPadHandle>(null);

  const run = async (input: tf.Tensor) => {
    try {
      const [p] = await predictArray(model, input);
      setProbs(p);
    } catch {
      setErr("This model couldn't make a prediction with that input.");
    } finally {
      input.dispose();
    }
  };

  return (
    <div className="space-y-2 rounded-xl border border-cyan/40 p-3">
      <p className="text-sm font-semibold">🔮 Predict with “{meta.name}”</p>
      {err && <p className="text-xs text-amber">{err}</p>}
      {meta.lab === "playground" && (
        <div className="space-y-2">
          {(["x₁", "x₂"] as const).map((l, i) => (
            <label key={l} className="block text-sm">{l} = <span className="mono">{x[i].toFixed(2)}</span>
              <input type="range" min={-1.2} max={1.2} step={0.01} value={x[i]} onChange={(e) => { const nx: [number, number] = i ? [x[0], Number(e.target.value)] : [Number(e.target.value), x[1]]; setX(nx); void run(tf.tensor2d([featurize(nx[0], nx[1], (meta.features ?? ["x1", "x2"]) as FeatureId[])])); }} />
            </label>
          ))}
          {probs && (meta.task === "classification" ? <Bars probs={[1 - probs[0], probs[0]]} names={names.length ? names : ["0", "1"]} /> : <p className="mono">ŷ = {probs[0].toFixed(3)}</p>)}
        </div>
      )}
      {meta.lab === "data" && (() => {
        const p = FittedPipelineSchema.safeParse(meta.preprocess);
        if (!p.success) return <p className="text-xs text-amber">This model has no preprocessing info.</p>;
        const pl = p.data;
        const onPredict = async (vals: Record<string, string>): Promise<PredictResult | null> => {
          const v = encodeFeatures((n) => vals[n], pl);
          if (!v) return null;
          const t = tf.tensor2d([v]);
          const [out] = await predictArray(model, t);
          t.dispose();
          return pl.task === "classification" ? { kind: "class", probs: out, classNames: pl.classNames ?? [] } : { kind: "value", value: denormY(out[0], pl) };
        };
        return <PredictForm pipeline={pl} defaults={Object.fromEntries(pl.features.map((f) => [f.name, f.kind === "categorical" ? f.categories?.[0] ?? "" : String(f.fill ?? "")]))} onPredict={onPredict} />;
      })()}
      {meta.lab === "doodle" && (
        <div className="flex flex-wrap gap-4">
          <div className="w-52"><DrawingPad ref={pad} label="Draw to predict" onStroke={() => { const img = pad.current?.capture(); if (img) void run(tf.tensor4d(img, [1, 28, 28, 1])); }} /></div>
          <div className="space-y-2">
            <button className="btn btn-sm" onClick={() => { pad.current?.clear(); setProbs(null); }}>🧽 Clear</button>
            {probs && <Bars probs={probs} names={names} />}
          </div>
        </div>
      )}
      {meta.lab === "image" && (
        <div className="space-y-2">
          <label className="btn btn-sm">🖼️ Choose a photo<input type="file" accept="image/*" className="sr-only" onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            try {
              const ex = await loadFeatureExtractor();
              const bmp = await fileToBitmap(f);
              const emb = ex.embed(toSquareCanvas(bmp));
              bmp.close();
              await run(tf.tensor2d(emb, [1, emb.length]));
            } catch {
              setErr("Couldn't process that photo (MobileNet needs an internet connection the first time).");
            }
          }} /></label>
          {probs && <Bars probs={probs} names={names} />}
        </div>
      )}
    </div>
  );
}
