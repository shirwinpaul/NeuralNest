"use client";
import { useState } from "react";
import { classColor } from "@/config/palette";
import type { FittedPipeline } from "@/lib/ml/tabular";

export type PredictResult = { kind: "class"; probs: number[]; classNames: string[] } | { kind: "value"; value: number };

/** Enter new feature values → prediction with confidence. Values stay in the browser. */
export function PredictForm({ pipeline, defaults, onPredict }: {
  pipeline: FittedPipeline;
  defaults: Record<string, string>;
  onPredict: (values: Record<string, string>) => Promise<PredictResult | null>;
}) {
  const [values, setValues] = useState<Record<string, string>>(defaults);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    const r = await onPredict(values);
    if (!r) setErr("Please fill in every value (this model was trained with ‘drop rows with missing values’).");
    setResult(r);
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {pipeline.features.map((f) => (
          <label key={f.name} className="block">
            <span className="label">{f.name}</span>
            {f.kind === "categorical" ? (
              <select className="select" value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}>
                {f.categories!.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input className="input mono" inputMode="decimal" value={values[f.name] ?? ""} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
            )}
          </label>
        ))}
      </div>
      <button type="submit" className="btn btn-primary">🔮 Predict</button>
      {err && <p className="text-sm text-amber" role="alert">{err}</p>}
      <div aria-live="polite">
        {result?.kind === "value" && <p className="text-lg">Predicted <b>{pipeline.target}</b>: <span className="mono text-cyan">{result.value.toFixed(2)}</span></p>}
        {result?.kind === "class" && (
          <ul className="space-y-1">
            {result.classNames.map((c, i) => ({ c, i, p: result.probs[i] })).sort((a, b) => b.p - a.p).map(({ c, i, p }) => (
              <li key={c} className="flex items-center gap-2 text-sm">
                <span className="w-28 truncate">{c}</span>
                <span className="h-3 rounded" style={{ width: `${Math.max(2, p * 60)}%`, background: classColor(i) }} />
                <span className="mono">{(p * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </form>
  );
}
