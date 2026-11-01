"use client";
import { useMemo, useState } from "react";

/** A bumpy 1D loss so learners can see local minima and overshooting. */
const loss = (w: number) => 0.15 * (w - 1) ** 2 + 0.6 * Math.sin(1.6 * w) + 1.2;
const dloss = (w: number) => 0.3 * (w - 1) + 0.96 * Math.cos(1.6 * w);
const W0 = -5, W1 = 6;

/** Interactive 1D gradient descent: w ← w − η · dL/dw */
export function GradientDescentToy() {
  const [start, setStart] = useState(-4);
  const [lr, setLr] = useState(0.5);
  const [path, setPath] = useState<number[]>([-4]);
  const w = path[path.length - 1];
  const W = 320, H = 160;
  const curve = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i <= 120; i++) {
      const x = W0 + ((W1 - W0) * i) / 120;
      pts.push(`${(i / 120) * W},${H - (loss(x) / 5) * H}`);
    }
    return pts.join(" ");
  }, []);
  const px = (x: number) => ((x - W0) / (W1 - W0)) * W;
  const py = (x: number) => H - (Math.min(5, loss(x)) / 5) * H;
  const step = (n: number) => {
    const p = [...path];
    for (let i = 0; i < n; i++) {
      const cur = p[p.length - 1];
      const next = Math.max(W0, Math.min(W1, cur - lr * dloss(cur)));
      p.push(next);
    }
    setPath(p.slice(-60));
  };
  return (
    <div className="panel space-y-3 p-4">
      <p className="font-semibold">⛰️ Gradient descent toy</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md rounded-lg bg-bg-2" role="img" aria-label={`Loss curve. Current weight ${w.toFixed(2)}, loss ${loss(w).toFixed(3)}.`}>
        <polyline points={curve} fill="none" stroke="var(--cyan)" strokeWidth="2" />
        {path.map((p, i) => <circle key={i} cx={px(p)} cy={py(p)} r={2} fill="var(--violet)" opacity={0.5} />)}
        <circle cx={px(w)} cy={py(w)} r={7} fill="var(--amber)" />
      </svg>
      <p className="mono text-sm" aria-live="polite">w = {w.toFixed(3)} · loss = {loss(w).toFixed(3)} · gradient = {dloss(w).toFixed(3)} · steps = {path.length - 1}</p>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block text-sm"><span className="label">Starting weight: {start.toFixed(1)}</span>
          <input type="range" min={W0} max={W1} step={0.1} value={start} onChange={(e) => { const v = Number(e.target.value); setStart(v); setPath([v]); }} /></label>
        <label className="block text-sm"><span className="label">Learning rate η: {lr.toFixed(2)}</span>
          <input type="range" min={0.05} max={4} step={0.05} value={lr} onChange={(e) => setLr(Number(e.target.value))} /></label>
      </div>
      <div className="flex gap-2">
        <button className="btn btn-sm btn-primary" onClick={() => step(1)}>Step</button>
        <button className="btn btn-sm" onClick={() => step(10)}>10 steps</button>
        <button className="btn btn-sm" onClick={() => setPath([start])}>Reset</button>
      </div>
      <p className="text-xs text-muted">Each step: w ← w − η × gradient. Try η = 0.2 (slow but steady), η = 1 (fast) and η = 3.5 (bounces around). Different starting points can end in different valleys (local minima).</p>
    </div>
  );
}

const FNS = {
  relu: (z: number) => Math.max(0, z),
  sigmoid: (z: number) => 1 / (1 + Math.exp(-z)),
  tanh: (z: number) => Math.tanh(z),
} as const;

export function ActivationsToy() {
  const [fn, setFn] = useState<keyof typeof FNS>("relu");
  const [z, setZ] = useState(1);
  const W = 300, H = 150;
  const f = FNS[fn];
  const pts = Array.from({ length: 101 }, (_, i) => {
    const x = -5 + i / 10;
    return `${(i / 100) * W},${H / 2 - Math.max(-1.5, Math.min(5, f(x))) * (H / 7)}`;
  }).join(" ");
  return (
    <div className="panel space-y-3 p-4">
      <p className="font-semibold">🧠 Activation explorer</p>
      <div className="flex gap-2" role="radiogroup" aria-label="Activation function">
        {(Object.keys(FNS) as (keyof typeof FNS)[]).map((k) => <button key={k} role="radio" aria-checked={fn === k} className={`btn btn-sm ${fn === k ? "btn-primary" : ""}`} onClick={() => setFn(k)}>{k}</button>)}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md rounded-lg bg-bg-2" role="img" aria-label={`${fn}(${z}) = ${f(z).toFixed(3)}`}>
        <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="var(--border)" /><line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="var(--border)" />
        <polyline points={pts} fill="none" stroke="var(--cyan)" strokeWidth="2.5" />
        <circle cx={((z + 5) / 10) * W} cy={H / 2 - f(z) * (H / 7)} r={6} fill="var(--amber)" />
      </svg>
      <label className="block text-sm"><span className="label">Input z = {z.toFixed(1)} → output {f(z).toFixed(3)}</span>
        <input type="range" min={-5} max={5} step={0.1} value={z} onChange={(e) => setZ(Number(e.target.value))} /></label>
    </div>
  );
}
