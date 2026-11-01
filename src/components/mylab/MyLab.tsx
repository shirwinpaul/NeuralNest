"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ALL_BADGES } from "@/data/challenges";
import { LESSONS } from "@/data/lessons";
import { CLASS_COLORS } from "@/config/palette";
import {
  clearRuns, compareRuns, deleteRun, exportBackup, importBackup, listRuns, localStorageUsage, MAX_COMPARE, toggleStar, type RunRecord,
} from "@/lib/ml/storage";
import { loadProgress, ProgressSchema, saveProgress, type Progress } from "@/lib/progress";
import { downloadText } from "@/lib/download";
import { SavedModels } from "./SavedModels";

const LAB_EMOJI: Record<string, string> = { playground: "🎯", data: "📊", doodle: "✏️", image: "📷" };

export function MyLab() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [usage, setUsage] = useState(0);
  const [quota, setQuota] = useState<{ usage: number; quota: number } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setRuns(listRuns());
    setProgress(loadProgress());
    setUsage(localStorageUsage());
    navigator.storage?.estimate?.().then((e) => setQuota({ usage: e.usage ?? 0, quota: e.quota ?? 0 })).catch(() => {});
  }, []);

  useEffect(() => {
    // Subscribe to local storage changes made anywhere in the app.
    const t = setTimeout(refresh, 0);
    window.addEventListener("nn-runs", refresh);
    window.addEventListener("nn-progress", refresh);
    return () => {
      clearTimeout(t);
      window.removeEventListener("nn-runs", refresh);
      window.removeEventListener("nn-progress", refresh);
    };
  }, [refresh]);

  const comparison = useMemo(() => (selected.length ? compareRuns(selected) : null), [selected, runs]); // eslint-disable-line react-hooks/exhaustive-deps
  const toggleSel = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= MAX_COMPARE ? s : [...s, id]));

  const onExport = () => downloadText(`neuralnest-backup-${new Date().toISOString().slice(0, 10)}.json`, exportBackup(loadProgress()), "application/json");
  const onImport = async (file: File) => {
    try {
      if (file.size > 2_000_000) throw new Error("too big");
      const res = importBackup(await file.text());
      const p = ProgressSchema.safeParse(res.progress);
      if (p.success) saveProgress(p.data);
      setMsg(`Imported ${res.runs} runs.`);
      refresh();
    } catch {
      setMsg("That file isn't a NeuralNest backup.");
    }
  };

  const starred = runs.filter((r) => r.starred);
  const curveData = useMemo(() => {
    if (!comparison) return [];
    const maxLen = Math.max(...comparison.runs.map((r) => r.history.length));
    return Array.from({ length: maxLen }, (_, i) => {
      const row: Record<string, number | undefined> = { point: i + 1 };
      comparison.runs.forEach((r, k) => (row[`r${k}`] = r.history[i]?.valLoss ?? r.history[i]?.loss));
      return row;
    });
  }, [comparison]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">🧪 My Lab</h1>
        <p className="text-muted">Your runs, best models, badges and progress — all stored only on this device.</p>
      </div>
      {msg && <p role="status" className="panel p-3 text-sm">{msg}</p>}

      <section className="panel p-4" aria-labelledby="ml-runs">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h2 id="ml-runs" className="font-semibold">Recent runs ({runs.length})</h2>
          <span className="text-xs text-muted">Tick up to {MAX_COMPARE} runs to compare.</span>
        </div>
        {runs.length === 0 ? (
          <p className="text-sm text-muted">No runs yet. Train something in the <Link className="text-cyan underline" href="/playground">Playground</Link> and press 💾 Save run.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr><th className="p-1">Compare</th><th className="p-1">Run</th><th className="p-1">Architecture</th><th className="p-1">Key metric</th><th className="p-1">When</th><th className="p-1">Actions</th></tr>
              </thead>
              <tbody>
                {runs.slice(0, 40).map((r) => (
                  <tr key={r.id} className="border-t border-border">
                    <td className="p-1"><input type="checkbox" aria-label={`Compare ${r.name}`} checked={selected.includes(r.id)} onChange={() => toggleSel(r.id)} disabled={!selected.includes(r.id) && selected.length >= MAX_COMPARE} /></td>
                    <td className="p-1">{LAB_EMOJI[r.lab]} {r.name}</td>
                    <td className="mono p-1 text-xs">{r.architecture}</td>
                    <td className="mono p-1 text-xs">{keyMetric(r)}</td>
                    <td className="p-1 text-xs text-muted">{new Date(r.createdAt).toLocaleString()}</td>
                    <td className="space-x-1 p-1 whitespace-nowrap">
                      <button className="btn btn-sm" aria-label={r.starred ? "Unstar" : "Mark as best"} aria-pressed={r.starred} onClick={() => toggleStar(r.id)}>{r.starred ? "⭐" : "☆"}</button>
                      <button className="btn btn-sm" aria-label={`Delete ${r.name}`} onClick={() => { deleteRun(r.id); setSelected((s) => s.filter((x) => x !== r.id)); }}>🗑</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {comparison && comparison.runs.length > 0 && (
        <section className="panel space-y-3 p-4" aria-labelledby="ml-cmp">
          <h2 id="ml-cmp" className="font-semibold">Side-by-side comparison</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData} margin={{ left: -16, right: 8 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="point" stroke="var(--muted)" fontSize={11} />
                <YAxis stroke="var(--muted)" fontSize={11} />
                <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {comparison.runs.map((r, k) => (
                  <Line key={r.id} dataKey={`r${k}`} name={`${k + 1}. ${r.name}`} stroke={CLASS_COLORS[k]} dot={false} strokeWidth={2} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-muted">Curves show validation/test loss where available (otherwise training loss), down-sampled.</p>
          <div className="overflow-x-auto">
            <table className="mono w-full text-xs">
              <thead><tr><th className="text-left">metric</th>{comparison.runs.map((r, k) => <th key={r.id} className="text-right">{k + 1}. {r.name}</th>)}</tr></thead>
              <tbody>
                {comparison.metricKeys.map((m) => (
                  <tr key={m} className="border-t border-border">
                    <td>{m}</td>
                    {comparison.runs.map((r) => (
                      <td key={r.id} className={`text-right ${comparison.best[m] === r.id ? "font-bold text-good" : ""}`}>{fmt(r.metrics[m])}</td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t border-border"><td>params</td>{comparison.runs.map((r) => <td key={r.id} className="text-right">{r.params}</td>)}</tr>
                <tr className="border-t border-border"><td>duration</td>{comparison.runs.map((r) => <td key={r.id} className="text-right">{(r.durationMs / 1000).toFixed(1)}s</td>)}</tr>
                <tr className="border-t border-border"><td>backend</td>{comparison.runs.map((r) => <td key={r.id} className="text-right">{r.backend}</td>)}</tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <section className="panel p-4" aria-labelledby="ml-best">
          <h2 id="ml-best" className="mb-2 font-semibold">⭐ Best runs</h2>
          {starred.length === 0 ? <p className="text-sm text-muted">Star a run to mark it as your best.</p> : (
            <ul className="space-y-1 text-sm">{starred.map((r) => <li key={r.id}>{LAB_EMOJI[r.lab]} {r.name} — <span className="mono">{keyMetric(r)}</span></li>)}</ul>
          )}
        </section>
        <SavedModels />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <section className="panel p-4" aria-labelledby="ml-badges">
          <h2 id="ml-badges" className="mb-2 font-semibold">🏅 Badges ({Object.keys(progress?.badges ?? {}).length}/{ALL_BADGES.length})</h2>
          <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {ALL_BADGES.map((b) => {
              const got = !!progress?.badges[b.id];
              return (
                <li key={b.id} className={`rounded-xl border p-2 text-center text-xs ${got ? "border-cyan" : "border-border opacity-50"}`} title={b.how}>
                  <span className="text-2xl" aria-hidden="true">{got ? b.emoji : "🔒"}</span>
                  <p className="font-semibold">{b.name}</p>
                  <p className="text-muted">{b.how}</p>
                </li>
              );
            })}
          </ul>
        </section>
        <section className="panel p-4" aria-labelledby="ml-academy">
          <h2 id="ml-academy" className="mb-2 font-semibold">🎓 Academy progress</h2>
          <p className="text-sm">{progress?.lessonsDone.length ?? 0} of {LESSONS.length} lessons done</p>
          <div className="mt-2 h-2 overflow-hidden rounded bg-bg-2" role="progressbar" aria-valuemin={0} aria-valuemax={LESSONS.length} aria-valuenow={progress?.lessonsDone.length ?? 0}>
            <div className="h-full bg-gradient-to-r from-cyan to-violet" style={{ width: `${((progress?.lessonsDone.length ?? 0) / LESSONS.length) * 100}%` }} />
          </div>
          <Link href="/academy" className="btn btn-sm mt-3">Continue learning →</Link>
        </section>
      </div>

      <section className="panel space-y-3 p-4" aria-labelledby="ml-storage">
        <h2 id="ml-storage" className="font-semibold">💾 Storage & backup</h2>
        <p className="text-sm text-muted">
          Runs & progress: <span className="mono">{(usage / 1024).toFixed(1)} KB</span>
          {quota && quota.quota > 0 && <> · Browser storage used by this site: <span className="mono">{(quota.usage / 1024 / 1024).toFixed(2)} MB</span> of {(quota.quota / 1024 / 1024 / 1024).toFixed(1)} GB</>}
        </p>
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-sm" onClick={onExport}>⬇️ Export all metadata (JSON)</button>
          <label className="btn btn-sm">
            ⬆️ Import backup
            <input type="file" accept="application/json,.json" className="sr-only" onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])} />
          </label>
          <button className="btn btn-sm" onClick={() => confirm("Delete all runs except ⭐ starred ones?") && clearRuns({ keepStarred: true })}>🧹 Clear unstarred runs</button>
          <button className="btn btn-sm" onClick={() => confirm("Delete ALL runs?") && clearRuns({ keepStarred: false })}>🗑 Clear all runs</button>
        </div>
      </section>
    </div>
  );
}

function fmt(v?: number) {
  return v === undefined ? "–" : Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(2);
}
function keyMetric(r: RunRecord): string {
  const m = r.metrics;
  if (m.testAccuracy !== undefined) return `test acc ${(m.testAccuracy * 100).toFixed(1)}%`;
  if (m.testR2 !== undefined) return `test R² ${m.testR2.toFixed(3)}`;
  if (m.testMse !== undefined) return `test MSE ${m.testMse.toFixed(4)}`;
  const k = Object.keys(m)[0];
  return k ? `${k} ${fmt(m[k])}` : "–";
}
