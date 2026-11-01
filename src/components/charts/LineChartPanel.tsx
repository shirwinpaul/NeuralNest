"use client";
import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface Series { key: string; label: string; color: string }

/** Line chart with a "show as table" toggle so the data is accessible. */
export function LineChartPanel({
  title, data, xKey, series, height = 200, yDomain,
}: {
  title: string;
  data: Record<string, number | undefined>[];
  xKey: string;
  series: Series[];
  height?: number;
  yDomain?: [number | "auto", number | "auto"];
}) {
  const [table, setTable] = useState(false);
  return (
    <figure className="panel p-3">
      <div className="mb-2 flex items-center justify-between">
        <figcaption className="text-sm font-semibold">{title}</figcaption>
        <button className="btn btn-sm" onClick={() => setTable(!table)} aria-pressed={table}>{table ? "📈 Chart" : "🔢 Table"}</button>
      </div>
      {data.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted">No data yet — press Train.</p>
      ) : table ? (
        <div className="max-h-56 overflow-auto">
          <table className="mono w-full text-xs">
            <thead><tr><th className="text-left">{xKey}</th>{series.map((s) => <th key={s.key} className="text-right">{s.label}</th>)}</tr></thead>
            <tbody>
              {data.map((d, i) => (
                <tr key={i}><td>{d[xKey]}</td>{series.map((s) => <td key={s.key} className="text-right">{fmt(d[s.key])}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey={xKey} stroke="var(--muted)" fontSize={11} />
              <YAxis stroke="var(--muted)" fontSize={11} domain={yDomain ?? ["auto", "auto"]} tickFormatter={(v: number) => fmt(v)} />
              <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} formatter={(v) => fmt(v as number)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {series.map((s) => (
                <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}

function fmt(v: number | undefined): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  return a !== 0 && (a < 0.001 || a >= 10000) ? v.toExponential(2) : v.toFixed(a < 1 ? 3 : 2);
}
