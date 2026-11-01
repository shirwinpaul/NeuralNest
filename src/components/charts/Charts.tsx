"use client";
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, Cell, ErrorBar } from "recharts";
import { useState } from "react";

const tip = { background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 };

/** Scatter plot with an optional y=x / y=0 reference and a table view. */
export function ScatterPanel({ title, data, xLabel, yLabel, diagonal, zeroLine }: {
  title: string;
  data: { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  zeroLine?: boolean;
}) {
  const [table, setTable] = useState(false);
  const xs = data.map((d) => d.x), ys = data.map((d) => d.y);
  const lo = Math.min(...xs, ...(diagonal ? ys : [])), hi = Math.max(...xs, ...(diagonal ? ys : []));
  return (
    <figure className="panel p-3">
      <div className="mb-2 flex items-center justify-between">
        <figcaption className="text-sm font-semibold">{title}</figcaption>
        <button className="btn btn-sm" onClick={() => setTable(!table)} aria-pressed={table}>{table ? "📈 Chart" : "🔢 Table"}</button>
      </div>
      {table ? (
        <div className="max-h-56 overflow-auto">
          <table className="mono w-full text-xs"><thead><tr><th className="text-left">{xLabel}</th><th className="text-right">{yLabel}</th></tr></thead>
            <tbody>{data.slice(0, 300).map((d, i) => <tr key={i}><td>{d.x.toFixed(3)}</td><td className="text-right">{d.y.toFixed(3)}</td></tr>)}</tbody></table>
        </div>
      ) : (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 4, right: 8, bottom: 14, left: -8 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name={xLabel} stroke="var(--muted)" fontSize={11} domain={diagonal ? [lo, hi] : ["auto", "auto"]} label={{ value: xLabel, position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis type="number" dataKey="y" name={yLabel} stroke="var(--muted)" fontSize={11} domain={diagonal ? [lo, hi] : ["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(1)} />
              <Tooltip contentStyle={tip} formatter={(v) => (v as number).toFixed(3)} />
              {diagonal && <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="var(--amber)" strokeDasharray="4 4" />}
              {zeroLine && <ReferenceLine y={0} stroke="var(--amber)" strokeDasharray="4 4" />}
              <Scatter data={data} fill="var(--cyan)" isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}

/** Horizontal bar chart (e.g. feature importance), with error bars. */
export function HBarPanel({ title, data, valueLabel, note }: { title: string; data: { name: string; value: number; err?: number }[]; valueLabel: string; note?: string }) {
  return (
    <figure className="panel p-3">
      <figcaption className="mb-2 text-sm font-semibold">{title}</figcaption>
      <div style={{ height: Math.max(120, data.length * 28 + 40) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis type="number" stroke="var(--muted)" fontSize={11} tickFormatter={(v: number) => v.toFixed(2)} />
            <YAxis type="category" dataKey="name" stroke="var(--muted)" fontSize={11} width={110} />
            <Tooltip contentStyle={tip} formatter={(v) => [(v as number).toFixed(4), valueLabel]} />
            <Bar dataKey="value" isAnimationActive={false}>
              {data.map((d, i) => <Cell key={i} fill={d.value >= 0 ? "var(--cyan)" : "var(--amber)"} />)}
              <ErrorBar dataKey="err" stroke="var(--muted)" width={4} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="sr-only"><caption>{title}</caption><tbody>{data.map((d) => <tr key={d.name}><th>{d.name}</th><td>{d.value.toFixed(4)}</td></tr>)}</tbody></table>
      {note && <p className="mt-1 text-xs text-muted">{note}</p>}
    </figure>
  );
}

/** Tiny histogram bars for a data profile. */
export function MiniHistogram({ bins, label }: { bins: { x0: number; x1: number; count: number }[]; label: string }) {
  const max = Math.max(1, ...bins.map((b) => b.count));
  return (
    <div className="flex h-10 items-end gap-px" role="img" aria-label={`Histogram of ${label}: ${bins.map((b) => b.count).join(", ")}`}>
      {bins.map((b, i) => (
        <div key={i} className="flex-1 rounded-t bg-cyan/70" style={{ height: `${(b.count / max) * 100}%` }} title={`${b.x0.toFixed(2)}–${b.x1.toFixed(2)}: ${b.count}`} />
      ))}
    </div>
  );
}

/** Correlation heatmap for numeric columns (−1 amber … +1 cyan). */
export function CorrelationHeatmap({ columns, matrix }: { columns: string[]; matrix: number[][] }) {
  if (columns.length < 2) return <p className="text-sm text-muted">Need at least 2 numeric columns.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="mono border-separate border-spacing-0.5 text-[10px]">
        <caption className="mb-1 text-left text-xs text-muted">Pearson correlation (−1 … +1)</caption>
        <thead><tr><th />{columns.map((c) => <th key={c} scope="col" className="max-w-16 truncate p-1 font-normal" title={c}>{c}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="max-w-24 truncate p-1 text-left font-normal" title={columns[i]}>{columns[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="h-8 w-10 rounded text-center"
                  style={{ background: `color-mix(in srgb, ${v >= 0 ? "var(--cyan)" : "var(--amber)"} ${Math.round(Math.abs(v) * 80)}%, transparent)` }}>{v.toFixed(2)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
