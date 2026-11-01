"use client";
import { CorrelationHeatmap, MiniHistogram } from "@/components/charts/Charts";
import { classColor } from "@/config/palette";
import type { Profile, Warning } from "@/lib/ml/profile";

export function ProfileView({ profile, warnings, balance, target }: {
  profile: Profile;
  warnings: Warning[];
  balance: { label: string; count: number }[] | null;
  target: string | null;
}) {
  const total = balance?.reduce((a, b) => a + b.count, 0) ?? 0;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 text-sm">
        <span className="chip mono">{profile.rows.toLocaleString()} rows</span>
        <span className="chip mono">{profile.columns.length} columns</span>
        <span className="chip mono">{profile.columns.filter((c) => c.type === "numeric").length} numeric</span>
        <span className="chip mono">{profile.columns.filter((c) => c.type === "categorical").length} categorical</span>
        <span className="chip mono">{profile.columns.filter((c) => c.type === "text").length} text (ignored)</span>
      </div>

      {warnings.length > 0 && (
        <ul className="space-y-1" aria-label="Data warnings">
          {warnings.map((w, i) => (
            <li key={i} className="rounded-lg border border-amber/60 bg-amber/10 px-3 py-2 text-sm">⚠️ {w.message}</li>
          ))}
        </ul>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr><th className="p-1">Column</th><th className="p-1">Type</th><th className="p-1">Missing</th><th className="p-1">Unique</th><th className="p-1">Summary</th><th className="p-1 w-32">Distribution</th></tr>
          </thead>
          <tbody>
            {profile.columns.map((c) => (
              <tr key={c.name} className={`border-t border-border ${c.name === target ? "bg-violet/10" : ""}`}>
                <td className="p-1 font-medium">{c.name}{c.name === target && " 🎯"}</td>
                <td className="p-1"><span className="chip">{c.type}</span></td>
                <td className="mono p-1">{c.missing}</td>
                <td className="mono p-1">{c.unique}</td>
                <td className="mono p-1 text-xs">
                  {c.type === "numeric"
                    ? `min ${f(c.min)} · mean ${f(c.mean)} · max ${f(c.max)} · sd ${f(c.std)}`
                    : (c.top ?? []).slice(0, 4).map((t) => `${t.value} (${t.count})`).join(", ")}
                </td>
                <td className="p-1">{c.histogram ? <MiniHistogram bins={c.histogram} label={c.name} /> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {balance && balance.length > 0 && (
          <div>
            <p className="label">Class balance of “{target}”</p>
            <ul className="space-y-1">
              {balance.map((b, i) => (
                <li key={b.label} className="flex items-center gap-2 text-xs">
                  <span className="w-24 truncate">{b.label}</span>
                  <span className="h-3 rounded" style={{ width: `${(b.count / total) * 100}%`, background: classColor(i) }} />
                  <span className="mono">{b.count} ({((b.count / total) * 100).toFixed(0)}%)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <div>
          <p className="label">Correlations between numeric columns</p>
          <CorrelationHeatmap columns={profile.correlation.columns} matrix={profile.correlation.matrix} />
        </div>
      </div>
    </div>
  );
}

const f = (v?: number) => (v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(2));
