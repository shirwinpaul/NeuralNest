"use client";

/** Confusion matrix as an accessible table with colour intensity. Rows = true class, columns = predicted. */
export function ConfusionMatrix({ matrix, labels, onCellClick }: { matrix: number[][]; labels: string[]; onCellClick?: (t: number, p: number) => void }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="mono border-separate border-spacing-1 text-xs">
        <caption className="mb-1 text-left text-xs text-muted">Rows: true class · Columns: predicted class</caption>
        <thead>
          <tr>
            <th scope="col" className="p-1 text-left text-muted">true ↓ / pred →</th>
            {labels.map((l) => <th scope="col" key={l} className="max-w-20 truncate p-1">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="max-w-24 truncate p-1 text-left">{labels[i]}</th>
              {row.map((v, j) => {
                const a = v / max;
                const diag = i === j;
                const bg = diag ? `color-mix(in srgb, var(--good) ${Math.round(a * 70)}%, transparent)` : `color-mix(in srgb, var(--bad) ${Math.round(a * 70)}%, transparent)`;
                return (
                  <td key={j} className="min-w-10 rounded p-0 text-center">
                    {onCellClick && !diag && v > 0 ? (
                      <button className="h-full w-full rounded p-2" style={{ background: bg }} onClick={() => onCellClick(i, j)} aria-label={`${v} ${labels[i]} predicted as ${labels[j]} — show examples`}>{v}</button>
                    ) : (
                      <div className="rounded p-2" style={{ background: bg }}>{v}</div>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
