"use client";
import type { EngineStats } from "@/lib/ml/engine";
import type { EngineStatus } from "@/hooks/useEngine";

export function EnginePanel({ engine, stats }: { engine: EngineStatus; stats: EngineStats | null }) {
  return (
    <div className="panel flex flex-wrap items-center gap-3 px-3 py-2 text-xs" aria-label="Engine status">
      <span className="font-semibold text-muted">⚙️ Engine</span>
      {engine.status === "loading" && <span className="pulse">starting TensorFlow.js…</span>}
      {engine.status === "error" && <span className="text-amber">{engine.message}</span>}
      {engine.status === "ready" && (
        <>
          <span className="chip mono">backend: <b className="text-cyan">{stats?.backend ?? engine.backend}</b></span>
          <span className="chip mono">tensors: {stats?.numTensors ?? 0}</span>
          <span className="chip mono">memory: {((stats?.numBytes ?? 0) / 1024 / 1024).toFixed(2)} MB</span>
        </>
      )}
    </div>
  );
}
