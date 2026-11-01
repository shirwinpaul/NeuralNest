"use client";
import { useEffect, useState } from "react";
import { engineStats, initEngine, type EngineStats } from "@/lib/ml/engine";
import { useSettings } from "./useSettings";

export type EngineStatus = { status: "loading" } | { status: "ready"; backend: string } | { status: "error"; message: string };

/** Initialise TF.js with the user's backend preference; exposes status + live memory stats. */
export function useEngine(): { engine: EngineStatus; stats: EngineStats | null } {
  const { settings } = useSettings();
  const [engine, setEngine] = useState<EngineStatus>({ status: "loading" });
  const [stats, setStats] = useState<EngineStats | null>(null);

  useEffect(() => {
    let alive = true;
    initEngine(settings.backend)
      .then((b) => alive && setEngine({ status: "ready", backend: b }))
      .catch(() => alive && setEngine({ status: "error", message: "Your browser couldn't start the maths engine. Try Chrome or Edge, or pick the CPU backend in Settings." }));
    return () => {
      alive = false;
    };
  }, [settings.backend]);

  useEffect(() => {
    if (engine.status !== "ready") return;
    const tick = () => {
      try {
        setStats(engineStats());
      } catch {
        /* engine not ready */
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [engine.status]);

  return { engine, stats };
}
