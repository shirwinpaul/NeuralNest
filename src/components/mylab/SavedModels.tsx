"use client";
import dynamic from "next/dynamic";

/** Lazy wrapper: TensorFlow.js is only downloaded when My Lab's model panel mounts. */
export const SavedModels = dynamic(() => import("./SavedModelsPanel"), {
  ssr: false,
  loading: () => (
    <section className="panel p-4">
      <h2 className="mb-2 font-semibold">🧠 My Models</h2>
      <p className="pulse text-sm text-muted">Loading…</p>
    </section>
  ),
});
