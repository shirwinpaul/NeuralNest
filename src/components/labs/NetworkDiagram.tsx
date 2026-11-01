"use client";
import { useEffect, useRef } from "react";

const MAX_NODES = 10;

/**
 * Canvas network visualiser. `sizes` = nodes per layer (input … output).
 * `weights[l][i][j]` (optional) = weight from node i in layer l to node j in layer l+1;
 * line colour = sign (cyan +, amber −), thickness/opacity = magnitude.
 * Layers bigger than MAX_NODES are drawn capped with a "+N" label.
 */
export function NetworkDiagram({
  sizes, labels, weights, training = false, height = 220,
}: {
  sizes: number[];
  labels?: string[];
  weights?: number[][][] | null;
  training?: boolean;
  height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = height;
    c.width = W * dpr;
    c.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(c);
    const cyan = css.getPropertyValue("--cyan").trim() || "#22d3ee";
    const amber = css.getPropertyValue("--amber").trim() || "#fbbf24";
    const panel = css.getPropertyValue("--panel-2").trim() || "#17224d";
    const text = css.getPropertyValue("--muted").trim() || "#a3aed6";

    const shown = sizes.map((s) => Math.min(s, MAX_NODES));
    const padX = 36, padY = 22;
    const colX = (l: number) => (sizes.length === 1 ? W / 2 : padX + (l * (W - 2 * padX)) / (sizes.length - 1));
    const nodeY = (l: number, i: number) => padY + ((i + 0.5) * (H - 2 * padY)) / shown[l];

    let maxW = 1e-6;
    weights?.forEach((m) => m.forEach((r) => r.forEach((w) => (maxW = Math.max(maxW, Math.abs(w))))));

    for (let l = 0; l < sizes.length - 1; l++) {
      for (let i = 0; i < shown[l]; i++) {
        for (let j = 0; j < shown[l + 1]; j++) {
          const w = weights?.[l]?.[i]?.[j];
          const mag = w === undefined ? 0.25 : Math.abs(w) / maxW;
          ctx.strokeStyle = w !== undefined && w < 0 ? amber : cyan;
          ctx.globalAlpha = 0.12 + 0.75 * mag;
          ctx.lineWidth = 0.5 + 2.5 * mag;
          ctx.beginPath();
          ctx.moveTo(colX(l), nodeY(l, i));
          ctx.lineTo(colX(l + 1), nodeY(l + 1, j));
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    const r = Math.max(5, Math.min(11, (H - 2 * padY) / (2.6 * Math.max(...shown))));
    sizes.forEach((s, l) => {
      for (let i = 0; i < shown[l]; i++) {
        ctx.beginPath();
        ctx.arc(colX(l), nodeY(l, i), r, 0, Math.PI * 2);
        ctx.fillStyle = panel;
        ctx.fill();
        ctx.strokeStyle = l === sizes.length - 1 ? amber : cyan;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.fillStyle = text;
      ctx.font = "11px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(labels?.[l] ?? String(s), colX(l), H - 4);
      if (s > MAX_NODES) ctx.fillText(`+${s - MAX_NODES}`, colX(l), 12);
    });
  }, [sizes, labels, weights, height]);

  return (
    <canvas ref={ref} style={{ height, width: "100%" }} className={training ? "pulse" : ""}
      role="img"
      aria-label={`Network diagram with layer sizes ${sizes.join(", ")}`} />
  );
}

/** Extract dense-layer kernels as nested arrays for the diagram (small models only). */
export function denseKernels(model: { layers: { getWeights: () => { shape: number[]; arraySync: () => unknown }[] }[] }): number[][][] {
  const out: number[][][] = [];
  for (const layer of model.layers) {
    const ws = layer.getWeights();
    if (ws.length && ws[0].shape.length === 2) out.push(ws[0].arraySync() as number[][]);
  }
  return out;
}
