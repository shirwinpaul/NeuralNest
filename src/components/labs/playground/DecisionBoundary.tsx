"use client";
import { useEffect, useRef } from "react";
import { classColor, hexToRgb } from "@/config/palette";
import type { ToyPoint } from "@/lib/ml/datasets/toy";

export const GRID_RES = 40;
const RANGE = 1.25; // canvas covers [-RANGE, RANGE]²

/**
 * Canvas heatmap of the model's output over the input plane.
 * Classification: colour blends class 0 ↔ class 1 by predicted probability.
 * Regression: draws the predicted curve y = f(x1).
 */
export function DecisionBoundary({
  grid, curve, train, test, task, showTest,
}: {
  grid: Float32Array | null;
  curve: { x: number; y: number }[] | null;
  train: ToyPoint[];
  test: ToyPoint[];
  task: "classification" | "regression";
  showTest: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const S = c.width;
    const toPx = (v: number) => ((v + RANGE) / (2 * RANGE)) * S;
    const toPy = (v: number) => S - ((v + RANGE) / (2 * RANGE)) * S;
    const bg = getComputedStyle(c).getPropertyValue("--bg-2").trim() || "#0c1330";
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, S, S);

    const c0 = hexToRgb(classColor(0)), c1 = hexToRgb(classColor(1));
    if (task === "classification" && grid) {
      // Paint the low-res grid to an offscreen image, then scale up smoothly.
      const img = new ImageData(GRID_RES, GRID_RES);
      for (let gy = 0; gy < GRID_RES; gy++) {
        for (let gx = 0; gx < GRID_RES; gx++) {
          const p = grid[gy * GRID_RES + gx];
          const conf = Math.abs(p - 0.5) * 2; // 0 at the boundary, 1 when certain
          const col = p >= 0.5 ? c1 : c0;
          const o = ((GRID_RES - 1 - gy) * GRID_RES + gx) * 4; // flip y so +x2 is up
          img.data[o] = col[0];
          img.data[o + 1] = col[1];
          img.data[o + 2] = col[2];
          img.data[o + 3] = Math.round(40 + conf * 120);
        }
      }
      const off = document.createElement("canvas");
      off.width = off.height = GRID_RES;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, S, S);
    }

    // axes
    ctx.strokeStyle = "rgba(160,170,210,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toPx(0), 0); ctx.lineTo(toPx(0), S);
    ctx.moveTo(0, toPy(0)); ctx.lineTo(S, toPy(0));
    ctx.stroke();

    const drawPts = (pts: ToyPoint[], hollow: boolean) => {
      for (const p of pts) {
        const y = task === "regression" ? p.label : p.x2;
        ctx.beginPath();
        ctx.arc(toPx(p.x1), toPy(y), hollow ? 3.6 : 3.2, 0, Math.PI * 2);
        const col = task === "regression" ? "#56B4E9" : classColor(p.label);
        if (hollow) {
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        } else {
          ctx.fillStyle = col;
          ctx.fill();
          ctx.strokeStyle = "rgba(255,255,255,0.8)";
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    };
    drawPts(train, false);
    if (showTest) drawPts(test, true);

    if (task === "regression" && curve) {
      ctx.strokeStyle = classColor(0);
      ctx.lineWidth = 3;
      ctx.beginPath();
      curve.forEach((pt, i) => (i ? ctx.lineTo(toPx(pt.x), toPy(pt.y)) : ctx.moveTo(toPx(pt.x), toPy(pt.y))));
      ctx.stroke();
    }
  }, [grid, curve, train, test, task, showTest]);

  return (
    <canvas ref={ref} width={420} height={420} className="aspect-square w-full max-w-[420px] rounded-xl border border-border"
      role="img" aria-label={task === "classification" ? "Decision boundary: background colour shows which class the network predicts in each region. Filled dots are training points, rings are test points." : "Regression fit: dots are data, the line is the network's prediction."} />
  );
}

/** Grid of (x1, x2) coordinates in row-major order, matching the heatmap. */
export function gridCoords(): [number, number][] {
  const out: [number, number][] = [];
  for (let gy = 0; gy < GRID_RES; gy++)
    for (let gx = 0; gx < GRID_RES; gx++)
      out.push([-RANGE + ((gx + 0.5) * 2 * RANGE) / GRID_RES, -RANGE + ((gy + 0.5) * 2 * RANGE) / GRID_RES]);
  return out;
}
