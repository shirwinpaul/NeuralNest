"use client";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";
import { preprocessDoodle, rgbaToInk } from "@/lib/ml/doodle";

export interface DrawingPadHandle {
  clear: () => void;
  /** Preprocessed 28×28 ink image, or null if empty. */
  capture: () => Float32Array | null;
}

const SIZE = 280;

/** Big drawing area (mouse, touch, pen). Arrow keys + space let keyboard users draw too. */
export const DrawingPad = forwardRef<DrawingPadHandle, { onStroke?: () => void; label: string }>(function DrawingPad({ onStroke, label }, ref) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const cursor = useRef({ x: SIZE / 2, y: SIZE / 2 });

  const ctx = () => canvas.current?.getContext("2d", { willReadFrequently: true }) ?? null;
  const clear = useCallback(() => {
    const c = ctx();
    if (!c) return;
    c.fillStyle = "#ffffff";
    c.fillRect(0, 0, SIZE, SIZE);
  }, []);
  useEffect(clear, [clear]);

  useImperativeHandle(ref, () => ({
    clear,
    capture: () => {
      const c = ctx();
      if (!c) return null;
      return preprocessDoodle(rgbaToInk(c.getImageData(0, 0, SIZE, SIZE).data), SIZE, SIZE);
    },
  }), [clear]);

  const pos = (e: React.PointerEvent) => {
    const r = canvas.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  };
  const line = (a: { x: number; y: number }, b: { x: number; y: number }) => {
    const c = ctx();
    if (!c) return;
    c.strokeStyle = "#000";
    c.lineWidth = 16;
    c.lineCap = "round";
    c.lineJoin = "round";
    c.beginPath();
    c.moveTo(a.x, a.y);
    c.lineTo(b.x, b.y);
    c.stroke();
  };

  return (
    <canvas
      ref={canvas}
      width={SIZE}
      height={SIZE}
      tabIndex={0}
      role="img"
      aria-label={`${label}. Draw with mouse, finger or pen. Keyboard: arrow keys move the pen, hold Shift while moving to draw.`}
      className="aspect-square w-full max-w-[320px] cursor-crosshair touch-none rounded-xl border-2 border-cyan/50 bg-white"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        drawing.current = true;
        last.current = pos(e);
        line(last.current, last.current);
      }}
      onPointerMove={(e) => {
        if (!drawing.current || !last.current) return;
        const p = pos(e);
        line(last.current, p);
        last.current = p;
      }}
      onPointerUp={() => {
        drawing.current = false;
        last.current = null;
        onStroke?.();
      }}
      onPointerCancel={() => (drawing.current = false)}
      onKeyDown={(e) => {
        const step = 10;
        const d = { ArrowUp: [0, -step], ArrowDown: [0, step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] }[e.key];
        if (!d) return;
        e.preventDefault();
        const a = { ...cursor.current };
        cursor.current = { x: Math.min(SIZE, Math.max(0, a.x + d[0])), y: Math.min(SIZE, Math.max(0, a.y + d[1])) };
        if (e.shiftKey) {
          line(a, cursor.current);
          onStroke?.();
        }
      }}
    />
  );
});

/** Render a 28×28 ink image as a small thumbnail. Optional saliency overlay (0..1). */
export function Thumb({ img, size = 48, overlay, label }: { img: Float32Array; size?: number; overlay?: Float32Array | null; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current?.getContext("2d");
    if (!c) return;
    const id = new ImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = 255 - Math.round(img[i] * 255);
      let r = v, g = v, b = v;
      if (overlay) {
        const s = overlay[i];
        // blend towards hot magenta/amber for salient pixels
        r = Math.round(r * (1 - s) + 255 * s);
        g = Math.round(g * (1 - s) + 80 * s);
        b = Math.round(b * (1 - s) + 40 * s);
      }
      id.data.set([r, g, b, 255], i * 4);
    }
    c.putImageData(id, 0, 0);
  }, [img, overlay]);
  return <canvas ref={ref} width={28} height={28} style={{ width: size, height: size, imageRendering: "pixelated" }} className="rounded border border-border" role="img" aria-label={label ?? "drawing thumbnail"} />;
}
