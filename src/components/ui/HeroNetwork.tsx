"use client";
import { motion, useReducedMotionConfig as useReducedMotion } from "framer-motion";

const LAYERS = [3, 5, 5, 2];
const W = 420, H = 260;

/** Decorative animated network for the landing hero (no TF.js needed). */
export function HeroNetwork() {
  const reduce = useReducedMotion();
  const nodes = LAYERS.map((n, li) =>
    Array.from({ length: n }, (_, i) => ({ x: 40 + (li * (W - 80)) / (LAYERS.length - 1), y: ((i + 1) * H) / (n + 1) })),
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" aria-hidden="true">
      {nodes.slice(0, -1).map((layer, li) =>
        layer.flatMap((a, ai) =>
          nodes[li + 1].map((b, bi) => (
            <motion.line key={`${li}-${ai}-${bi}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              stroke={(ai + bi) % 2 ? "var(--violet)" : "var(--cyan)"} strokeWidth={1}
              initial={{ opacity: 0.15 }}
              animate={reduce ? { opacity: 0.35 } : { opacity: [0.1, 0.7, 0.1] }}
              transition={{ duration: 2.4, repeat: Infinity, delay: (li * 0.5 + ((ai * 7 + bi * 3) % 10) * 0.12) }} />
          )),
        ),
      )}
      {nodes.flatMap((layer, li) =>
        layer.map((p, i) => (
          <motion.circle key={`n${li}-${i}`} cx={p.x} cy={p.y} r={11}
            fill="var(--panel-2)" stroke={li === LAYERS.length - 1 ? "var(--amber)" : "var(--cyan)"} strokeWidth={2}
            animate={reduce ? undefined : { scale: [1, 1.15, 1] }}
            transition={{ duration: 2.4, repeat: Infinity, delay: li * 0.5 + i * 0.1 }} />
        )),
      )}
    </svg>
  );
}
