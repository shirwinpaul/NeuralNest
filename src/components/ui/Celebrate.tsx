"use client";
import { AnimatePresence, motion, useReducedMotionConfig as useReducedMotion } from "framer-motion";
import { useEffect } from "react";

const BITS = ["🎉", "✨", "⭐", "🧠", "⚡", "💜", "💙"];

/** Confetti burst + "aha!" badge reveal. Calls onDone after a few seconds. */
export function Celebrate({ badge, onDone }: { badge: { emoji: string; name: string } | null; onDone: () => void }) {
  const reduce = useReducedMotion();
  useEffect(() => {
    if (!badge) return;
    const t = setTimeout(onDone, 3800);
    return () => clearTimeout(t);
  }, [badge, onDone]);
  return (
    <AnimatePresence>
      {badge && (
        <motion.div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
          {!reduce &&
            Array.from({ length: 28 }, (_, i) => (
              <motion.span key={i} className="absolute text-2xl" initial={{ x: 0, y: 0, scale: 0.4, opacity: 1 }}
                animate={{ x: Math.cos(i * 2.4) * (180 + (i % 5) * 40), y: Math.sin(i * 2.4) * (160 + (i % 4) * 40) + 80, rotate: i * 40, opacity: 0, scale: 1 }}
                transition={{ duration: 1.8, ease: "easeOut" }}>{BITS[i % BITS.length]}</motion.span>
            ))}
          <motion.div role="status" aria-live="assertive" className="panel glow px-6 py-5 text-center"
            initial={{ scale: 0.5, rotate: -8 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: "spring", stiffness: 260, damping: 14 }}>
            <p className="text-xs uppercase tracking-widest text-cyan">aha! badge unlocked</p>
            <p className="mt-1 text-5xl">{badge.emoji}</p>
            <p className="mt-1 font-display text-xl font-bold">{badge.name}</p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
