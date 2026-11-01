/** Nesty — NeuralNest's original robot mascot (pure SVG, no external art). */
export function Nesty({ size = 64, mood = "happy", className = "" }: { size?: number; mood?: "happy" | "thinking" | "wow"; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" className={className} role="img" aria-label="Nesty the robot">
      <defs>
        <linearGradient id="nesty-body" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="var(--cyan)" />
          <stop offset="1" stopColor="var(--violet)" />
        </linearGradient>
      </defs>
      <line x1="32" y1="6" x2="32" y2="14" stroke="var(--cyan)" strokeWidth="2" />
      <circle cx="32" cy="5" r="3" fill="var(--amber)" className="pulse" />
      <rect x="12" y="14" width="40" height="32" rx="12" fill="url(#nesty-body)" />
      <rect x="17" y="20" width="30" height="18" rx="8" fill="var(--bg)" />
      {mood === "thinking" ? (
        <>
          <rect x="22" y="27" width="7" height="2.5" rx="1" fill="var(--cyan)" />
          <rect x="35" y="27" width="7" height="2.5" rx="1" fill="var(--cyan)" />
        </>
      ) : (
        <>
          <circle cx="25.5" cy="28" r={mood === "wow" ? 4 : 3} fill="var(--cyan)" />
          <circle cx="38.5" cy="28" r={mood === "wow" ? 4 : 3} fill="var(--cyan)" />
        </>
      )}
      <path d={mood === "wow" ? "M29 34 a3 3 0 0 0 6 0" : "M27 33 q5 4 10 0"} stroke="var(--violet)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <rect x="20" y="46" width="24" height="10" rx="4" fill="var(--panel-2)" stroke="var(--border)" />
      <circle cx="26" cy="51" r="1.6" fill="var(--good)" />
      <circle cx="32" cy="51" r="1.6" fill="var(--amber)" />
      <circle cx="38" cy="51" r="1.6" fill="var(--bad)" />
    </svg>
  );
}
