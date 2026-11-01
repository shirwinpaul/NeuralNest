import type { Illustration } from "@/data/lessons/types";

const C = "var(--cyan)", V = "var(--violet)", A = "var(--amber)", P = "var(--panel-2)", M = "var(--muted)";

/** Small original SVG illustrations for each lesson. Decorative. */
export function LessonArt({ kind, size = 120 }: { kind: Illustration; size?: number }) {
  return (
    <svg viewBox="0 0 120 80" width={size} height={(size * 2) / 3} aria-hidden="true">
      {ART[kind]}
    </svg>
  );
}

const dots = (pts: [number, number, string][]) => pts.map(([x, y, c], i) => <circle key={i} cx={x} cy={y} r={4} fill={c} />);

const ART: Record<Illustration, React.ReactNode> = {
  ml: (<>
    <rect x="8" y="22" width="30" height="36" rx="6" fill={P} stroke={C} />
    {dots([[16, 32, A], [28, 32, C], [16, 46, C], [28, 46, A]])}
    <path d="M44 40 h18" stroke={M} strokeWidth="2" markerEnd="url(#a)" />
    <circle cx="80" cy="40" r="16" fill={P} stroke={V} strokeWidth="2" />
    <path d="M72 40 q8 -10 16 0 q-8 10 -16 0" fill="none" stroke={C} strokeWidth="2" />
    <path d="M100 40 h12" stroke={M} strokeWidth="2" />
    <text x="104" y="34" fontSize="10" fill={A}>✓</text>
  </>),
  features: (<>
    {[0, 1, 2, 3].map((r) => [0, 1, 2].map((c) => <rect key={`${r}${c}`} x={14 + c * 26} y={10 + r * 15} width="24" height="13" rx="2" fill={c === 2 ? "color-mix(in srgb, var(--amber) 35%, transparent)" : P} stroke={c === 2 ? A : C} strokeWidth="0.8" />))}
    <text x="20" y="77" fontSize="7" fill={M}>features</text><text x="68" y="77" fontSize="7" fill={A}>label</text>
  </>),
  neuron: (<>
    {[20, 40, 60].map((y) => <line key={y} x1="14" y1={y} x2="60" y2="40" stroke={C} strokeWidth="1.5" />)}
    {[20, 40, 60].map((y) => <circle key={`c${y}`} cx="14" cy={y} r="5" fill={P} stroke={C} />)}
    <circle cx="64" cy="40" r="14" fill={P} stroke={V} strokeWidth="2" />
    <text x="58" y="44" fontSize="11" fill={V}>Σ</text>
    <path d="M84 52 L94 52 L104 28 L114 28" fill="none" stroke={A} strokeWidth="2" />
  </>),
  gradient: (<>
    <path d="M6 18 Q 40 90 64 56 T 114 20" fill="none" stroke={C} strokeWidth="2" />
    <circle cx="30" cy="44" r="5" fill={A} />
    <path d="M36 50 l10 6" stroke={A} strokeWidth="2" />
    <circle cx="52" cy="60" r="3" fill={V} />
  </>),
  lr: (<>
    <path d="M10 20 Q 60 110 110 20" fill="none" stroke={C} strokeWidth="2" />
    <path d="M20 36 L 98 34 L 26 48 L 92 46" fill="none" stroke={A} strokeWidth="1.5" strokeDasharray="3 2" />
    <path d="M30 52 l6 4 l6 3 l6 2" fill="none" stroke={V} strokeWidth="2" />
  </>),
  epochs: (<>
    <circle cx="40" cy="40" r="24" fill="none" stroke={C} strokeWidth="2" strokeDasharray="6 4" />
    <path d="M58 22 l6 -4 l0 8 z" fill={C} />
    {[0, 1, 2, 3].map((i) => <rect key={i} x={78} y={14 + i * 14} width="30" height="10" rx="2" fill={P} stroke={V} />)}
  </>),
  overfit: (<>
    {dots([[15, 60, C], [30, 48, C], [45, 44, A], [60, 36, C], [75, 30, A], [90, 24, C], [105, 18, C]])}
    <path d="M10 62 L110 16" stroke={V} strokeWidth="1.5" strokeDasharray="4 3" />
    <path d="M10 62 C20 40 26 60 32 46 S44 30 48 46 S60 20 64 36 S76 10 80 30 S100 30 110 16" fill="none" stroke={A} strokeWidth="1.5" />
  </>),
  split: (<>
    <rect x="6" y="28" width="66" height="24" rx="4" fill="color-mix(in srgb, var(--cyan) 30%, transparent)" stroke={C} />
    <rect x="74" y="28" width="18" height="24" rx="4" fill="color-mix(in srgb, var(--violet) 30%, transparent)" stroke={V} />
    <rect x="94" y="28" width="20" height="24" rx="4" fill="color-mix(in srgb, var(--amber) 30%, transparent)" stroke={A} />
    <text x="22" y="64" fontSize="7" fill={M}>train</text><text x="74" y="64" fontSize="7" fill={M}>val</text><text x="96" y="64" fontSize="7" fill={M}>test</text>
  </>),
  dropout: (<>
    {[16, 40, 64].map((y, i) => [30, 60, 90].map((x, j) => <circle key={`${i}${j}`} cx={x} cy={y} r="7" fill={(i + j) % 3 === 1 ? "none" : P} stroke={(i + j) % 3 === 1 ? M : C} strokeDasharray={(i + j) % 3 === 1 ? "2 2" : undefined} />))}
  </>),
  cnn: (<>
    <rect x="6" y="14" width="40" height="40" fill={P} stroke={C} />
    {[0, 1, 2].map((i) => <line key={i} x1="6" y1={14 + i * 13} x2="46" y2={14 + i * 13} stroke={M} strokeWidth="0.4" />)}
    <rect x="14" y="22" width="14" height="14" fill="none" stroke={A} strokeWidth="2" />
    <path d="M50 34 h10" stroke={M} strokeWidth="2" />
    {[0, 1, 2].map((i) => <rect key={`f${i}`} x={64 + i * 6} y={18 + i * 4} width="24" height="24" fill={P} stroke={V} />)}
    <path d="M100 40 h8" stroke={M} strokeWidth="2" /><circle cx="112" cy="40" r="4" fill={A} />
  </>),
  transfer: (<>
    <rect x="6" y="18" width="60" height="44" rx="6" fill={P} stroke={M} strokeDasharray="4 2" />
    <text x="14" y="44" fontSize="9" fill={M}>🔒 MobileNet</text>
    <path d="M68 40 h10" stroke={C} strokeWidth="2" />
    <rect x="80" y="26" width="32" height="28" rx="6" fill="color-mix(in srgb, var(--cyan) 25%, transparent)" stroke={C} />
    <text x="86" y="44" fontSize="9" fill={C}>head</text>
  </>),
  confusion: (<>
    {[0, 1, 2].map((r) => [0, 1, 2].map((c) => <rect key={`${r}${c}`} x={30 + c * 20} y={10 + r * 20} width="18" height="18" rx="3" fill={r === c ? "color-mix(in srgb, var(--good) 60%, transparent)" : (r + c) % 2 ? "color-mix(in srgb, var(--bad) 30%, transparent)" : P} />))}
  </>),
  regression: (<>
    {dots([[15, 62, C], [28, 52, C], [40, 50, C], [52, 40, C], [66, 36, C], [80, 26, C], [96, 22, C], [108, 12, C]])}
    <line x1="8" y1="66" x2="114" y2="10" stroke={A} strokeWidth="2" />
    <line x1="52" y1="40" x2="52" y2="43" stroke={V} strokeWidth="2" />
  </>),
  bias: (<>
    <line x1="60" y1="20" x2="60" y2="70" stroke={M} strokeWidth="2" />
    <line x1="20" y1="30" x2="100" y2="22" stroke={C} strokeWidth="2" />
    <path d="M14 30 h16 l-8 14 z" fill={A} /><path d="M92 22 h16 l-8 10 z" fill={V} />
    <rect x="48" y="68" width="24" height="5" rx="2" fill={M} />
  </>),
  card: (<>
    <rect x="24" y="8" width="72" height="64" rx="6" fill={P} stroke={C} />
    <circle cx="40" cy="26" r="8" fill="none" stroke={V} strokeWidth="2" />
    {[0, 1, 2, 3].map((i) => <rect key={i} x="54" y={20 + i * 10} width={32 - i * 4} height="4" rx="2" fill={M} />)}
    <rect x="32" y="50" width="56" height="4" rx="2" fill={A} />
  </>),
};
