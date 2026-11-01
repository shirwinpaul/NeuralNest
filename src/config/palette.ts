/** Okabe–Ito colour-blind-safe palette for classes. */
export const CLASS_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7", "#0072B2", "#D55E00", "#999999", "#882255", "#44AA99"];
export const classColor = (i: number) => CLASS_COLORS[i % CLASS_COLORS.length];

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
