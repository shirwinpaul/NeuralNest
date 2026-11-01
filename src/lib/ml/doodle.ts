/**
 * Doodle preprocessing, MNIST-style:
 *  1. find the bounding box of the ink,
 *  2. scale it (keeping aspect ratio) so the longest side is 20 px,
 *  3. paste it into a 28×28 image so the ink's centre of mass sits at the centre.
 * Input/output pixels are ink intensity in [0, 1] (1 = ink), row-major.
 */
import type { Rng } from "./rng";

export const DOODLE_SIZE = 28;
const INNER = 20;

export interface Bounds { x0: number; y0: number; x1: number; y1: number }

export function inkBounds(px: ArrayLike<number>, w: number, h: number, threshold = 0.05): Bounds | null {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (px[y * w + x] > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

/** Area-average resample of a crop to (tw × th). */
function resample(px: ArrayLike<number>, w: number, b: Bounds, tw: number, th: number): Float32Array {
  const out = new Float32Array(tw * th);
  const bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
  const sx = bw / tw, sy = bh / th;
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      const fx0 = b.x0 + tx * sx, fx1 = fx0 + sx, fy0 = b.y0 + ty * sy, fy1 = fy0 + sy;
      let sum = 0, area = 0;
      for (let y = Math.floor(fy0); y < Math.ceil(fy1); y++) {
        const wy = Math.min(fy1, y + 1) - Math.max(fy0, y);
        for (let x = Math.floor(fx0); x < Math.ceil(fx1); x++) {
          const wx = Math.min(fx1, x + 1) - Math.max(fx0, x);
          const a = wx * wy;
          sum += px[y * w + x] * a;
          area += a;
        }
      }
      out[ty * tw + tx] = area ? sum / area : 0;
    }
  }
  return out;
}

export function centreOfMass(img: ArrayLike<number>, w: number, h: number): { cx: number; cy: number } {
  let m = 0, cx = 0, cy = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x];
      m += v;
      cx += v * x;
      cy += v * y;
    }
  return m ? { cx: cx / m, cy: cy / m } : { cx: (w - 1) / 2, cy: (h - 1) / 2 };
}

/** Full preprocessing. Returns null for an empty drawing. */
export function preprocessDoodle(px: ArrayLike<number>, w: number, h: number): Float32Array | null {
  const b = inkBounds(px, w, h);
  if (!b) return null;
  const bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
  const scale = INNER / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * scale)), th = Math.max(1, Math.round(bh * scale));
  const small = resample(px, w, b, tw, th);
  const { cx, cy } = centreOfMass(small, tw, th);
  // Offset so the centre of mass lands on the image centre (13.5, 13.5).
  const ox = Math.round((DOODLE_SIZE - 1) / 2 - cx), oy = Math.round((DOODLE_SIZE - 1) / 2 - cy);
  const out = new Float32Array(DOODLE_SIZE * DOODLE_SIZE);
  for (let y = 0; y < th; y++)
    for (let x = 0; x < tw; x++) {
      const X = x + ox, Y = y + oy;
      if (X >= 0 && X < DOODLE_SIZE && Y >= 0 && Y < DOODLE_SIZE) out[Y * DOODLE_SIZE + X] = Math.min(1, small[y * tw + x]);
    }
  // Stretch contrast so the darkest stroke is full intensity.
  let mx = 0;
  for (const v of out) if (v > mx) mx = v;
  if (mx > 0) for (let i = 0; i < out.length; i++) out[i] /= mx;
  return out;
}

/** Small random shift (±2 px) + rotation (±12°) via inverse mapping with bilinear sampling. */
export function augment(img: Float32Array, rng: Rng, size = DOODLE_SIZE): Float32Array {
  const angle = ((rng() * 2 - 1) * 12 * Math.PI) / 180;
  const dx = Math.round((rng() * 2 - 1) * 2), dy = Math.round((rng() * 2 - 1) * 2);
  const c = (size - 1) / 2, cos = Math.cos(angle), sin = Math.sin(angle);
  const out = new Float32Array(size * size);
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= size || y >= size ? 0 : img[y * size + x]);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const rx = x - c - dx, ry = y - c - dy;
      const sx = cos * rx + sin * ry + c, sy = -sin * rx + cos * ry + c;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      out[y * size + x] = at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
    }
  return out;
}

/** Convert canvas RGBA (dark ink on a light pad) to ink intensity [0, 1]. */
export function rgbaToInk(data: Uint8ClampedArray): Float32Array {
  const out = new Float32Array(data.length / 4);
  for (let i = 0; i < out.length; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2], a = data[i * 4 + 3] / 255;
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    out[i] = a * (1 - lum);
  }
  return out;
}
