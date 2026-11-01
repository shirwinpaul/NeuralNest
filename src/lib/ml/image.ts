/** Browser image helpers for the Image Teacher: frame capture, MobileNet embeddings, tiny pixels. */
import * as tf from "@tensorflow/tfjs";
import { MOBILENET } from "@/config/models";

type Source = HTMLVideoElement | HTMLImageElement | ImageBitmap | HTMLCanvasElement;
export const SMALL = 32; // the "from scratch" CNN sees 32×32×3 images

interface Extractor { embed: (c: HTMLCanvasElement) => Float32Array; dim: number }
let extractor: Promise<Extractor> | null = null;

/** Lazy-load MobileNet once (≈ 8 MB) and expose an embedding function. */
export function loadFeatureExtractor(): Promise<Extractor> {
  if (!extractor) {
    extractor = (async () => {
      const mobilenet = await import("@tensorflow-models/mobilenet");
      const net = await mobilenet.load({ version: 2, alpha: 0.5, modelUrl: MOBILENET.url, inputRange: [0, 1] });
      const embed = (c: HTMLCanvasElement) => {
        const t = net.infer(c, true);
        const out = t.dataSync() as Float32Array;
        t.dispose();
        return Float32Array.from(out);
      };
      const probe = document.createElement("canvas");
      probe.width = probe.height = MOBILENET.inputSize;
      return { embed, dim: embed(probe).length };
    })().catch((e) => {
      extractor = null; // allow retry
      throw e;
    });
  }
  return extractor;
}

const dims = (s: Source) =>
  s instanceof HTMLVideoElement ? { w: s.videoWidth, h: s.videoHeight } : s instanceof HTMLImageElement ? { w: s.naturalWidth, h: s.naturalHeight } : { w: s.width, h: s.height };

/** Centre-crop to a square and resize to `size`. Webcam frames are mirrored to match the preview. */
export function toSquareCanvas(src: Source, size: number = MOBILENET.inputSize, mirror = false): HTMLCanvasElement {
  const { w, h } = dims(src);
  const side = Math.min(w, h);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  if (mirror) {
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(src, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
  return c;
}

export function smallPixels(c: HTMLCanvasElement): Float32Array {
  const s = document.createElement("canvas");
  s.width = s.height = SMALL;
  const ctx = s.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(c, 0, 0, SMALL, SMALL);
  const d = ctx.getImageData(0, 0, SMALL, SMALL).data;
  const out = new Float32Array(SMALL * SMALL * 3);
  for (let i = 0; i < SMALL * SMALL; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = d[i * 4 + k] / 255;
  return out;
}

export function thumbnail(c: HTMLCanvasElement, size = 64): string {
  const t = document.createElement("canvas");
  t.width = t.height = size;
  t.getContext("2d")!.drawImage(c, 0, 0, size, size);
  return t.toDataURL("image/jpeg", 0.7);
}

export async function fileToBitmap(file: File): Promise<ImageBitmap> {
  if (!file.type.startsWith("image/")) throw new Error("Not an image");
  if (file.size > 15 * 1024 * 1024) throw new Error("Image too large");
  return createImageBitmap(file);
}

export async function urlToImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

export function stackFlat(arrs: Float32Array[], shape: number[]): tf.Tensor {
  const per = shape.reduce((a, b) => a * b, 1);
  const buf = new Float32Array(arrs.length * per);
  arrs.forEach((a, i) => buf.set(a, i * per));
  return tf.tensor(buf, [arrs.length, ...shape]);
}
