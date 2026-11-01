/** Loads the optional MNIST starter pack (public/data/mnist-sprite.png + labels) in the browser. */
import { z } from "zod";

const LabelsSchema = z.object({ count: z.number(), cols: z.number(), size: z.literal(28), labels: z.string() });

export async function loadMnist(perClass: number): Promise<{ images: Float32Array[]; labels: number[] }> {
  const res = await fetch("/data/mnist-labels.json");
  if (!res.ok) throw new Error("MNIST starter pack not found. Run `npm run fetch:mnist` once to download it.");
  const meta = LabelsSchema.parse(await res.json());
  const img = new Image();
  img.src = "/data/mnist-sprite.png";
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, img.width, img.height).data;
  const counts = new Array(10).fill(0);
  const images: Float32Array[] = [], labels: number[] = [];
  for (let i = 0; i < meta.count; i++) {
    const d = Number(meta.labels[i]);
    if (counts[d] >= perClass) continue;
    counts[d]++;
    const ox = (i % meta.cols) * 28, oy = Math.floor(i / meta.cols) * 28;
    const out = new Float32Array(784);
    // MNIST is white ink on black, so the red channel is already ink intensity.
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) out[y * 28 + x] = data[((oy + y) * img.width + ox + x) * 4] / 255;
    images.push(out);
    labels.push(d);
    if (counts.every((n) => n >= perClass)) break;
  }
  return { images, labels };
}
