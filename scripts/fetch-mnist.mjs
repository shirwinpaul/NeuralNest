#!/usr/bin/env node
/**
 * Fetches a small subset of MNIST (CC BY-SA 3.0 — Yann LeCun, Corinna Cortes, Christopher J.C. Burges)
 * and stores it as a compact grayscale PNG sprite + labels in public/data/.
 *
 *   npm run fetch:mnist            (default 5000 images, the cap)
 *   node scripts/fetch-mnist.mjs 2000
 *
 * Output:
 *   public/data/mnist-sprite.png   100 digits per row, each 28×28, white ink on black
 *   public/data/mnist-labels.json  { count, cols, size, labels: "504192..." }
 * Uses only Node built-ins (fetch, zlib). Run once; the app works without it.
 */
import { gunzipSync, deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const COUNT = Math.min(5000, Math.max(100, Number(process.argv[2]) || 5000));
const COLS = 100;
const SIZE = 28;
const outDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data");

async function get(name) {
  const res = await fetch(MIRROR + name);
  if (!res.ok) throw new Error(`Download failed (${res.status}) for ${name}`);
  return gunzipSync(Buffer.from(await res.arrayBuffer()));
}

// Minimal PNG encoder (8-bit grayscale, filter 0).
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function encodePNG(width, height, gray) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 0; // grayscale
  const raw = Buffer.alloc((width + 1) * height);
  for (let y = 0; y < height; y++) gray.copy(raw, y * (width + 1) + 1, y * width, (y + 1) * width);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const main = async () => {
  console.log(`Downloading MNIST (taking ${COUNT} images)…`);
  const imgs = await get("train-images-idx3-ubyte.gz");
  const labs = await get("train-labels-idx1-ubyte.gz");
  if (imgs.readUInt32BE(0) !== 2051 || labs.readUInt32BE(0) !== 2049) throw new Error("Unexpected MNIST file format");
  const rows = Math.ceil(COUNT / COLS);
  const W = COLS * SIZE, H = rows * SIZE;
  const sprite = Buffer.alloc(W * H);
  let labels = "";
  for (let i = 0; i < COUNT; i++) {
    const src = 16 + i * SIZE * SIZE;
    const ox = (i % COLS) * SIZE, oy = Math.floor(i / COLS) * SIZE;
    for (let y = 0; y < SIZE; y++) imgs.copy(sprite, (oy + y) * W + ox, src + y * SIZE, src + (y + 1) * SIZE);
    labels += String(labs[8 + i]);
  }
  mkdirSync(outDir, { recursive: true });
  const png = encodePNG(W, H, sprite);
  writeFileSync(join(outDir, "mnist-sprite.png"), png);
  writeFileSync(join(outDir, "mnist-labels.json"), JSON.stringify({ count: COUNT, cols: COLS, size: SIZE, source: "MNIST (CC BY-SA 3.0) — Yann LeCun, Corinna Cortes, Christopher J.C. Burges", labels }));
  console.log(`Saved ${COUNT} digits → public/data/mnist-sprite.png (${(png.length / 1024).toFixed(0)} KB) + mnist-labels.json`);
};

main().catch((e) => {
  console.error("Could not fetch MNIST:", e.message);
  process.exit(1);
});
