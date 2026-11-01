#!/usr/bin/env node
/**
 * Generates small synthetic sample photos for the Image Teacher (no real people, no downloads):
 *   public/samples/setting-a/  circles on a YELLOW background, squares on a BLUE background
 *   public/samples/setting-b/  the same shapes with the backgrounds SWAPPED
 * Training on setting A and testing on setting B demonstrates a model that learned the
 * background instead of the shape — the bias activity.
 *   node scripts/make-samples.mjs
 */
import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const S = 96, PER = 16;
const root = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "samples");

let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (b) => { let c = ~0; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return ~c >>> 0; };
const chunk = (t, d) => { const l = Buffer.alloc(4); l.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const c = Buffer.alloc(4); c.writeUInt32BE(crc32(td)); return Buffer.concat([l, td, c]); };
function png(rgb) {
  const h = Buffer.alloc(13);
  h.writeUInt32BE(S, 0); h.writeUInt32BE(S, 4); h[8] = 8; h[9] = 2;
  const raw = Buffer.alloc((S * 3 + 1) * S);
  for (let y = 0; y < S; y++) rgb.copy(raw, y * (S * 3 + 1) + 1, y * S * 3, (y + 1) * S * 3);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", h), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

const YELLOW = [245, 214, 110], BLUE = [110, 160, 235];
function draw(shape, bg) {
  const img = Buffer.alloc(S * S * 3);
  const light = 0.85 + rnd() * 0.3; // lighting variation
  const cx = 30 + rnd() * 36, cy = 30 + rnd() * 36, r = 14 + rnd() * 10, rot = rnd() * Math.PI;
  const ink = [70 + rnd() * 40, 70 + rnd() * 40, 80 + rnd() * 40];
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      const dx = x - cx, dy = y - cy;
      const rx = dx * Math.cos(rot) + dy * Math.sin(rot), ry = -dx * Math.sin(rot) + dy * Math.cos(rot);
      const inside = shape === "circle" ? dx * dx + dy * dy <= r * r : Math.abs(rx) <= r * 0.9 && Math.abs(ry) <= r * 0.9;
      const noise = (rnd() - 0.5) * 30;
      const col = inside ? ink : bg;
      const o = (y * S + x) * 3;
      for (let k = 0; k < 3; k++) img[o + k] = Math.max(0, Math.min(255, Math.round(col[k] * light + noise)));
    }
  return png(img);
}

const index = {};
for (const [setting, bgOf] of [["setting-a", { circle: YELLOW, square: BLUE }], ["setting-b", { circle: BLUE, square: YELLOW }]]) {
  mkdirSync(join(root, setting), { recursive: true });
  index[setting] = {};
  for (const shape of ["circle", "square"]) {
    index[setting][shape] = [];
    for (let i = 0; i < PER; i++) {
      const name = `${shape}-${String(i + 1).padStart(2, "0")}.png`;
      writeFileSync(join(root, setting, name), draw(shape, bgOf[shape]));
      index[setting][shape].push(`/samples/${setting}/${name}`);
    }
  }
}
writeFileSync(join(root, "index.json"), JSON.stringify(index, null, 1));
console.log(`Wrote ${PER * 4} sample images to public/samples/`);
