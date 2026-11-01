/** Built-in sample datasets for the Data Lab. Generated ones are synthetic (seeded) — no personal data. */
import { gaussian, makeRng } from "../rng";
import type { Table } from "../profile";

export interface SampleDataset {
  id: string;
  name: string;
  emoji: string;
  description: string;
  target: string;
  source: string;
}

export const SAMPLE_DATASETS: SampleDataset[] = [
  { id: "iris", name: "Iris flowers", emoji: "🌸", target: "species", description: "150 flowers, 4 measurements, 3 species. The classic first dataset.", source: "R. A. Fisher (1936), via the UCI Machine Learning Repository" },
  { id: "study", name: "Study hours → exam score", emoji: "📚", target: "exam_score", description: "Synthetic: how study, sleep and practice tests relate to a score (regression).", source: "Generated in NeuralNest (synthetic)" },
  { id: "fruit", name: "Fruit sweetness & size → type", emoji: "🍎", target: "fruit", description: "Synthetic: classify apple / orange / banana / grape from 4 features.", source: "Generated in NeuralNest (synthetic)" },
  { id: "cricket", name: "Weather → cricket match played?", emoji: "🏏", target: "played", description: "Synthetic: will the match go ahead given the weather? Includes categorical columns and some missing values.", source: "Generated in NeuralNest (synthetic)" },
];

const r2 = (v: number) => (Math.round(v * 100) / 100).toString();

export function generateStudy(n = 300, seed = 11): Table {
  const rng = makeRng(`study:${seed}`);
  const rows: string[][] = [];
  for (let i = 0; i < n; i++) {
    const hours = Math.max(0, Math.min(12, gaussian(rng, 5, 2.4)));
    const sleep = Math.max(4, Math.min(10, gaussian(rng, 7.5, 1.1)));
    const tests = Math.floor(rng() * 6);
    const music = rng() < 0.5 ? "yes" : "no"; // deliberately unrelated to the score
    const score = Math.max(0, Math.min(100, 28 + 5.2 * hours + 2.5 * (sleep - 7) + 2.8 * tests + gaussian(rng, 0, 5)));
    rows.push([r2(hours), r2(sleep), String(tests), music, r2(score)]);
  }
  return { columns: ["study_hours", "sleep_hours", "practice_tests", "listens_to_music", "exam_score"], rows };
}

export function generateFruit(n = 320, seed = 5): Table {
  const rng = makeRng(`fruit:${seed}`);
  const kinds = [
    { name: "apple", sweet: 6, size: 7.5, round: 0.9, colors: ["red", "green", "red", "yellow"] },
    { name: "orange", sweet: 5, size: 8, round: 0.95, colors: ["orange"] },
    { name: "banana", sweet: 7.5, size: 18, round: 0.2, colors: ["yellow", "yellow", "green"] },
    { name: "grape", sweet: 7, size: 2, round: 0.85, colors: ["green", "red", "purple"] },
  ];
  const rows: string[][] = [];
  for (let i = 0; i < n; i++) {
    const k = kinds[i % kinds.length];
    rows.push([
      r2(Math.max(0, gaussian(rng, k.sweet, 1))),
      r2(Math.max(0.5, gaussian(rng, k.size, k.size * 0.15))),
      r2(Math.min(1, Math.max(0, gaussian(rng, k.round, 0.07)))),
      k.colors[Math.floor(rng() * k.colors.length)],
      k.name,
    ]);
  }
  return { columns: ["sweetness", "size_cm", "roundness", "colour", "fruit"], rows };
}

export function generateCricket(n = 360, seed = 3): Table {
  const rng = makeRng(`cricket:${seed}`);
  const outlooks = ["sunny", "overcast", "rainy"];
  const rows: string[][] = [];
  for (let i = 0; i < n; i++) {
    const outlook = outlooks[Math.floor(rng() * 3)];
    const temp = gaussian(rng, outlook === "sunny" ? 31 : outlook === "overcast" ? 26 : 22, 4);
    const humidity = Math.min(100, Math.max(20, gaussian(rng, outlook === "rainy" ? 85 : 60, 12)));
    const wind = Math.max(0, gaussian(rng, 14, 7));
    const ground = rng() < 0.85 ? "covered" : "open";
    // Hidden rule (+ noise): rain with open ground or very high wind cancels the match.
    let score = 1.5 - (outlook === "rainy" ? 2.2 : 0) - (humidity > 88 ? 0.8 : 0) - (wind > 28 ? 1.5 : 0) + (ground === "covered" ? 0.6 : -0.6) + gaussian(rng, 0, 0.7);
    if (temp > 40) score -= 1;
    const played = score > 0 ? "yes" : "no";
    // ~4% missing humidity readings to practise missing-value handling
    rows.push([outlook, r2(temp), rng() < 0.04 ? "" : r2(humidity), r2(wind), ground, played]);
  }
  return { columns: ["outlook", "temperature_c", "humidity_pct", "wind_kmh", "ground", "played"], rows };
}

export async function loadSample(id: string): Promise<Table> {
  switch (id) {
    case "study": return generateStudy();
    case "fruit": return generateFruit();
    case "cricket": return generateCricket();
    case "iris": {
      const res = await fetch("/data/iris.csv");
      if (!res.ok) throw new Error("Could not load Iris");
      const lines = (await res.text()).trim().split(/\r?\n/);
      return { columns: lines[0].split(","), rows: lines.slice(1).map((l) => l.split(",")) };
    }
    default:
      throw new Error(`Unknown sample ${id}`);
  }
}
