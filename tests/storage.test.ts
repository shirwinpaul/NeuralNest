import { describe, expect, it } from "vitest";
import {
  clearRuns, compareRuns, deleteRun, exportBackup, importBackup, listRuns, memoryKV, parseMeta, saveRun, serializeMeta, toggleStar,
  MAX_RUNS, type ModelMeta, type NewRun,
} from "@/lib/ml/storage";

const run = (over: Partial<NewRun> = {}): NewRun => ({
  lab: "playground",
  name: "circles run",
  durationMs: 1234,
  backend: "cpu",
  architecture: "2 → 4 tanh → 1 sigmoid",
  params: 17,
  hyperparams: { learningRate: 0.03, batchSize: 10 },
  dataset: { name: "circles", nTrain: 140, nTest: 60 },
  history: Array.from({ length: 120 }, (_, i) => ({ epoch: i + 1, loss: 1 / (i + 1), valLoss: 1.2 / (i + 1) })),
  metrics: { testAccuracy: 0.9, testLoss: 0.2 },
  ...over,
});

describe("experiment tracker", () => {
  it("saves runs with down-sampled history and lists newest first", async () => {
    const kv = memoryKV();
    const a = saveRun(run({ name: "a" }), kv);
    await new Promise((r) => setTimeout(r, 2));
    saveRun(run({ name: "b" }), kv);
    const runs = listRuns(kv);
    expect(runs.map((r) => r.name)).toEqual(["b", "a"]);
    expect(a.history.length).toBeLessThanOrEqual(50);
    expect(a.history[a.history.length - 1].epoch).toBe(120);
  });

  it("stars, compares (best per metric), deletes and clears", () => {
    const kv = memoryKV();
    const a = saveRun(run({ metrics: { testAccuracy: 0.8, testLoss: 0.3 } }), kv);
    const b = saveRun(run({ metrics: { testAccuracy: 0.95, testLoss: 0.4 } }), kv);
    const c = compareRuns([a.id, b.id], kv);
    expect(c.runs).toHaveLength(2);
    expect(c.best.testAccuracy).toBe(b.id);
    expect(c.best.testLoss).toBe(a.id); // lower is better
    toggleStar(a.id, kv);
    expect(listRuns(kv).find((r) => r.id === a.id)?.starred).toBe(true);
    clearRuns({ keepStarred: true }, kv);
    expect(listRuns(kv).map((r) => r.id)).toEqual([a.id]);
    deleteRun(a.id, kv);
    expect(listRuns(kv)).toHaveLength(0);
  });

  it("compare is capped at 4 runs", () => {
    const kv = memoryKV();
    const ids = Array.from({ length: 6 }, () => saveRun(run(), kv).id);
    expect(compareRuns(ids, kv).runs).toHaveLength(4);
  });

  it("caps the number of stored runs but keeps starred ones", () => {
    const kv = memoryKV();
    const first = saveRun(run({ name: "keep" }), kv);
    toggleStar(first.id, kv);
    for (let i = 0; i < MAX_RUNS + 5; i++) saveRun(run(), kv);
    const runs = listRuns(kv);
    expect(runs.length).toBe(MAX_RUNS);
    expect(runs.some((r) => r.id === first.id)).toBe(true);
  });

  it("ignores corrupted storage instead of crashing", () => {
    const kv = memoryKV();
    kv.setItem("nn.runs.v1", "{not json");
    expect(listRuns(kv)).toEqual([]);
  });

  it("backup export/import round-trip", () => {
    const kv = memoryKV();
    saveRun(run(), kv);
    const json = exportBackup({ badges: {} }, kv);
    const kv2 = memoryKV();
    const res = importBackup(json, kv2);
    expect(res.runs).toBe(1);
    expect(listRuns(kv2)).toEqual(listRuns(kv));
    expect(() => importBackup('{"app":"other"}', kv2)).toThrow();
  });
});

describe("model metadata", () => {
  it("round-trips through JSON", () => {
    const meta: ModelMeta = {
      app: "neuralnest", version: 1, name: "iris", lab: "data", task: "classification", createdAt: 1, inputShape: [4],
      classNames: ["setosa", "versicolor", "virginica"], features: ["a", "b", "c", "d"],
      norm: { method: "zscore", shift: [1, 2, 3, 4], scale: [1, 1, 1, 1] }, featureExtractor: "none", metrics: { testAccuracy: 0.97 },
    };
    expect(parseMeta(serializeMeta(meta))).toEqual(meta);
  });
  it("rejects invalid metadata", () => {
    expect(() => parseMeta(JSON.stringify({ app: "neuralnest", version: 1, name: "" }))).toThrow();
  });
});
