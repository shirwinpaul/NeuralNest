import { describe, expect, it } from "vitest";
import { LESSONS } from "@/data/lessons";
import { GLOSSARY } from "@/data/glossary";
import { PLAYGROUND_PRESETS } from "@/data/presets";
import { ALL_BADGES, CHALLENGES } from "@/data/challenges";
import { resolvePreset } from "@/data/presets";
import { LIMITS } from "@/config/limits";

describe("Academy content", () => {
  it("has 12–15 lessons with unique ids and valid 3-question quizzes", () => {
    expect(LESSONS.length).toBeGreaterThanOrEqual(12);
    expect(LESSONS.length).toBeLessThanOrEqual(15);
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    for (const l of LESSONS) {
      expect(l.quiz).toHaveLength(3);
      for (const q of l.quiz) expect(q.answer).toBeLessThan(q.options.length);
      expect(l.sections.length).toBeGreaterThan(0);
    }
  });

  it("Try-it links point at real labs and presets", () => {
    const routes = ["/playground", "/data-lab", "/doodle", "/image-teacher"];
    for (const l of LESSONS) {
      if (!l.tryIt) continue;
      const [path, qs] = l.tryIt.href.split("?");
      expect(routes).toContain(path);
      const preset = new URLSearchParams(qs ?? "").get("preset");
      if (preset && path === "/playground") expect(PLAYGROUND_PRESETS[preset]).toBeDefined();
    }
  });

  it("glossary has 50+ unique terms, each linked to an existing lesson", () => {
    expect(GLOSSARY.length).toBeGreaterThanOrEqual(50);
    expect(new Set(GLOSSARY.map((g) => g.term)).size).toBe(GLOSSARY.length);
    const ids = new Set(LESSONS.map((l) => l.id));
    for (const g of GLOSSARY) expect(ids.has(g.lesson)).toBe(true);
  });
});

describe("Playground challenges & presets", () => {
  it("every preset respects the Playground limits", () => {
    for (const id of Object.keys(PLAYGROUND_PRESETS)) {
      const p = resolvePreset(id);
      expect(p.hidden.length).toBeLessThanOrEqual(4);
      expect(Math.max(0, ...p.hidden)).toBeLessThanOrEqual(LIMITS.playground.maxUnits);
      expect(p.learningRate).toBeLessThanOrEqual(LIMITS.playground.maxLearningRate);
    }
  });

  it("spiral challenge is checked from real metrics", () => {
    const ch = CHALLENGES.find((c) => c.id === "spiral-20")!;
    const base = { dataset: "spirals" as const, hiddenLayers: [8, 8], numPoints: 300, l2: 0, epoch: 100, trainLoss: 0.1, features: ["x1", "x2"] };
    expect(ch.check({ ...base, testAcc: 0.93, trainAcc: 0.95 }, {}).solved).toBe(true);
    expect(ch.check({ ...base, testAcc: 0.85, trainAcc: 0.95 }, {}).solved).toBe(false);
    expect(ch.check({ ...base, hiddenLayers: [8, 8, 8], testAcc: 0.95, trainAcc: 0.95 }, {}).solved).toBe(false);
  });

  it("overfit-then-fix needs both steps in order", () => {
    const ch = CHALLENGES.find((c) => c.id === "overfit-fix")!;
    const s = { dataset: "moons" as const, hiddenLayers: [8, 8, 8], numPoints: 40, l2: 0, epoch: 200, trainLoss: 0.01, features: ["x1", "x2"] };
    const step1 = ch.check({ ...s, trainAcc: 1, testAcc: 0.75 }, {});
    expect(step1.solved).toBe(false);
    expect(step1.setFlags).toContain("overfit-seen");
    expect(ch.check({ ...s, l2: 0.01, trainAcc: 0.9, testAcc: 0.88 }, {}).solved).toBe(false); // no flag yet
    expect(ch.check({ ...s, l2: 0.01, trainAcc: 0.9, testAcc: 0.88 }, { "overfit-seen": true }).solved).toBe(true);
  });

  it("every challenge badge is listed in ALL_BADGES", () => {
    const ids = new Set(ALL_BADGES.map((b) => b.id));
    for (const c of CHALLENGES) expect(ids.has(c.badge.id)).toBe(true);
  });
});
