/** Playground challenges, checked in code from real metrics. */
import type { ToyKind } from "@/lib/ml/datasets/toy";

export interface PlaygroundSnapshot {
  dataset: ToyKind;
  hiddenLayers: number[]; // units per hidden layer
  numPoints: number;
  l2: number;
  epoch: number;
  trainAcc?: number;
  testAcc?: number;
  trainLoss: number;
  testLoss?: number;
  features: string[];
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  badge: { id: string; emoji: string; name: string };
  /** `flags` holds memory for multi-step challenges; may return flags to set. */
  check: (s: PlaygroundSnapshot, flags: Record<string, boolean>) => { solved: boolean; setFlags?: string[]; hint?: string };
  preset?: string;
}

const neurons = (s: PlaygroundSnapshot) => s.hiddenLayers.reduce((a, b) => a + b, 0);
const MIN_EPOCHS = 20;

export const CHALLENGES: Challenge[] = [
  {
    id: "xor-no-hidden",
    title: "XOR without hidden layers",
    description: "Reach ≥ 95% test accuracy on XOR with zero hidden layers. Hint: a clever feature can do the work of a neuron.",
    badge: { id: "feature-engineer", emoji: "🧪", name: "Feature Engineer" },
    check: (s) => ({ solved: s.dataset === "xor" && s.hiddenLayers.length === 0 && s.epoch >= MIN_EPOCHS && (s.testAcc ?? 0) >= 0.95 }),
  },
  {
    id: "circles-small",
    title: "Circles with 4 neurons or fewer",
    description: "Get ≥ 90% test accuracy on Circles using at most 4 hidden neurons in total.",
    badge: { id: "minimalist", emoji: "🪶", name: "Minimalist" },
    check: (s) => ({ solved: s.dataset === "circles" && neurons(s) <= 4 && s.epoch >= MIN_EPOCHS && (s.testAcc ?? 0) >= 0.9 }),
  },
  {
    id: "spiral-20",
    title: "Solve the spiral with fewer than 20 neurons",
    description: "Reach ≥ 90% test accuracy on Two spirals with fewer than 20 hidden neurons in total.",
    badge: { id: "spiral-tamer", emoji: "🌀", name: "Spiral Tamer" },
    check: (s) => ({ solved: s.dataset === "spirals" && neurons(s) < 20 && s.epoch >= MIN_EPOCHS && (s.testAcc ?? 0) >= 0.9 }),
    preset: "spiral",
  },
  {
    id: "overfit-fix",
    title: "Make it overfit, then fix it",
    description:
      "Step 1: with ≤ 60 points, make training accuracy beat test accuracy by ≥ 15 points. Step 2: add L2 regularisation (or shrink the model) until the gap is under 5 points with ≥ 85% test accuracy.",
    badge: { id: "overfit-detective", emoji: "🕵️", name: "Overfitting Detective" },
    check: (s, flags) => {
      if (s.epoch < MIN_EPOCHS || s.trainAcc === undefined || s.testAcc === undefined) return { solved: false };
      const gap = s.trainAcc - s.testAcc;
      if (!flags["overfit-seen"]) {
        if (s.numPoints <= 60 && gap >= 0.15) return { solved: false, setFlags: ["overfit-seen"], hint: "Overfitting spotted! Now fix it with L2 or a smaller model." };
        return { solved: false };
      }
      return { solved: gap < 0.05 && s.testAcc >= 0.85 && (s.l2 > 0 || neurons(s) <= 4) };
    },
    preset: "overfit",
  },
  {
    id: "sine-fit",
    title: "Fit the sine wave",
    description: "On Noisy sine (regression), get the test loss (MSE) below 0.02.",
    badge: { id: "wave-rider", emoji: "🌊", name: "Wave Rider" },
    check: (s) => ({ solved: s.dataset === "sine" && s.epoch >= MIN_EPOCHS && s.testLoss !== undefined && s.testLoss < 0.02 }),
  },
];

/** All badges in the app (Playground challenges + Academy). */
export const ALL_BADGES: { id: string; emoji: string; name: string; how: string }[] = [
  ...CHALLENGES.map((c) => ({ ...c.badge, how: c.title })),
  { id: "first-run", emoji: "🚀", name: "First Run", how: "Train any model" },
  { id: "data-scientist", emoji: "📊", name: "Data Scientist", how: "Compare baseline, linear and NN in the Data Lab" },
  { id: "doodle-artist", emoji: "✏️", name: "Doodle Artist", how: "Train a Doodle Classifier" },
  { id: "transfer-master", emoji: "📷", name: "Transfer Master", how: "Train an Image Teacher model" },
  { id: "bias-buster", emoji: "⚖️", name: "Bias Buster", how: "Complete the bias activity" },
  { id: "scholar", emoji: "🎓", name: "Scholar", how: "Finish 5 Academy lessons" },
  { id: "professor", emoji: "🧑‍🏫", name: "Professor", how: "Finish every Academy lesson" },
  { id: "quiz-ace", emoji: "💯", name: "Quiz Ace", how: "Score 3/3 on a quiz" },
];
