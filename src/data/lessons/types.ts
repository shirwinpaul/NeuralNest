export interface QuizQuestion {
  q: string;
  options: string[];
  answer: number; // index into options
  why: string;
}

export type Illustration =
  | "ml" | "features" | "neuron" | "gradient" | "lr" | "epochs" | "overfit" | "split" | "dropout"
  | "cnn" | "transfer" | "confusion" | "regression" | "bias" | "card";

export interface Lesson {
  id: string;
  title: string;
  emoji: string;
  summary: string;
  illustration: Illustration;
  sections: { heading: string; body: string[] }[];
  tryIt?: { href: string; label: string };
  interactive?: "gradient-descent" | "activations";
  quiz: [QuizQuestion, QuizQuestion, QuizQuestion];
}
