export const LABS = [
  { id: "playground", href: "/playground", emoji: "🎯", title: "Neural Playground", blurb: "Train a tiny network on 2D puzzles and watch its decision boundary grow live.", heavy: false },
  { id: "doodle", href: "/doodle", emoji: "✏️", title: "Doodle Classifier", blurb: "Draw your own classes and train a convolutional network from scratch.", heavy: true },
  { id: "image", href: "/image-teacher", emoji: "📷", title: "Image Teacher", blurb: "Teach your webcam or photos with transfer learning on MobileNet.", heavy: true },
  { id: "data", href: "/data-lab", emoji: "📊", title: "Data Lab", blurb: "Load a CSV, profile it, then compare baseline vs linear vs neural network.", heavy: false },
  { id: "academy", href: "/academy", emoji: "🎓", title: "ML Academy", blurb: "15 short lessons, quizzes and a glossary — no AI needed.", heavy: false },
] as const;

export type LabId = (typeof LABS)[number]["id"];

export const NAV = [
  { href: "/playground", label: "Playground" },
  { href: "/doodle", label: "Doodle" },
  { href: "/image-teacher", label: "Image Teacher" },
  { href: "/data-lab", label: "Data Lab" },
  { href: "/academy", label: "Academy" },
  { href: "/glossary", label: "Glossary" },
  { href: "/my-lab", label: "My Lab" },
] as const;
