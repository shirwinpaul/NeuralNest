/** Development-only mock tutor (MOCK_AI=true): deterministic replies built from the real summary. */
import type { ProviderCall } from "./providers";

export const mockProvider: ProviderCall = async (_system, user) => {
  const m = user.match(/<data name="run_summary">([\s\S]*?)<\/data>/);
  const s = m ? JSON.parse(m[1]) : {};
  const metrics: Record<string, number> = s.metrics ?? {};
  const key = Object.keys(metrics)[0];
  const val = key !== undefined ? metrics[key] : undefined;
  const lr = typeof s.hyperparams?.learningRate === "number" ? s.hyperparams.learningRate : 0.01;
  if (user.startsWith("Task: Explain")) {
    return JSON.stringify({
      summary: `(Mock tutor) Your ${s.lab} model has ${s.params} parameters and was trained on ${s.dataset?.nTrain} examples.${key ? ` Its ${key} is ${val?.toFixed(3)}.` : ""}`,
      whatWentWell: ["The loss went down during training, so the network learned something."],
      concerns: key ? [{ issue: "Check the gap between training and test results for overfitting.", metric: `${key} = ${val?.toFixed(3)}` }] : [],
      nextExperiments: [{ change: "Try a smaller learning rate", why: "Smaller steps can give a smoother, more stable loss curve.", configPatch: { learningRate: lr / 3 } }],
      conceptLinks: ["learning-rate", "overfitting"],
    });
  }
  if (user.startsWith("Task: Answer")) return JSON.stringify({ answer: "(Mock tutor) Great question! Look at the gap between your training and test loss — a big gap usually means overfitting.", conceptLink: "overfitting" });
  return JSON.stringify({
    questions: [
      { q: `(Mock) How many parameters does this model have?`, options: [String(s.params), "0", "1,000,000"], answer: 0, why: "It's shown under the architecture." },
      { q: "If test loss rises while training loss falls, that is…", options: ["Overfitting", "Underfitting", "Perfect"], answer: 0, why: "The model is memorising training data." },
      { q: "Which set gives the most honest score?", options: ["Training", "Test"], answer: 1, why: "The model never trained on it." },
    ],
  });
};
