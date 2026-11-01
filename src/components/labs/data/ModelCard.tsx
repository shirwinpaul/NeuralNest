"use client";
import { useState } from "react";

export interface ModelCardData {
  modelName: string;
  datasetName: string;
  datasetSource: string;
  task: string;
  target: string;
  features: string[];
  rows: { train: number; val: number; test: number };
  metrics: { label: string; value: string }[];
  baseline: string;
  architecture: string;
  warnings: string[];
}

/** One-page, editable, printable Model Card (print stylesheet shows only this card). */
export function ModelCard({ data }: { data: ModelCardData }) {
  const [intended, setIntended] = useState(`Learning exercise: predicting “${data.target}” from ${data.features.length} features on the ${data.datasetName} dataset. Not for real-world decisions.`);
  const [limitations, setLimitations] = useState(
    [`Trained on only ${data.rows.train} examples.`, "Performance on data from a different source or time is unknown.", ...data.warnings].join("\n"),
  );
  const [ethics, setEthics] = useState("The data contains no personal information. If used on people, check results separately for different groups and keep a human in the loop.");

  const print = () => {
    document.documentElement.classList.add("print-card");
    const done = () => {
      document.documentElement.classList.remove("print-card");
      window.removeEventListener("afterprint", done);
    };
    window.addEventListener("afterprint", done);
    window.print();
  };

  return (
    <div className="space-y-3">
      <div id="model-card" className="panel space-y-3 p-5 print:border-0">
        <header className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-widest text-cyan">Model Card</p>
            <h3 className="text-xl font-bold">{data.modelName}</h3>
            <p className="text-xs text-muted">Created {new Date().toLocaleDateString()} with NeuralNest · trained in the browser with TensorFlow.js</p>
          </div>
        </header>
        <CardSection title="Model details">
          <p className="mono text-xs">{data.architecture}</p>
          <p className="text-sm">Task: {data.task} · Target: <b>{data.target}</b></p>
        </CardSection>
        <CardSection title="Intended use">
          <Editable value={intended} onChange={setIntended} label="Intended use" />
        </CardSection>
        <CardSection title="Training data">
          <p className="text-sm">{data.datasetName} — {data.datasetSource}</p>
          <p className="text-sm">Split (seeded, stratified for classification): {data.rows.train} train · {data.rows.val} validation · {data.rows.test} test</p>
          <p className="text-xs text-muted">Features: {data.features.join(", ")}</p>
        </CardSection>
        <CardSection title="Evaluation (test set)">
          <table className="mono text-sm"><tbody>
            {data.metrics.map((m) => <tr key={m.label}><td className="pr-4">{m.label}</td><td className="text-right">{m.value}</td></tr>)}
          </tbody></table>
          <p className="text-sm">Baseline for comparison: {data.baseline}</p>
        </CardSection>
        <CardSection title="Limitations">
          <Editable value={limitations} onChange={setLimitations} label="Limitations" rows={4} />
        </CardSection>
        <CardSection title="Ethical considerations">
          <Editable value={ethics} onChange={setEthics} label="Ethical considerations" />
        </CardSection>
      </div>
      <button className="btn no-print" onClick={print}>🖨️ Print / save as PDF</button>
    </div>
  );
}

function CardSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h4 className="mb-1 text-sm font-semibold text-violet">{title}</h4>
      {children}
    </section>
  );
}

function Editable({ value, onChange, label, rows = 2 }: { value: string; onChange: (v: string) => void; label: string; rows?: number }) {
  return (
    <>
      <textarea aria-label={label} className="input no-print text-sm" rows={rows} value={value} onChange={(e) => onChange(e.target.value)} maxLength={1500} />
      <p className="print-only whitespace-pre-line text-sm">{value}</p>
    </>
  );
}
