"use client";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { Component, type ReactNode } from "react";
import { Nesty } from "@/components/ui/Nesty";

function Loading({ label }: { label: string }) {
  return (
    <div className="panel flex flex-col items-center gap-3 p-10 text-center" role="status" aria-live="polite">
      <Nesty size={72} mood="thinking" />
      <p className="pulse">Loading {label}…</p>
      <p className="text-xs text-muted">Downloading TensorFlow.js (about 1–2 MB, once). Everything runs on your device.</p>
    </div>
  );
}

/** Catches unexpected render errors so a lab never shows a raw crash. */
class LabErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <div className="panel p-8 text-center" role="alert">
          <Nesty size={64} mood="wow" className="mx-auto" />
          <p className="mt-2 font-semibold">Oops — this lab hit a problem.</p>
          <p className="text-sm text-muted">It may be low memory or an older browser. Reload the page, or try Chrome/Edge.</p>
          <button className="btn mt-3" onClick={() => location.reload()}>Reload</button>
        </div>
      );
    return this.props.children;
  }
}

const Playground = dynamic(() => import("./playground/Playground"), { ssr: false, loading: () => <Loading label="the Playground" /> });
const DataLab = dynamic(() => import("./data/DataLab"), { ssr: false, loading: () => <Loading label="the Data Lab" /> });
const Doodle = dynamic(() => import("./doodle/DoodleLab"), { ssr: false, loading: () => <Loading label="the Doodle Classifier" /> });
const ImageTeacher = dynamic(() => import("./image/ImageTeacher"), { ssr: false, loading: () => <Loading label="the Image Teacher" /> });

export function LabLoader({ lab }: { lab: "playground" | "data" | "doodle" | "image" }) {
  const params = useSearchParams();
  const preset = params.get("preset");
  return (
    <LabErrorBoundary>
      {lab === "playground" && <Playground presetId={preset} />}
      {lab === "data" && <DataLab presetId={preset} />}
      {lab === "doodle" && <Doodle presetId={preset} />}
      {lab === "image" && <ImageTeacher presetId={preset} />}
    </LabErrorBoundary>
  );
}
