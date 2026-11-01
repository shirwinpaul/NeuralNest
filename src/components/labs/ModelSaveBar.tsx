"use client";
import { useState } from "react";
import type * as tf from "@tensorflow/tfjs";
import { downloadBlob } from "@/lib/download";
import { modelToZip, saveModelToBrowser } from "@/lib/ml/modelio";
import type { ModelMeta } from "@/lib/ml/storage";

/** "Download model" + "Save to My Models" for any trained model. */
export function ModelSaveBar({ getModel, meta }: { getModel: () => tf.LayersModel | null; meta: () => ModelMeta | null }) {
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const download = async () => {
    const m = getModel(), md = meta();
    if (!m || !md) return;
    setBusy(true);
    try {
      const zip = await modelToZip(m, md);
      downloadBlob(`${md.name.replace(/[^\w-]+/g, "_").slice(0, 40) || "model"}.neuralnest.zip`, new Blob([zip as BlobPart], { type: "application/zip" }));
      setStatus("Downloaded ✓ (model.json + weights.bin + metadata, zipped)");
    } catch {
      setStatus("Couldn't export the model.");
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    const m = getModel(), md = meta();
    if (!m || !md) return;
    setBusy(true);
    try {
      await saveModelToBrowser(m, md);
      setStatus("Saved to My Models ✓ (stored in this browser's IndexedDB)");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Couldn't save the model (storage may be full).");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button className="btn btn-sm" onClick={download} disabled={busy}>⬇️ Download model (.zip)</button>
      <button className="btn btn-sm" onClick={save} disabled={busy}>💾 Save to My Models</button>
      {status && <span className="text-xs text-muted" role="status">{status}</span>}
    </div>
  );
}
