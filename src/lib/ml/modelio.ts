/**
 * Model export / import (standard TF.js files: model.json + weights.bin, zipped with a
 * small neuralnest-meta.json) and "My Models" stored in IndexedDB (indexeddb://).
 */
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";
import { z } from "zod";
import { parseMeta, serializeMeta, type KV, type ModelMeta } from "./storage";

export const META_FILE = "neuralnest-meta.json";
const MAX_ZIP_BYTES = 30 * 1024 * 1024;

/** Serialise a model to the standard TF.js model.json + one weights.bin. */
export async function modelToFiles(model: tf.LayersModel): Promise<{ modelJson: string; weights: ArrayBuffer }> {
  let out: { modelJson: string; weights: ArrayBuffer } | null = null;
  await model.save(
    tf.io.withSaveHandler(async (a) => {
      const weights = tf.io.CompositeArrayBuffer.join(a.weightData ?? new ArrayBuffer(0));
      const json = {
        modelTopology: a.modelTopology,
        format: a.format ?? "layers-model",
        generatedBy: a.generatedBy,
        convertedBy: a.convertedBy ?? null,
        weightsManifest: [{ paths: ["weights.bin"], weights: a.weightSpecs ?? [] }],
      };
      out = { modelJson: JSON.stringify(json), weights };
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    }),
  );
  if (!out) throw new Error("Model could not be serialised");
  return out;
}

export async function modelToZip(model: tf.LayersModel, meta: ModelMeta): Promise<Uint8Array> {
  const { modelJson, weights } = await modelToFiles(model);
  const zip = new JSZip();
  zip.file("model.json", modelJson);
  zip.file("weights.bin", weights);
  zip.file(META_FILE, serializeMeta(meta));
  zip.file("README.txt", `NeuralNest model "${meta.name}" (${meta.lab} lab).\nStandard TensorFlow.js files: load with tf.loadLayersModel(). Re-import in NeuralNest → My Lab → Import model.\n`);
  return zip.generateAsync({ type: "uint8array", compression: "DEFLATE" });
}

const ModelJsonSchema = z.object({
  modelTopology: z.record(z.string(), z.unknown()),
  weightsManifest: z.array(z.object({ paths: z.array(z.string()), weights: z.array(z.unknown()) })).min(1),
});

/** Build a model from model.json text + weights; validates shape of the JSON first. */
export async function filesToModel(modelJsonText: string, weights: ArrayBuffer): Promise<tf.LayersModel> {
  const json = ModelJsonSchema.parse(JSON.parse(modelJsonText));
  const weightSpecs = json.weightsManifest.flatMap((g) => g.weights) as tf.io.WeightsManifestEntry[];
  return tf.loadLayersModel(tf.io.fromMemory({ modelTopology: json.modelTopology, weightSpecs, weightData: weights }));
}

export async function zipToModel(data: ArrayBuffer | Uint8Array): Promise<{ model: tf.LayersModel; meta: ModelMeta }> {
  if (data.byteLength > MAX_ZIP_BYTES) throw new Error("That file is too large to be a NeuralNest model.");
  const zip = await JSZip.loadAsync(data);
  const mj = zip.file("model.json"), wb = zip.file("weights.bin"), mf = zip.file(META_FILE);
  if (!mj || !wb || !mf) throw new Error("The zip must contain model.json, weights.bin and neuralnest-meta.json.");
  const meta = parseMeta(await mf.async("string"));
  const model = await filesToModel(await mj.async("string"), await wb.async("arraybuffer"));
  return { model, meta };
}

// ---------- My Models (IndexedDB weights + metadata index in localStorage) ----------
const INDEX_KEY = "nn.models.v1";
export const MAX_SAVED_MODELS = 12;
const SavedSchema = z.array(z.object({ id: z.string(), savedAt: z.number(), meta: z.unknown() }));
export interface SavedModel { id: string; savedAt: number; meta: ModelMeta }

function kv(): KV | null {
  try {
    return typeof localStorage !== "undefined" ? localStorage : null;
  } catch {
    return null;
  }
}

export function listSavedModels(store: KV | null = kv()): SavedModel[] {
  try {
    const raw = store?.getItem(INDEX_KEY);
    if (!raw) return [];
    return SavedSchema.parse(JSON.parse(raw)).flatMap((r) => {
      try {
        return [{ id: r.id, savedAt: r.savedAt, meta: parseMeta(JSON.stringify(r.meta)) }];
      } catch {
        return [];
      }
    });
  } catch {
    return [];
  }
}

function writeIndex(list: SavedModel[], store: KV | null = kv()) {
  store?.setItem(INDEX_KEY, JSON.stringify(list));
  if (typeof window !== "undefined") window.dispatchEvent(new Event("nn-models"));
}

const idbUrl = (id: string) => `indexeddb://neuralnest-${id}`;

export async function saveModelToBrowser(model: tf.LayersModel, meta: ModelMeta): Promise<SavedModel> {
  const list = listSavedModels();
  if (list.length >= MAX_SAVED_MODELS) throw new Error(`You can keep up to ${MAX_SAVED_MODELS} models. Delete one in My Lab first.`);
  const id = `${meta.lab}-${Date.now().toString(36)}`;
  await model.save(idbUrl(id));
  const rec = { id, savedAt: Date.now(), meta };
  writeIndex([rec, ...list]);
  return rec;
}

export async function loadModelFromBrowser(id: string): Promise<tf.LayersModel> {
  return tf.loadLayersModel(idbUrl(id));
}

export async function deleteModelFromBrowser(id: string): Promise<void> {
  try {
    await tf.io.removeModel(idbUrl(id));
  } catch {
    /* already gone */
  }
  writeIndex(listSavedModels().filter((m) => m.id !== id));
}
