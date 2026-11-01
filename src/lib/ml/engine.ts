/**
 * TensorFlow.js engine setup: picks a backend (WebGL → WASM → CPU) and exposes
 * memory stats for the "Engine" panel. Imported only from lab chunks, so the
 * landing page never downloads TensorFlow.js.
 */
import * as tf from "@tensorflow/tfjs";

export type BackendPref = "auto" | "webgl" | "wasm" | "cpu";
export type BackendName = "webgl" | "wasm" | "cpu";

const WASM_VERSION = "4.22.0";
let wasmRegistered = false;
let initPromise: Promise<BackendName> | null = null;
let currentPref: BackendPref | null = null;

async function tryBackend(name: BackendName): Promise<boolean> {
  try {
    if (name === "wasm" && !wasmRegistered) {
      const wasm = await import("@tensorflow/tfjs-backend-wasm");
      // WASM binaries are served from the public npm CDN (same version as the JS package).
      wasm.setWasmPaths(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${WASM_VERSION}/dist/`);
      wasmRegistered = true;
    }
    const ok = await tf.setBackend(name);
    if (!ok) return false;
    await tf.ready();
    // Smoke test: tiny op to make sure the backend really works on this device.
    const t = tf.tidy(() => tf.add(tf.scalar(1), tf.scalar(1)));
    const v = (await t.data())[0];
    t.dispose();
    return v === 2;
  } catch {
    return false;
  }
}

/** Initialise TF.js with the user's preference, falling back gracefully. */
export function initEngine(pref: BackendPref = "auto"): Promise<BackendName> {
  if (initPromise && currentPref === pref) return initPromise;
  currentPref = pref;
  const order: BackendName[] =
    pref === "auto" ? ["webgl", "wasm", "cpu"] : [pref, ...(["webgl", "wasm", "cpu"] as BackendName[]).filter((b) => b !== pref)];
  initPromise = (async () => {
    for (const b of order) {
      if (await tryBackend(b)) {
        notify();
        return b;
      }
    }
    throw new Error("No TensorFlow.js backend could start on this device.");
  })();
  return initPromise;
}

export function activeBackend(): string {
  try {
    return tf.getBackend() ?? "none";
  } catch {
    return "none";
  }
}

export interface EngineStats {
  backend: string;
  numTensors: number;
  numBytes: number;
}
export function engineStats(): EngineStats {
  const m = tf.memory();
  return { backend: activeBackend(), numTensors: m.numTensors, numBytes: m.numBytes };
}

type Listener = () => void;
const listeners = new Set<Listener>();
export function onEngineChange(l: Listener): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}
function notify() {
  listeners.forEach((l) => l());
}

export { tf };
