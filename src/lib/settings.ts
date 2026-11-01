import { z } from "zod";

export const SettingsSchema = z.object({
  theme: z.enum(["dark", "light"]).default("dark"),
  backend: z.enum(["auto", "webgl", "wasm", "cpu"]).default("auto"),
  reducedMotion: z.enum(["system", "on", "off"]).default("system"),
  /** Bring-your-own Gemini key. Stored only in this browser's localStorage. */
  geminiKey: z.string().max(200).default(""),
});
export type Settings = z.infer<typeof SettingsSchema>;

export const DEFAULT_SETTINGS: Settings = SettingsSchema.parse({});
const KEY = "nn.settings.v1";

export function loadSettings(): Settings {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(KEY) : null;
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = SettingsSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage may be blocked (private mode) — settings just won't persist */
  }
}

/** Reflect theme / motion settings on <html> so CSS can react. */
export function applySettingsToDocument(s: Settings): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.dataset.theme = s.theme;
  if (s.reducedMotion === "system") delete root.dataset.motion;
  else root.dataset.motion = s.reducedMotion === "on" ? "off" : "on";
}

export function prefersReducedMotion(s: Settings): boolean {
  if (s.reducedMotion === "on") return true;
  if (s.reducedMotion === "off") return false;
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}
