"use client";
import { useEffect, useRef, useState } from "react";
import { useSettings } from "@/hooks/useSettings";

export function SettingsDialog() {
  const { settings, update, settingsOpen, setSettingsOpen } = useSettings();
  const ref = useRef<HTMLDialogElement>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (settingsOpen && !d.open) {
      setKeyDraft(settings.geminiKey);
      d.showModal();
    }
    if (!settingsOpen && d.open) d.close();
  }, [settingsOpen, settings.geminiKey]);

  return (
    <dialog ref={ref} onClose={() => setSettingsOpen(false)} aria-labelledby="settings-title"
      className="panel m-auto w-[min(92vw,520px)] p-0 text-text backdrop:bg-black/60">
      <form method="dialog" className="space-y-5 p-5" onSubmit={() => update({ geminiKey: keyDraft.trim() })}>
        <div className="flex items-center justify-between">
          <h2 id="settings-title" className="text-xl font-bold">⚙️ Settings</h2>
          <button type="button" className="btn btn-sm" onClick={() => setSettingsOpen(false)} aria-label="Close settings">✕</button>
        </div>

        <fieldset>
          <legend className="label">Theme</legend>
          <div className="flex gap-2">
            {(["dark", "light"] as const).map((t) => (
              <button type="button" key={t} aria-pressed={settings.theme === t} onClick={() => update({ theme: t })}
                className={`btn btn-sm ${settings.theme === t ? "btn-primary" : ""}`}>{t === "dark" ? "🌙 Lab at night" : "☀️ Daylight"}</button>
            ))}
          </div>
        </fieldset>

        <div>
          <label className="label" htmlFor="backend-pref">Compute backend (TensorFlow.js)</label>
          <select id="backend-pref" className="select" value={settings.backend}
            onChange={(e) => update({ backend: e.target.value as typeof settings.backend })}>
            <option value="auto">Auto (WebGL → WASM → CPU)</option>
            <option value="webgl">WebGL (GPU, fastest)</option>
            <option value="wasm">WASM (good on most laptops)</option>
            <option value="cpu">CPU (slowest, always works)</option>
          </select>
          <p className="mt-1 text-xs text-muted">Takes effect when you open or reload a lab. The active backend is shown in each lab&apos;s Engine panel.</p>
        </div>

        <div>
          <label className="label" htmlFor="motion-pref">Animations</label>
          <select id="motion-pref" className="select" value={settings.reducedMotion}
            onChange={(e) => update({ reducedMotion: e.target.value as typeof settings.reducedMotion })}>
            <option value="system">Follow my device setting</option>
            <option value="on">Reduce motion</option>
            <option value="off">Full animations</option>
          </select>
        </div>

        <div>
          <label className="label" htmlFor="byok">Your own Gemini key (optional)</label>
          <div className="flex gap-2">
            <input id="byok" className="input mono" type={showKey ? "text" : "password"} autoComplete="off" spellCheck={false}
              value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)} placeholder="Leave empty to use the app's key" />
            <button type="button" className="btn btn-sm" onClick={() => setShowKey(!showKey)}>{showKey ? "Hide" : "Show"}</button>
          </div>
          <p className="mt-1 text-xs text-muted">Stored only in this browser (localStorage). Sent with each tutor request in a header, never logged or saved on the server.</p>
        </div>

        <div className="flex justify-end gap-2">
          {settings.geminiKey && (
            <button type="button" className="btn btn-sm" onClick={() => { setKeyDraft(""); update({ geminiKey: "" }); }}>Remove key</button>
          )}
          <button type="submit" className="btn btn-primary">Save</button>
        </div>
      </form>
    </dialog>
  );
}
