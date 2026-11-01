"use client";
import { MotionConfig } from "framer-motion";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { applySettingsToDocument, DEFAULT_SETTINGS, loadSettings, saveSettings, type Settings } from "@/lib/settings";

interface Ctx {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  settingsOpen: boolean;
  setSettingsOpen: (v: boolean) => void;
}
const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    const s = loadSettings();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- hydrate from localStorage after mount
    setSettings(s);
    applySettingsToDocument(s);
  }, []);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      applySettingsToDocument(next);
      return next;
    });
  }, []);

  return (
    <SettingsContext.Provider value={{ settings, update, settingsOpen, setSettingsOpen }}>
      <MotionConfig reducedMotion={settings.reducedMotion === "on" ? "always" : settings.reducedMotion === "off" ? "never" : "user"}>{children}</MotionConfig>
    </SettingsContext.Provider>
  );
}

export function useSettings(): Ctx {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}
