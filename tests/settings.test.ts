import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, SettingsSchema } from "@/lib/settings";

describe("settings", () => {
  it("defaults to dark mode and auto backend", () => {
    expect(DEFAULT_SETTINGS.theme).toBe("dark");
    expect(DEFAULT_SETTINGS.backend).toBe("auto");
    expect(DEFAULT_SETTINGS.geminiKey).toBe("");
  });
  it("rejects unknown backends", () => {
    expect(SettingsSchema.safeParse({ backend: "quantum" }).success).toBe(false);
  });
});
