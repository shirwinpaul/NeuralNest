"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { NAV } from "@/config/site";
import { useSettings } from "@/hooks/useSettings";
import { Nesty } from "./Nesty";

export function Header() {
  const pathname = usePathname();
  const { settings, update, setSettingsOpen } = useSettings();
  const [menu, setMenu] = useState(false);
  return (
    <header className="no-print sticky top-0 z-40 border-b border-border bg-bg/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-2">
        <Link href="/" className="flex items-center gap-2 font-display text-lg font-bold">
          <Nesty size={32} />
          <span>Neural<span className="text-cyan">Nest</span></span>
        </Link>
        <nav aria-label="Main" className="ml-4 hidden gap-1 lg:flex">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href}
              aria-current={pathname.startsWith(n.href) ? "page" : undefined}
              className={`rounded-lg px-3 py-1.5 text-sm ${pathname.startsWith(n.href) ? "bg-panel-2 text-cyan" : "text-muted hover:text-text"}`}>
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <button className="btn btn-sm" aria-label={`Switch to ${settings.theme === "dark" ? "light" : "dark"} mode`}
            onClick={() => update({ theme: settings.theme === "dark" ? "light" : "dark" })}>
            {settings.theme === "dark" ? "☀️" : "🌙"}
          </button>
          <button className="btn btn-sm" onClick={() => setSettingsOpen(true)} aria-label="Open settings">⚙️ <span className="hidden sm:inline">Settings</span></button>
          <button className="btn btn-sm lg:hidden" aria-expanded={menu} aria-controls="mobile-nav" onClick={() => setMenu(!menu)} aria-label="Menu">☰</button>
        </div>
      </div>
      {menu && (
        <nav id="mobile-nav" aria-label="Mobile" className="grid grid-cols-2 gap-1 border-t border-border px-4 py-2 lg:hidden">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} onClick={() => setMenu(false)} className="rounded-lg px-3 py-2 text-sm hover:bg-panel-2">{n.label}</Link>
          ))}
        </nav>
      )}
    </header>
  );
}
