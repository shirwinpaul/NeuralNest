"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { GLOSSARY } from "@/data/glossary";
import { getLesson } from "@/data/lessons";

export function GlossaryView() {
  const [q, setQ] = useState("");
  const items = useMemo(() => {
    const s = q.trim().toLowerCase();
    return [...GLOSSARY].sort((a, b) => a.term.localeCompare(b.term)).filter((g) => !s || g.term.toLowerCase().includes(s) || g.def.toLowerCase().includes(s));
  }, [q]);
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-3xl font-bold">📖 Glossary</h1>
        <p className="text-muted">{GLOSSARY.length} machine-learning words in plain language.</p>
      </div>
      <label className="block max-w-md">
        <span className="label">Search</span>
        <input className="input" type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. overfitting" />
      </label>
      <p className="sr-only" aria-live="polite">{items.length} terms shown</p>
      <dl className="grid gap-3 md:grid-cols-2">
        {items.map((g) => {
          const l = getLesson(g.lesson);
          return (
            <div key={g.term} id={g.term.toLowerCase().replace(/[^a-z0-9]+/g, "-")} className="panel p-3">
              <dt className="font-semibold text-cyan">{g.term}</dt>
              <dd className="mt-1 text-sm">{g.def}</dd>
              {l && <dd className="mt-1 text-xs"><Link href={`/academy/${l.id}`} className="text-violet underline">Lesson: {l.title}</Link></dd>}
            </div>
          );
        })}
      </dl>
    </div>
  );
}
