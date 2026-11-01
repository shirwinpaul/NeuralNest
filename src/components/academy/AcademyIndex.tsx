"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { LESSONS } from "@/data/lessons";
import { loadProgress, type Progress } from "@/lib/progress";
import { LessonArt } from "./LessonArt";

export function AcademyIndex() {
  const [p, setP] = useState<Progress | null>(null);
  useEffect(() => {
    const read = () => setP(loadProgress());
    const t = setTimeout(read, 0);
    window.addEventListener("nn-progress", read);
    return () => {
      clearTimeout(t);
      window.removeEventListener("nn-progress", read);
    };
  }, []);
  const done = p?.lessonsDone.length ?? 0;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">🎓 ML Academy</h1>
        <p className="text-muted">{LESSONS.length} short lessons. Each has a 3-question quiz and a “Try it in the lab” button. No AI needed.</p>
        <p className="mt-2 text-sm" aria-live="polite">Progress: <b>{done}</b>/{LESSONS.length} lessons complete · <Link href="/glossary" className="text-cyan underline">Glossary</Link></p>
        <div className="mt-1 h-2 max-w-md overflow-hidden rounded bg-bg-2" role="progressbar" aria-valuemin={0} aria-valuemax={LESSONS.length} aria-valuenow={done} aria-label="Academy progress">
          <div className="h-full bg-gradient-to-r from-cyan to-violet" style={{ width: `${(done / LESSONS.length) * 100}%` }} />
        </div>
      </div>
      <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {LESSONS.map((l, i) => {
          const isDone = p?.lessonsDone.includes(l.id);
          return (
            <li key={l.id}>
              <Link href={`/academy/${l.id}`} className={`panel flex h-full gap-3 p-4 transition hover:glow ${isDone ? "border-good" : ""}`}>
                <LessonArt kind={l.illustration} size={72} />
                <div>
                  <p className="text-xs text-muted">Lesson {i + 1} {isDone && <span className="text-good">✓ done</span>}</p>
                  <h2 className="font-semibold">{l.emoji} {l.title}</h2>
                  <p className="mt-1 text-sm text-muted">{l.summary}</p>
                </div>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
