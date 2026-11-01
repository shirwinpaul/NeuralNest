"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { AI_CONFIG } from "@/config/ai";
import type { LabKey } from "@/config/limits";
import { getLesson } from "@/data/lessons";
import { useSettings } from "@/hooks/useSettings";
import { describePatch, sanitizePatch, type SafePatch } from "@/lib/ai/patch";
import type { RunSummary, TutorMode, TutorResponse } from "@/lib/ai/schemas";
import { Nesty } from "@/components/ui/Nesty";

type OkResponse = Extract<TutorResponse, { ok: true }>;

/** On-demand AI tutor: one tap = one API call. Only the compact summary is sent. */
export function TutorPanel({ lab, getSummary, onApplyPatch }: { lab: LabKey; getSummary: () => RunSummary | null; onApplyPatch?: (p: SafePatch) => void }) {
  const { settings } = useSettings();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState<TutorMode | null>(null);
  const [res, setRes] = useState<OkResponse | null>(null);
  const [error, setError] = useState<{ kind: string; retry?: number } | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [question, setQuestion] = useState("");
  const [showPayload, setShowPayload] = useState(false);
  const [quizAns, setQuizAns] = useState<(number | null)[]>([null, null, null]);
  const [applied, setApplied] = useState<number | null>(null);

  useEffect(() => {
    if (countdown <= 0) return;
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  const summary = open ? getSummary() : null;

  const ask = async (mode: TutorMode) => {
    const s = getSummary();
    if (!s || countdown > 0) return;
    setLoading(mode);
    setError(null);
    setApplied(null);
    try {
      const r = await fetch("/api/tutor", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(settings.geminiKey ? { "x-gemini-key": settings.geminiKey } : {}) },
        body: JSON.stringify({ mode, summary: s, ...(mode === "ask" ? { question: question.trim().slice(0, AI_CONFIG.questionMaxChars) } : {}) }),
      });
      const data = (await r.json()) as TutorResponse;
      if (data.ok) {
        setRes(data);
        setQuizAns([null, null, null]);
      } else {
        setError({ kind: data.error, retry: data.retryAfterSec });
        if (data.retryAfterSec && (data.error === "busy" || data.error === "rate-limited")) setCountdown(data.retryAfterSec);
      }
    } catch {
      setError({ kind: "offline" });
    } finally {
      setLoading(null);
    }
  };

  return (
    <aside className="panel no-print p-4" aria-labelledby={`tutor-${lab}`}>
      <button className="flex w-full items-center gap-3 text-left" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={`tutor-body-${lab}`}>
        <Nesty size={40} mood={loading ? "thinking" : "happy"} />
        <span id={`tutor-${lab}`} className="font-semibold">🤖 Ask Nesty, the AI tutor</span>
        <span className="ml-auto text-muted">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div id={`tutor-body-${lab}`} className="mt-3 space-y-3">
          {!summary ? (
            <p className="text-sm text-muted">Train a model first — then Nesty can explain your real results.</p>
          ) : (
            <>
              <div className="flex flex-wrap gap-2">
                <button className="btn btn-sm btn-primary" disabled={!!loading || countdown > 0} onClick={() => ask("explain")}>{loading === "explain" ? "Thinking…" : "💡 Explain my results"}</button>
                <button className="btn btn-sm" disabled={!!loading || countdown > 0} onClick={() => ask("quiz")}>{loading === "quiz" ? "Thinking…" : "📝 Quiz me on this run"}</button>
              </div>
              <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (question.trim()) void ask("ask"); }}>
                <label className="sr-only" htmlFor={`q-${lab}`}>Ask a question</label>
                <input id={`q-${lab}`} className="input" maxLength={AI_CONFIG.questionMaxChars} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about this run…" />
                <button className="btn btn-sm" type="submit" disabled={!question.trim() || !!loading || countdown > 0}>{loading === "ask" ? "…" : "Ask"}</button>
              </form>
              <p className="text-right text-[10px] text-muted">{question.length}/{AI_CONFIG.questionMaxChars}</p>
            </>
          )}

          <div aria-live="polite">
            {error && (error.kind === "busy" || error.kind === "rate-limited") && (
              <p className="text-sm text-amber">⏳ The tutor is thinking about other students — try again in {countdown > 0 ? countdown : error.retry ?? 30} seconds.</p>
            )}
            {error && (error.kind === "offline" || error.kind === "bad-request") && (
              <p className="text-sm text-amber">📴 The tutor is offline, but the <Link href="/glossary" className="underline">Glossary</Link> and <Link href="/academy" className="underline">Academy</Link> have the answers too!</p>
            )}
            {res?.mode === "explain" && (
              <div className="space-y-2 text-sm">
                <p>{res.data.summary}</p>
                {res.data.whatWentWell.length > 0 && <div><p className="font-semibold text-good">What went well</p><ul className="list-disc pl-5">{res.data.whatWentWell.map((w, i) => <li key={i}>{w}</li>)}</ul></div>}
                {res.data.concerns.length > 0 && <div><p className="font-semibold text-amber">Things to watch</p><ul className="list-disc pl-5">{res.data.concerns.map((c, i) => <li key={i}>{c.issue} <span className="mono text-xs text-muted">({c.metric})</span></li>)}</ul></div>}
                {res.data.nextExperiments.length > 0 && (
                  <div>
                    <p className="font-semibold text-cyan">Try next</p>
                    <ul className="space-y-2">
                      {res.data.nextExperiments.map((x, i) => {
                        const patch = x.configPatch ? sanitizePatch(lab, x.configPatch) : null;
                        return (
                          <li key={i} className="rounded-lg border border-border p-2">
                            <p><b>{x.change}</b> — {x.why}</p>
                            {patch && onApplyPatch && (
                              <div className="mt-1 flex items-center gap-2">
                                <span className="mono text-xs text-muted">{describePatch(patch)} (checked against lab limits)</span>
                                <button className="btn btn-sm" onClick={() => { onApplyPatch(patch); setApplied(i); }}>{applied === i ? "✓ Applied — train again" : "Try it"}</button>
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}
                <LessonLinks ids={res.data.conceptLinks} />
              </div>
            )}
            {res?.mode === "ask" && (
              <div className="space-y-1 text-sm">
                <p>{res.data.answer}</p>
                {res.data.conceptLink && <LessonLinks ids={[res.data.conceptLink]} />}
              </div>
            )}
            {res?.mode === "quiz" && (
              <ol className="space-y-2 text-sm">
                {res.data.questions.map((q, i) => (
                  <li key={i}>
                    <p className="font-medium">{i + 1}. {q.q}</p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {q.options.map((o, j) => (
                        <button key={j} className={`btn btn-sm ${quizAns[i] === j ? (j === q.answer ? "border-good" : "border-bad") : ""}`} disabled={quizAns[i] !== null}
                          onClick={() => setQuizAns(quizAns.map((a, k) => (k === i ? j : a)))}>{o}</button>
                      ))}
                    </div>
                    {quizAns[i] !== null && <p className={`text-xs ${quizAns[i] === q.answer ? "text-good" : "text-amber"}`}>{quizAns[i] === q.answer ? "✓ " : `✗ Answer: ${q.options[q.answer]}. `}{q.why}</p>}
                  </li>
                ))}
              </ol>
            )}
          </div>

          <p className="text-[11px] text-muted">The tutor is an AI and can make mistakes — check the numbers on your charts. 🔒 Only summary numbers and names are sent (never your drawings, photos or data rows).{res ? ` Answered by ${res.provider}.` : ""}</p>
          {summary && (
            <details open={showPayload} onToggle={(e) => setShowPayload((e.target as HTMLDetailsElement).open)}>
              <summary className="cursor-pointer text-[11px] text-muted">See exactly what gets sent</summary>
              <pre className="mono mt-1 max-h-48 overflow-auto rounded bg-bg-2 p-2 text-[10px]">{JSON.stringify(summary, null, 1)}</pre>
            </details>
          )}
        </div>
      )}
    </aside>
  );
}

function LessonLinks({ ids }: { ids: string[] }) {
  const ls = ids.map(getLesson).filter((l): l is NonNullable<typeof l> => !!l);
  if (!ls.length) return null;
  return <p className="text-xs">📚 Learn more: {ls.map((l, i) => <span key={l.id}>{i > 0 && " · "}<Link href={`/academy/${l.id}`} className="text-cyan underline">{l.title}</Link></span>)}</p>;
}
