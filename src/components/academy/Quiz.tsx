"use client";
import { useState } from "react";
import { LESSONS, type QuizQuestion } from "@/data/lessons";
import { loadProgress, recordQuiz, unlockBadge } from "@/lib/progress";
import { Celebrate } from "@/components/ui/Celebrate";
import { ALL_BADGES } from "@/data/challenges";

export function Quiz({ lessonId, questions }: { lessonId: string; questions: QuizQuestion[] }) {
  const [answers, setAnswers] = useState<(number | null)[]>(questions.map(() => null));
  const [submitted, setSubmitted] = useState(false);
  const [badge, setBadge] = useState<{ emoji: string; name: string } | null>(null);
  const score = answers.filter((a, i) => a === questions[i].answer).length;

  const submit = () => {
    setSubmitted(true);
    recordQuiz(lessonId, score);
    const done = loadProgress().lessonsDone.length;
    const unlocked: string[] = [];
    if (score === questions.length && unlockBadge("quiz-ace")) unlocked.push("quiz-ace");
    if (done >= 5 && unlockBadge("scholar")) unlocked.push("scholar");
    if (done >= LESSONS.length && unlockBadge("professor")) unlocked.push("professor");
    const b = ALL_BADGES.find((x) => x.id === unlocked[unlocked.length - 1]);
    if (b) setBadge(b);
  };

  return (
    <section className="panel space-y-4 p-4" aria-labelledby="quiz-h">
      <Celebrate badge={badge} onDone={() => setBadge(null)} />
      <h2 id="quiz-h" className="text-lg font-semibold">📝 Quick quiz</h2>
      {questions.map((q, i) => (
        <fieldset key={i} className="space-y-1">
          <legend className="font-medium">{i + 1}. {q.q}</legend>
          {q.options.map((o, j) => {
            const chosen = answers[i] === j;
            const state = submitted ? (j === q.answer ? "border-good" : chosen ? "border-bad" : "border-border") : chosen ? "border-cyan" : "border-border";
            return (
              <label key={j} className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${state}`}>
                <input type="radio" name={`q${i}`} checked={chosen} disabled={submitted} onChange={() => setAnswers(answers.map((a, k) => (k === i ? j : a)))} />
                {o}
              </label>
            );
          })}
          {submitted && <p className={`text-xs ${answers[i] === q.answer ? "text-good" : "text-amber"}`}>{answers[i] === q.answer ? "✓ Correct. " : "✗ Not quite. "}{q.why}</p>}
        </fieldset>
      ))}
      {!submitted ? (
        <button className="btn btn-primary" onClick={submit} disabled={answers.some((a) => a === null)}>Check answers</button>
      ) : (
        <div className="flex items-center gap-3" aria-live="polite">
          <p className="font-semibold">Score: {score}/{questions.length} {score >= 2 ? "— lesson complete! ✅" : "— try again to complete the lesson."}</p>
          <button className="btn btn-sm" onClick={() => { setSubmitted(false); setAnswers(questions.map(() => null)); }}>Retry</button>
        </div>
      )}
    </section>
  );
}
