/** Badges, challenge state and Academy progress — saved locally only. */
import { z } from "zod";

const ProgressSchema = z.object({
  badges: z.record(z.string(), z.number()).default({}), // badgeId → unlock timestamp
  lessonsDone: z.array(z.string()).default([]),
  quizScores: z.record(z.string(), z.number()).default({}), // lessonId → best score (0–3)
  flags: z.record(z.string(), z.boolean()).default({}), // multi-step challenge memory
});
export type Progress = z.infer<typeof ProgressSchema>;

const KEY = "nn.progress.v1";
const EMPTY: Progress = { badges: {}, lessonsDone: [], quizScores: {}, flags: {} };

export function loadProgress(): Progress {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return structuredClone(EMPTY);
    const p = ProgressSchema.safeParse(JSON.parse(raw));
    return p.success ? p.data : structuredClone(EMPTY);
  } catch {
    return structuredClone(EMPTY);
  }
}

export function saveProgress(p: Progress): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
    window.dispatchEvent(new Event("nn-progress"));
  } catch {
    /* storage unavailable */
  }
}

/** Returns true if the badge is newly unlocked. */
export function unlockBadge(id: string): boolean {
  const p = loadProgress();
  if (p.badges[id]) return false;
  p.badges[id] = Date.now();
  saveProgress(p);
  return true;
}

export function setFlag(id: string, v = true): void {
  const p = loadProgress();
  p.flags[id] = v;
  saveProgress(p);
}

export function recordQuiz(lessonId: string, score: number): void {
  const p = loadProgress();
  p.quizScores[lessonId] = Math.max(p.quizScores[lessonId] ?? 0, score);
  if (score >= 2 && !p.lessonsDone.includes(lessonId)) p.lessonsDone.push(lessonId);
  saveProgress(p);
}

export function markLessonDone(lessonId: string): void {
  const p = loadProgress();
  if (!p.lessonsDone.includes(lessonId)) p.lessonsDone.push(lessonId);
  saveProgress(p);
}

export function resetProgress(): void {
  saveProgress(structuredClone(EMPTY));
}

export { ProgressSchema };
