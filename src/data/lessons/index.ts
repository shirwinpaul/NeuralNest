import type { Lesson } from "./types";
import { LESSONS_A } from "./part1";
import { LESSONS_B } from "./part2";

export const LESSONS: Lesson[] = [...LESSONS_A, ...LESSONS_B];
export const LESSON_IDS = LESSONS.map((l) => l.id);
export const getLesson = (id: string) => LESSONS.find((l) => l.id === id);
export type { Lesson, QuizQuestion } from "./types";
