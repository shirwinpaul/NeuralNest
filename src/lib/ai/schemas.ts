/** Zod schemas for the AI tutor: compact run summaries in, validated JSON out. */
import { z } from "zod";
import { AI_CONFIG } from "@/config/ai";

const short = (n: number) => z.string().max(n);
const finite = z.number().finite();

/**
 * The ONLY data the tutor ever sees: aggregated numbers + names. `.strict()` rejects
 * any extra field, so raw rows / pixels / images can't sneak into a request.
 */
export const RunSummarySchema = z
  .object({
    lab: z.enum(["playground", "data", "doodle", "image"]),
    task: z.enum(["classification", "regression"]),
    architecture: short(300),
    params: z.number().int().nonnegative(),
    hyperparams: z.record(short(40), z.union([finite, short(40), z.boolean()])).refine((r) => Object.keys(r).length <= 20, "too many hyperparams"),
    dataset: z
      .object({
        name: short(80),
        nTrain: z.number().int().nonnegative(),
        nTest: z.number().int().nonnegative(),
        classNames: z.array(short(40)).max(10).optional(),
        classCounts: z.array(z.number().int().nonnegative()).max(10).optional(),
        features: z.array(short(60)).max(30).optional(),
        featureTypes: z.array(z.enum(["numeric", "categorical", "text"])).max(30).optional(),
      })
      .strict(),
    history: z
      .array(z.object({ epoch: z.number().int(), loss: finite.nullable(), valLoss: finite.optional(), acc: finite.optional(), valAcc: finite.optional() }).strict())
      .max(20),
    metrics: z.record(short(40), finite).refine((r) => Object.keys(r).length <= 20, "too many metrics"),
    baseline: z.record(short(40), finite).optional(),
    confusion: z.array(z.array(z.number().int().nonnegative()).max(10)).max(10).optional(),
    notes: z.array(short(160)).max(5).optional(),
  })
  .strict();
export type RunSummary = z.infer<typeof RunSummarySchema>;

export const TutorModeSchema = z.enum(["explain", "ask", "quiz"]);
export type TutorMode = z.infer<typeof TutorModeSchema>;

export const TutorRequestSchema = z
  .object({
    mode: TutorModeSchema,
    summary: RunSummarySchema,
    question: z.string().trim().min(1).max(AI_CONFIG.questionMaxChars).optional(),
  })
  .strict()
  .refine((r) => r.mode !== "ask" || !!r.question, { message: "question required", path: ["question"] });
export type TutorRequest = z.infer<typeof TutorRequestSchema>;

// ---------- outputs ----------
export const ConfigPatchSchema = z
  .object({
    learningRate: z.number().optional(),
    batchSize: z.number().optional(),
    epochs: z.number().optional(),
    l2: z.number().optional(),
    dropout: z.number().optional(),
    hiddenLayers: z.array(z.number()).optional(),
    activation: z.string().optional(),
    optimizer: z.string().optional(),
  })
  .partial();
export type ConfigPatch = z.infer<typeof ConfigPatchSchema>;

export const ExplainSchema = z.object({
  summary: z.string().min(1).max(600),
  whatWentWell: z.array(z.string().max(300)).max(5),
  concerns: z.array(z.object({ issue: z.string().max(300), metric: z.string().max(120) })).max(5),
  nextExperiments: z.array(z.object({ change: z.string().max(200), why: z.string().max(300), configPatch: ConfigPatchSchema.nullish() })).max(3),
  conceptLinks: z.array(z.string().max(60)).max(5),
});
export const AskSchema = z.object({ answer: z.string().min(1).max(1200), conceptLink: z.string().max(60).nullish() });
export const QuizSchema = z.object({
  questions: z.array(z.object({ q: z.string().max(300), options: z.array(z.string().max(160)).min(2).max(4), answer: z.number().int().min(0).max(3), why: z.string().max(300) })).length(3),
});
export type ExplainOutput = z.infer<typeof ExplainSchema>;
export type AskOutput = z.infer<typeof AskSchema>;
export type QuizOutput = z.infer<typeof QuizSchema>;

export const OUTPUT_SCHEMAS = { explain: ExplainSchema, ask: AskSchema, quiz: QuizSchema } as const;

export type TutorResponse =
  | { ok: true; mode: "explain"; data: ExplainOutput; provider: string }
  | { ok: true; mode: "ask"; data: AskOutput; provider: string }
  | { ok: true; mode: "quiz"; data: QuizOutput; provider: string }
  | { ok: false; error: "busy"; retryAfterSec: number }
  | { ok: false; error: "offline" | "bad-request" | "rate-limited"; retryAfterSec?: number };
