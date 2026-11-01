# NeuralNest — Progress

Spec: `NEURALNEST_PROJECT_SPEC.md` (the spec refers to it as `PROJECT_SPEC.md`; same file, kept unchanged).

## Status
| Phase | Status |
|---|---|
| 1. Setup | ✅ done |
| 2. ML core | ✅ done |
| 3. Playground | ✅ done |
| 4. Tracker + visualiser + My Lab | ✅ done |
| 5. Data Lab | ✅ done |
| 6. Doodle Classifier | ✅ done |
| 7. Image Teacher | ✅ done |
| 8. Model export/import + My Models | ✅ done |
| 9. ML Academy + glossary | ✅ done |
| 10. AI tutor | ✅ done |
| 11. Polish + README | ✅ done |
| 12. End-to-end check | ⚠️ partly done (see below) |

## Phase log
### Phase 1 — Setup ✅
- Next.js 16.4 (App Router, Turbopack), React 19, TypeScript 5.9 strict, Tailwind v4, Framer Motion.
- Theme tokens in `src/app/globals.css` (dark "lab at night" default, light via `data-theme`), fonts Space Grotesk / Inter / JetBrains Mono.
- Layout with header, footer (credits, "Built by Anand Raj"), Settings dialog (theme, backend preference, reduced motion, BYOK Gemini key in localStorage).
- Landing page: animated network hero, tagline, 5 lab cards, privacy badge, "How it works".
- `.env.example` + `.env.local` (placeholders only), `.gitignore` covers `.env*.local`.

### Phase 2 — ML core ✅
- `src/lib/ml/`: `rng.ts` (seedrandom, Box–Muller, Fisher–Yates), `data.ts` (stratified seeded splits, min-max/z-score, one-hot, capped category encoder, drop/mean/mode), `metrics.ts` (accuracy, P/R/F1 per class + macro, confusion, log loss, MAE/MSE/RMSE/R², baselines, downsample), `builder.ts` (Zod config, analytic param count, per-lab limits, seeded initialisers, `disposeModel`), `trainer.ts` (epoch loop over `model.fit`, pause/resume/step/stop, early stopping, seeded per-epoch shuffle, `tf.nextFrame` yields), `engine.ts` (WebGL→WASM→CPU, memory stats). Limits in `src/config/limits.ts`.
- Tests: metrics vs hand-computed values, splits, normalisation, encoding, missing values, builder (param counts match TF.js, over-limit rejection, no tensor leaks), XOR learns ≥95%, linear regression recovers y=3x−2, pause/step/stop/early stopping.

### Phase 3 — Playground ✅
- Toy datasets (circles, XOR, spirals, moons, blobs, noisy sine), 7 engineered features, architecture editor (0–4 hidden layers, 1–8 neurons), activation/optimizer/LR/batch/L2, canvas decision-boundary heatmap (40×40, refreshed every 5 epochs), loss charts with table view, play/pause/step/reset, 5 challenges checked from real test metrics → badges + confetti. Presets via `?preset=` (`src/data/presets.ts`).

### Phase 4 — Tracker, visualiser, My Lab ✅
- `src/lib/ml/storage.ts`: runs in localStorage (metadata + metrics only, history down-sampled ≤50, cap 100 keeping ⭐), compare ≤4 with best-per-metric, backup export/import, model metadata schema. Canvas `NetworkDiagram` with live weight colouring. `/my-lab`: runs table, compare chart + table, ⭐ best, badges, Academy progress, storage usage, cleanup.
- Academy lesson content (15 lessons + quizzes) written early because My Lab uses it.

### Phase 5 — Data Lab ✅
- `profile.ts` (type detection, stats, histograms, Pearson correlations, ID/identical/near-perfect-correlation leakage, imbalance, few-rows warnings), `tabular.ts` (pipeline fitted on train rows only, one-hot with cap, missing strategies, z-scored regression targets, JSON-serialisable), `importance.ts` (grouped permutation importance), `datalab.ts` (baseline / logistic / MLP configs + evaluation), samples (real Iris CSV from scikit-learn's copy + 3 synthetic sets).
- UI: load (samples/CSV with size/row/col caps), profile, setup, train & compare, confusion / pred-vs-actual / residuals, importance, predict form, predictions CSV, editable printable Model Card (print shows only the card).
- Tests: end-to-end Iris logistic ≥ 85% (baseline < 40%), study-hours NN R² > 0.6.

### Phase 6 — Doodle ✅
- `doodle.ts` (bbox → 20 px → centre of mass in 28×28, contrast stretch, seeded shift/rotate augmentation), `saliency.ts` (|∂score/∂pixel|). `scripts/fetch-mnist.mjs` → `public/data/mnist-sprite.png` (5,000 digits, 807 KB) + labels. UI: classes with emoji, drawing pad (pointer + keyboard), gallery with delete, CNN template, augmentation toggle, charts, confusion matrix (click to filter) + misclassified gallery, live prediction bars, saliency overlay.

### Phase 7 — Image Teacher ✅
- MobileNet v2 α=0.5 via `@tensorflow-models/mobilenet` with a TF Hub `tfjs-model` URL (the package's default tfhub URLs now 404 — verified with curl; the configured URL in `src/config/models.ts` returns the model + weights). ≈ 8 MB, lazy-loaded.
- Webcam hold-to-capture, uploads, built-in synthetic sample images (`scripts/make-samples.mjs`: setting A = circles on yellow / squares on blue, setting B = swapped) for testing without a camera and for the bias activity. Transfer head vs scratch 32×32 CNN comparison, live prediction, test-a-photo.

### Phase 8 — Model export/import ✅
- `modelio.ts`: zip with standard `model.json` + `weights.bin` + `neuralnest-meta.json`; import validates metadata; IndexedDB "My Models" (cap 12) with Load & predict per lab type. Round-trip test (identical predictions).

### Phase 9 — Academy ✅
- 15 lessons (`src/data/lessons/`), original SVG illustrations, gradient-descent and activation toys, 3-question quizzes, badges (Quiz Ace, Scholar, Professor), glossary with 76 terms + search. Content-integrity tests.

### Phase 10 — AI tutor ✅
- `src/config/ai.ts`: `gemini-3.6-flash` (free tier, thinking `MINIMAL`) + Groq `llama-3.3-70b-versatile` (IDs checked in the official docs on 2026-10-24). `/api/tutor` (Node runtime, `maxDuration = 60`), per-IP 40/h, 16 KB body cap, strict Zod request (extra fields rejected), BYOK header, MOCK_AI only outside production.
- Orchestrator: 429 → wait retry-after ≤ 10 s → retry once → Groq; also falls back on 5xx/timeout/network/invalid JSON after one corrective retry; busy countdown vs offline message. Logs only provider/status/type/retry-after.
- `TutorPanel` in all 4 labs (Explain / Ask / Quiz me, "See exactly what gets sent", Try-it patches clamped by `patch.ts`).

### Phase 11 — Polish ✅
- MotionConfig follows the in-app reduced-motion setting; friendly 404; error boundary around every lab; README with setup, keys, Vercel steps, credits.
- Final: typecheck ✅, lint ✅, **98 tests in 13 files** ✅, production build ✅ (26 pages + `/api/tutor`).

### Phase 12 — End-to-end check (MOCK_AI=true) ⚠️
Done:
- Dev server ran with `MOCK_AI=true` (port 3005 — ports 3000/3001 were used by another project on this PC). Every route returned 200, unknown routes 404, static data files served; server log had no errors.
- (f) Tutor API: explain / ask / quiz returned schema-valid mock replies; payloads with extra `rows` or `pixels` fields were rejected (400). Patch clamping and "only summaries sent" are covered by unit tests.
- ML behind each lab verified headlessly: XOR learns, Iris logistic ≥ 85% vs baseline < 40%, study-hours NN R² > 0.6, importance finds the informative feature, doodle centring/augmentation/saliency, model zip export → re-import gives identical predictions.
- `MOCK_AI=false` in `.env.local` (never changed).
Not done (no browser automation available in this session):
- Clicking through each lab in a real browser: spiral challenge, overfitting with a tiny dataset, drawing 3×15 doodles, webcam / MobileNet download in the browser, model-card print, dark/light toggle and 360 px layout. These need a manual pass — steps are in the final instructions.

## Decisions
- TypeScript pinned to 5.9.3 (TS 7 is too new for the Next toolchain).
- ESLint 9 flat config using `eslint-config-next` core-web-vitals + typescript.
- Vitest runs single-worker (low-memory machine), CPU TF.js backend via `tests/setup.ts`.
- TF.js is only imported from lab chunks (`src/lib/ml/engine.ts`); landing page stays light.
- WASM backend binaries load from jsDelivr at the pinned 4.22.0 version.
- MobileNet v2 α=0.5 instead of 1.0 to save memory/download.
- AI models: `gemini-3.6-flash` + Groq `llama-3.3-70b-versatile` (`src/config/ai.ts`).

## Next
- Manual browser pass of Phase 12 (one lab at a time). If MobileNet fails to download, update the URL in `src/config/models.ts`.
