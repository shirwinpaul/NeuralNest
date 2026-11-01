# 🧠⚡ NeuralNest — Train-Your-Own-AI Lab

**Don't just use AI — build it.** NeuralNest is a free web app where students (13+) and curious beginners train **real neural networks in the browser** with TensorFlow.js, watch them learn live, test them, and understand *why* they behave the way they do.

> _Screenshot placeholders:_ `docs/screenshot-landing.png` · `docs/screenshot-playground.png` · `docs/screenshot-data-lab.png`

## Features
| Lab | What you do |
|---|---|
| 🎯 **Neural Playground** | 6 seeded 2D datasets, 7 engineered features, 0–4 hidden layers, live decision-boundary heatmap, loss curves, play/pause/step/reset, 5 challenges with badges |
| ✏️ **Doodle Classifier** | Define 2–5 classes, draw examples (auto-centred to 28×28 like MNIST), train a CNN, confusion matrix, "where it gets confused" gallery, live prediction, gradient saliency map. Optional MNIST digits starter pack |
| 📷 **Image Teacher** | Webcam (hold-to-capture) or photo uploads, MobileNet v2 transfer learning, scratch-CNN comparison, live prediction, hands-on **bias activity** (built-in sample images included) |
| 📊 **Data Lab** | CSV upload (≤ 5 MB / 50k rows / 50 cols) or samples, data profile + warnings (ID/leakage columns, imbalance, few rows), preprocessing, baseline vs linear/logistic vs neural network on one seeded split, confusion matrix / predicted-vs-actual / residuals, permutation importance, predict form, CSV export, printable **Model Card** |
| 🎓 **ML Academy** | 15 illustrated lessons, interactive gradient-descent & activation toys, quizzes, badges, 75+ term glossary — no AI needed |
| 🧪 **My Lab** | Experiment tracker, compare up to 4 runs, ⭐ best runs, saved models (IndexedDB) with Load & predict, model import, badges, storage cleanup, JSON backup |
| 🤖 **AI tutor** | "Explain my results", "Ask a question", "Quiz me" — one tap = one call; suggested changes are validated & clamped in code and only applied when you tap **Try it** |

## Principles
- **Real ML, honestly shown.** Every metric is computed in code from real predictions (`src/lib/ml/metrics.ts`). Nothing is faked.
- **Privacy first.** Drawings, photos, webcam frames and CSV rows never leave the browser. The tutor only receives a compact, schema-checked summary (architecture, hyperparameters, ≤ 20 down-sampled loss points, final metrics, confusion matrix, column names/types). Every tutor panel has "See exactly what gets sent".
- **Responsible AI.** Baselines everywhere, leakage & imbalance warnings, a bias activity, model cards.
- **Runs on modest devices.** Small default models, hard caps (`src/config/limits.ts`), `tf.tidy`/`dispose` everywhere, live tensor-memory "Engine" panel, WebGL → WASM → CPU fallback.

## Browser support
Chrome or Edge (latest) recommended. Firefox and Safari work for most labs; WebGL may be slower. The webcam needs HTTPS or `localhost`.

## Setup (Windows, step by step)
1. Install **Node.js 20 or newer** from <https://nodejs.org> (LTS).
2. Open the project folder in VS Code → **Terminal → New Terminal**.
3. `npm install` (if PowerShell blocks scripts, use `npm.cmd install`).
4. Open `.env.local` and paste your keys (see below). **Never commit this file** — it is git-ignored.
5. `npm run dev` → open <http://localhost:3000> in Chrome.

Optional: `npm run fetch:mnist` re-downloads the MNIST starter pack (already included in `public/data/`); `node scripts/make-samples.mjs` regenerates the sample shape images.

### Getting free AI keys
- **Gemini** (primary): <https://aistudio.google.com/apikey> → *Create API key* → paste into `GEMINI_API_KEY`.
- **Groq** (fallback): <https://console.groq.com/keys> → *Create API Key* → paste into `GROQ_API_KEY`.

The app works fully without keys — only the tutor shows "offline". Users can also add their own Gemini key in ⚙️ Settings (stored only in their browser, sent per request in a header, never logged).

### Environment variables
| Name | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini key (server only) |
| `GROQ_API_KEY` | Groq key (server only) |
| `MOCK_AI` | `true` = canned tutor replies for development. **Ignored in production builds.** |

Model IDs and all provider settings are in `src/config/ai.ts` (currently `gemini-3.6-flash` with minimal thinking, fallback `llama-3.3-70b-versatile`).

## Scripts
`npm run dev` · `npm run build` · `npm start` · `npm run typecheck` · `npm run lint` · `npm test` · `npm run check` (all four)

## Deploy to Vercel (free Hobby plan)
1. Push the project to a GitHub repository (`.env.local` is ignored automatically).
2. On <https://vercel.com> → **Add New → Project** → import the repo (framework: Next.js, defaults are fine).
3. **Settings → Environment Variables**: add `GEMINI_API_KEY` and `GROQ_API_KEY` (do *not* set `MOCK_AI`).
4. Deploy. The `/api/tutor` route sets `maxDuration = 60` (Hobby-compatible).

Notes: tutor requests are tiny JSON summaries (≤ 16 KB, far below Vercel's ~4.5 MB body limit); all heavy data stays in the browser. The per-IP limit (40 calls/hour) is **in-memory and best-effort on serverless** — each instance keeps its own counter and may restart.

## Credits & licences
- **MNIST** — Yann LeCun, Corinna Cortes, Christopher J.C. Burges. CC BY-SA 3.0. A 5,000-image subset is stored in `public/data/`.
- **Iris** — R. A. Fisher (1936), UCI Machine Learning Repository (file obtained via scikit-learn's copy).
- **MobileNet v2** — Google, Apache-2.0 (loaded at runtime from TF Hub / Kaggle Models via `@tensorflow-models/mobilenet`).
- **TensorFlow.js** — Google, Apache-2.0. Also: Next.js, React, Tailwind CSS, Framer Motion, Recharts, Papa Parse, Zod, seedrandom, JSZip.
- Other sample datasets and images are synthetic, generated in code. Illustrations and the Nesty mascot are original SVGs.

Built by **Anand Raj**.
