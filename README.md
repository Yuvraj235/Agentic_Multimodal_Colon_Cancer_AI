---
title: ColonAI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: A second pair of eyes for colon-cancer screening.
---

# ColonAI 🩺

**A second pair of eyes for colon-cancer screening.**

Colon cancer is one of the most preventable cancers in the world — *if* the polyp that turns into it is spotted during a routine colonoscopy. In practice, an average of **1 in 4 polyps is missed** during real screening exams, and the miss rate is even higher when the patient's scope is from a different manufacturer than the one the endoscopist trained on.

ColonAI is a research project that tries to close that gap. Upload a colonoscopy image (or stream a live colonoscopy feed) and the system gives you a plain-English answer:

> *"This looks like a growth on the bowel wall (a polyp). I'm fairly confident (87 %), and I'm focused on the upper-left of the image where the lesion appears to be. A doctor should still confirm this."*

— or, equally importantly, it refuses to answer when it's not sure.

> ⚕️ **Research & educational use only — not a medical device.** Every finding must be reviewed by a qualified clinician before any clinical decision is made.

---

## 🆕 What's new — latest hardening pass (medical-grade honesty)

A focused pass to make every output **real, not fake**, and to fix the dangerous gaps. In plain terms:

**Safer, more honest outputs**
- **Ulcerative-colitis severity — fixed.** It used to miss ~85 % of *moderate/severe* colitis (recall 0.15) by calling it "mild." After adding the **LIMUC** dataset (real Mayo-graded UC) + a UC-focused loss, it now catches **96 %** and never down-labels severe as mild. *(Trade-off: it now over-flags some mild cases as severe — the safe direction.)*
- **No more fake results.** Symptoms but no image? It no longer invents a "polyp" — it gives an **honest risk-factor assessment** instead.
- **Knows the image type.** Accepts white-light **and** NBI colonoscopy; politely refuses CT/MRI/photos and says what it thinks the image is.
- **Real "where is the polyp."** Localisation now uses the **segmentation outline** (accurate), with the fuzzy heat-map demoted to a labelled "model attention" view.
- **Honest staging.** No made-up stage from a photo — it says "not a diagnosis," and a **clinician can enter biopsy/scan findings (T/N/M) to get the exact AJCC stage** (international rulebook).
- **Real out-of-distribution detector.** The "is this a weird/unknown image?" safety net is now trained on **real** out-of-scope endoscopy (not synthetic noise) — held-out AUROC **0.996** — and is actually wired into the app.
- **View-quality warning.** Flags poor bowel-prep / obscured views as unreliable.
- **Safer chatbot.** Strict guardrails — won't diagnose, interpret your results, or invent facts; always carries a "not a doctor" disclaimer.

**New specialist models (the agent team grew)**
- **Histology specialist** — reads an H&E biopsy tile, identifies tumour vs normal tissue (9 types), feeding the staging direction.
- **Validated risk tools** — a literature-based relative-risk model **+ the APCS clinically-validated screening score** (every coefficient cited; none invented).
- **Per-site calibration tool** — a hospital can re-tune the confidence to its own data.

**Honest cross-vendor truth**
- Found & fixed a **data leak**: the "Pentax" test scope (ETIS) was accidentally in training, inflating the cross-vendor number. Measured honestly, cross-vendor segmentation is **~0.45 IoU** (not the previously claimed 0.61). Adding multi-centre **PolypGen** barely moved it — the real fix is a better image backbone (future work). We report the honest number.

**Governance**
- The autonomous daily bots that committed to this clinical repo are **paused** (human-review gate) during hardening.

> Every number above comes from a re-runnable evaluation on held-out data — no hand-edited metrics.

---

## 🚀 Try it now

| Mode | How |
| :-- | :-- |
| **Live web demo** | 👉 **[huggingface.co/spaces/Yuvraj2319/colonai](https://huggingface.co/spaces/Yuvraj2319/colonai)** — click through, no install, no signup. |
| **Run locally (one click)** | `./run_app.command` from this folder → opens at http://localhost:8501 |
| **Run locally (manual)** | `pip install -r requirements.txt && streamlit run app.py` |

Once it opens, click **Load Case A · Sigmoid Polyp** on the patient-info page and press **Analyse**. The full demo runs in under 30 seconds.

A friendly one-page instruction sheet for non-technical users is at [`RUN_ME.md`](./RUN_ME.md).

---

## 🤔 What does it actually do?

You give ColonAI three things:

1. A **colonoscopy image** (or a live video stream).
2. The **patient's symptoms** in plain English ("rectal bleeding for two weeks").
3. The **patient's medical history** (age, BMI, family history, etc.).

ColonAI then runs all three through a small team of specialised AI "agents" that each look at the case differently — one studies the picture, another reads the symptoms, another weighs the history — and a fourth one cross-checks the others. You get back:

- A **plain-English answer** that says what the AI thinks it sees, in everyday language.
- A **heat-map overlay** showing exactly where on the image the AI is paying attention. If the AI says "polyp", you can see whether it's actually looking at a polyp or just at a smudge.
- A **plain-English explanation** ("the lesion appears to be in the upper-left quadrant, the AI is paying attention to a circular region that overlaps with the predicted polyp boundary, two independent attention methods agree…").
- A **traffic-light verdict** — green when the AI is confident, amber when it isn't sure and you should ask a human, red when the picture isn't usable at all.
- **Clear next steps** based on the relevant clinical guidelines (BSG / NICE NG12).

It's all designed to be readable by someone who's never used medical software before.

---

## 🛟 Why is this important?

### Real screening misses polyps

Roughly a quarter of polyps are missed during routine colonoscopy. When a polyp is missed, it can quietly grow into a cancer over the next 5–10 years. AI assistance has been shown in large clinical trials to lift detection rates substantially — but only when the AI is trustworthy enough that the endoscopist actually pays attention to it.

### Most colonoscopy AIs only work on one brand of scope

If a model is trained on Olympus scopes (which most public datasets are), it tends to fail silently when it sees a Pentax scope. The classifier still says "polyp", but the highlight is on the wrong part of the image. The doctor sees the green light and trusts it — when in fact the AI is looking at the scope's HUD overlay, not the lesion.

We measured this directly. When we tested an off-the-shelf model on Pentax images for the first time:

- It still **correctly classified 95 % of Pentax polyps as polyps** ✅
- But its **attention heat-map landed in the wrong place 93 % of the time** ❌

A model that's right for the wrong reason is one of the most dangerous failure modes in medical AI.

### So we trained it to actually look at the polyp

ColonAI's training was reworked so the model is forced to focus its attention on the actual polyp pixels — not on the scope's branding or borders. We tested it against the same Pentax dataset afterwards:

| Test dataset (scope brand) | Old model (heat-map quality) | ColonAI (heat-map quality) |
| :-- | :--: | :--: |
| Olympus (familiar brand) | 0.24 | **0.42** |
| **Pentax (different brand)** | **0.07** | **0.16** |

That's a **+136 % improvement on the brand the model had never seen during training**. We also added a separate, dedicated segmentation step for localisation (~0.62 IoU on the familiar brand). **Honest update:** on a *truly held-out* different-brand scope, cross-vendor segmentation IoU is **~0.45** — the previously-claimed 0.61 was inflated by a now-fixed data leak (the test scope had leaked into training). See the honest-numbers table below.

### And we made it refuse to answer when it shouldn't

Every prediction passes through a central safety policy. If any of the following is true, the system **declines to display a confident reading**:

- The image doesn't look like a colonoscopy frame
- The model's confidence is below 75 %
- Different attention methods disagree about what the model is looking at
- The pathology head says "polyp" but the segmentation can't find one
- Anything in the pipeline crashes

Instead of a wrong-but-confident answer, the patient sees: *"Please ask a doctor to review this."* In screening, **silence is safer than overconfidence**.

---

## 🛡️ The three independent safety layers

ColonAI doesn't trust any single signal. A finding only reaches the user if it passes through three independent checks:

### 1. Symptom-driven safety net (clinical rules)
If the patient writes symptoms that match the NICE NG12 fast-track criteria (rectal bleeding ≥ 50 yr, iron-deficiency anaemia ≥ 60 yr, unexplained weight loss + abdominal pain, …), the urgency is escalated regardless of what the model says. The rules can only escalate, never down-grade.

### 2. Image-statistics safety net
A separate pixel-statistics engine looks for visible signs the model wasn't trained to recognise (heavy bleeding, deep cavitation, very disordered edges). If the picture looks abnormal in a way the model can't explain, the system flags it for review.

### 3. Cross-agent consistency net
The pathology agent, the visual-attention agent, the segmentation agent, and a second independent attribution method all have to **agree** on what they're seeing. If they disagree, the safety policy refuses to issue a confident answer — even if every individual agent is happy.

Every prediction is logged with the image's SHA-256, the verdict, and the agents' confidence levels — so post-hoc clinical review is always possible.

---

## 🌟 Highlights

- **6-agent multimodal pipeline** — image, symptom text, patient history, fusion, explainability, clinical recommendation.
- **Patient-friendly UI** — plain-English narrative, accessibility mode (larger fonts, dyslexia-friendly typography, high contrast, big tap targets), 1-click demo cases.
- **Live colonoscopy video** — runs at 20+ FPS on a MacBook (Apple Silicon). Single-frame false positives are filtered out by a 3-frame persistence check before anything is shown to the operator.
- **Latest Research feed** (Step 8) — auto-updated daily list of cancer-research news (ScienceDaily, MedicalXpress, Cancer Research UK). Pulled by a GitHub Action via public RSS — no API keys, no auth.
- **Calibrated confidence** — the percentage shown to the user has been calibrated against held-out data, so it's a meaningful probability rather than a vibes number.
- **REST API** — `/predict`, `/health`, `/version`, `/audit/today` — with optional `X-API-Key` auth and an audit log of every prediction.
- **Security-hardened** — upload size limits, decompression-bomb protection, MIME allow-list, sanitised error responses, CORS allow-list, owner-only audit-log permissions, XSS-safe HTML rendering. See [`SECURITY.md`](./SECURITY.md).

---

## 🤖 Daily automation (currently **paused**)

> ⏸️ **These autonomous daily bots are paused** during the medical-grade hardening period. An unsupervised job with `contents: write` that auto-commits "stale-file" deletions and external content into a *clinical* repo should run only behind a human-review (PR) gate. The cron triggers are commented out; both can still be run manually via `workflow_dispatch`. Re-enable after a review gate is in place.

| Workflow | What it does (when re-enabled) |
|:--|:--|
| [`auto-bug-check.yml`](.github/workflows/auto-bug-check.yml) | Pyflakes static analysis, JSON validity, stale-file cleanup, **live HF Space probe** via headless Chromium (Playwright). Opens a GitHub issue if the live demo silently drops to demo mode. |
| [`scrape-news.yml`](.github/workflows/scrape-news.yml) | Refreshes the cancer-news feed from public RSS sources, commits the new JSON. |

You can also run the same checks locally:

```bash
python3 scripts/auto_bug_check.py        # static analysis + cleanup + HTTP probe
python3 scripts/check_live_status.py     # Playwright probe — renders the live URL,
                                         # reads the sidebar, reports READY / DEMO_MODE
python3 scripts/scrape_cancer_news.py    # one-shot news refresh
```

---

## 📈 Honest numbers (cross-vendor held-out test)

We do **not** report a single accuracy number, because that's the metric most easily gamed. Here's the real performance:

| What we measure | On familiar brand | On unfamiliar brand |
| :-- | :--: | :--: |
| Got the diagnosis right? | 95 % | 90 % |
| Heat-map lands on the polyp? | 0.42 (IoU) | 0.16 (IoU) |
| Dedicated segmentation IoU (truly held-out) | 0.62 | **0.45** |
| Detection sensitivity (per-polyp, IoU ≥ 0.5) | 0.75–0.92 | ~0.50 |
| Confidence is well-calibrated (lower = better) | 0.06 ECE | 0.06 ECE |

> The unfamiliar-brand segmentation figure is now **0.45** (honest, fully held-out), correcting an earlier leaky **0.61**. This is the real cross-vendor generalisation gap.

We **also** publish what doesn't work / what we just fixed:

- **Ulcerative-colitis severity — now fixed.** Moderate-severe recall went from **0.15 → 0.96** after adding the LIMUC dataset + a UC-focused loss. It now errs toward *over*-calling severity (the safe direction), at some cost to mild precision.
- **Cross-vendor generalisation is real but limited (~0.45 IoU).** Adding multi-centre PolypGen barely moved it — the bottleneck is the frozen image backbone, so the genuine fix is backbone fine-tuning / an endoscopy-pretrained backbone (future work).
- Highly atypical lesions or invasive cancers are **out of distribution** — the (now real-image-trained) safety net catches them and asks for human review.

These limitations are *why* the safety net exists.

---

## ☁️ Deploy your own copy

ColonAI runs as a self-contained Streamlit app. You can host it for free on Hugging Face Spaces or Streamlit Community Cloud.

The trained model is ~600 MB and is **not** kept in this Git repo (it would inflate every clone). For a working live demo you'll publish it once to a Hugging Face model repo and the app will download it on startup.

### Option 1 — Hugging Face Spaces (recommended, free)

**Step 1 — Upload the checkpoint to a model repo (one time):**

1. Go to https://huggingface.co/new and create a **Model** repo, e.g. `Yuvraj2319/colonai-v2`. Set it to **public** (or keep it private and use a token).
2. Drag-and-drop `outputs/unified_multimodal_v2/checkpoints/best_model.pth` into the repo's **Files** tab. It uploads via the browser; the 600 MB will take a few minutes.

**Step 2 — Create the Space:**

1. Go to https://huggingface.co/new-space, name it e.g. `colonai`, pick **Streamlit** as the SDK, choose the **free CPU basic** hardware.
2. Under "Configure your Space", click **"Import from GitHub repo"** and point it at this repo: `Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI`.
3. Open the new Space → **Settings → Variables and secrets** → add:
   - `COLONAI_CHECKPOINT_HF_REPO` = `Yuvraj2319/colonai-v2` (or whatever you named it in step 1)
   - `COLONAI_CHECKPOINT_HF_FILE` = `best_model.pth`
4. The Space will rebuild automatically. First start takes ~3–4 minutes (downloading the checkpoint); after that it's instant.

**Step 3 — Update the demo link:**

Once your Space is live at `https://huggingface.co/spaces/Yuvraj2319/colonai`, edit the **"🚀 Try it now"** section at the top of this README and replace the placeholder line with that URL.

> **Why a separate model repo?** Hugging Face Spaces have a 50 GB hard limit and the free CPU tier reboots from a fresh container on every push. Storing the 600 MB checkpoint in a Model repo (which `huggingface_hub` caches on disk and fetches once) is the standard pattern and keeps the Space repo small.

### Option 2 — Streamlit Community Cloud (also free)

1. Go to https://share.streamlit.io, sign in with GitHub.
2. Pick this repo, set the entrypoint to `app.py`.
3. Add `COLONAI_CHECKPOINT_HF_REPO` and `COLONAI_CHECKPOINT_HF_FILE` as app secrets (same values as above).
4. Community Cloud reads `requirements.txt` automatically.

### Option 3 — Demo mode only (no checkpoint)

If you just want to show the UI without real predictions, skip the checkpoint setup entirely. The app boots into demo mode — the safety policy still works, the layout is identical, but the "Analyse" button shows a placeholder result.

### Option 4 — Self-host

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Run this behind a reverse proxy (nginx / Caddy) that terminates TLS. Read [`SECURITY.md`](./SECURITY.md) for the production checklist (API key, rate limiting, log rotation).

---

## 📂 What's in the repo

| Path | What it is |
| :-- | :-- |
| `app.py` | The Streamlit web application |
| `Dockerfile` | Docker image for HF Spaces / self-host |
| `scripts/serve_api.py` | The FastAPI REST service |
| `scripts/scrape_cancer_news.py` | Daily cancer-news scraper (6 public RSS feeds) |
| `scripts/auto_bug_check.py` | Static analysis + auto-cleanup + HTTP probe |
| `scripts/check_live_status.py` | Playwright probe — verifies the live URL has the real model loaded (not demo mode) |
| `src/app/` | Application helpers (safety policy, accessibility UI, cross-checks, security, segmentation, patient-ui, cancer-news rendering) |
| `src/agents/` | The 6 specialised AI agents and their orchestrator |
| `src/models/` | The model architecture |
| `src/data/` | Dataset loaders + tabular preprocessing |
| `.github/workflows/` | Daily auto-bug-check + cancer-news scraper (GitHub Actions) |
| `outputs/cancer_news.json` | Latest scraped news (refreshed daily by the cron) |
| `outputs/auto_bug_report.json` | Most recent bug-check status |
| `outputs/live_status.json` | Most recent Playwright probe of the live Space |
| `data/`, `outputs/` (most) | Gitignored — datasets and model artefacts |
| `RUN_ME.md` | Non-technical user instructions |
| `SECURITY.md` | Threat model and operator runbook |
| `ColonAI_Pitch.pdf` | Buyer-ready 7-page pitch deck |
| `ColonAI_Handover.pdf` | Full technical handover for future maintainers |


The training scripts, internal architecture details, exact loss functions, and dataset processing pipelines are intentionally kept inside the code rather than spelled out here. Read the paper for the academic context.

---

## 📚 Datasets used

ColonAI is trained on publicly available, research-licensed datasets only:

- **HyperKvasir** — University of Tromsø (Norway). 23-class gastrointestinal imagery.
- **CVC-ClinicDB / CVC-ColonDB / CVC-300** — Hospital Clinic de Barcelona. Polyp images with pixel-level masks.
- **Kvasir-SEG** — Polyp segmentation extension of Kvasir.
- **ETIS-LaribPolypDB** — Pentax-scope polyps. Used as the unseen-brand hold-out.
- **The Cancer Genome Atlas (TCGA)** clinical metadata for tabular patient features.

No patient-identifiable data is used. No private hospital data is used.

---

## 📝 License & citation

This project is released under the MIT license. You are welcome to read, run, and learn from it. If you use it in academic work, please cite the accompanying research paper (see [`research_paper.tex`](./research_paper.tex)).

If you spot a security issue, please report it privately rather than filing a public GitHub issue — see [`SECURITY.md`](./SECURITY.md).

---

## 🙏 Acknowledgements

ColonAI was built as a master's-degree research project at Amity University. It stands on the work of:

- The open-source endoscopy-AI community (HyperKvasir, CVC, Kvasir-SEG, ETIS-Larib teams)
- The PyTorch and Streamlit teams
- The clinical-NLP open-research community
- The clinicians who pushed for explainable, calibrated, safety-aware medical AI long before it was fashionable

> **A reminder.** This software exists to **help** clinicians, not replace them. The single most important number on the screen is not the confidence percentage — it's the line at the bottom that says *"Always confirm with a clinician."* Read it. Mean it.
