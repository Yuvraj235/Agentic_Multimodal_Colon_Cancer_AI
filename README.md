# ColonAI 🩺

**A second pair of eyes for colon-cancer screening.**

Colon cancer is one of the most preventable cancers in the world — *if* the polyp that turns into it is spotted during a routine colonoscopy. In practice, an average of **1 in 4 polyps is missed** during real screening exams, and the miss rate is even higher when the patient's scope is from a different manufacturer than the one the endoscopist trained on.

ColonAI is a research project that tries to close that gap. Upload a colonoscopy image (or stream a live colonoscopy feed) and the system gives you a plain-English answer:

> *"This looks like a growth on the bowel wall (a polyp). I'm fairly confident (87 %), and I'm focused on the upper-left of the image where the lesion appears to be. A doctor should still confirm this."*

— or, equally importantly, it refuses to answer when it's not sure.

> ⚕️ **Research & educational use only — not a medical device.** Every finding must be reviewed by a qualified clinician before any clinical decision is made.

---

## 🚀 Try it now

| Mode | How |
| :-- | :-- |
| **Live web demo** | _Live demo will be linked here once deployed — see the [Deploy](#-deploy-your-own-copy) section below._ |
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

That's a **+136 % improvement on the brand the model had never seen during training**. We also added a separate, dedicated segmentation step that lifts polyp localisation from 0.27 to **0.61** — clinically usable.

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
- **Calibrated confidence** — the percentage shown to the user has been calibrated against held-out data, so it's a meaningful probability rather than a vibes number.
- **REST API** — `/predict`, `/health`, `/version`, `/audit/today` — with optional `X-API-Key` auth and an audit log of every prediction.
- **Security-hardened** — upload size limits, decompression-bomb protection, MIME allow-list, sanitised error responses, CORS allow-list, owner-only audit-log permissions, XSS-safe HTML rendering. See [`SECURITY.md`](./SECURITY.md).

---

## 📈 Honest numbers (cross-vendor held-out test)

We do **not** report a single accuracy number, because that's the metric most easily gamed. Here's the real performance:

| What we measure | On familiar brand | On unfamiliar brand |
| :-- | :--: | :--: |
| Got the diagnosis right? | 95 % | 90 % |
| Heat-map lands on the polyp? | 0.42 (IoU) | 0.16 (IoU) |
| Dedicated segmentation IoU | 0.62 | **0.61** |
| Detection sensitivity (per-polyp, IoU ≥ 0.5) | 0.75–0.92 | 0.38 |
| Confidence is well-calibrated (lower = better) | 0.06 ECE | 0.06 ECE |

We **also** publish what doesn't work:

- The model struggles to grade ulcerative colitis severity (moderate-severe recall is low). It tends to over-call "mild" UC.
- Cross-vendor segmentation is good but cross-vendor detection (per-polyp F1) is still weaker than within-vendor.
- Highly atypical lesions or invasive cancers are **out of distribution** — the safety net catches them and asks for human review.

These limitations are *why* the safety net exists.

---

## ☁️ Deploy your own copy

ColonAI runs as a self-contained Streamlit app. You can host it for free.

### Option 1 — Hugging Face Spaces (recommended, free)

A Spaces config is included (see [`huggingface.yml`](./huggingface.yml)). From the GitHub repo:

1. Create a new Space at https://huggingface.co/new-space, pick **Streamlit** as the SDK.
2. Connect it to this GitHub repo.
3. Hugging Face will deploy automatically. The free CPU tier is fine for the demo.

When the Space is live, edit this README to replace the placeholder under **🚀 Try it now** with the Space URL.

### Option 2 — Streamlit Community Cloud (also free)

1. Go to https://share.streamlit.io, sign in with GitHub.
2. Pick this repo, set the entrypoint to `app.py`.
3. Add the same dependencies — Community Cloud reads `requirements.txt` automatically.

### Option 3 — Self-host

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Run this behind a reverse proxy (nginx / Caddy) that terminates TLS. Read [`SECURITY.md`](./SECURITY.md) for the production checklist (API key, rate limiting, log rotation).

---

## 📂 What's in the repo

| Path | What it is |
| :-- | :-- |
| `app.py` | The Streamlit web application |
| `scripts/serve_api.py` | The FastAPI REST service |
| `src/app/` | Application helpers (safety policy, accessibility UI, cross-checks, security) |
| `src/agents/` | The 6 specialised AI agents and their orchestrator |
| `src/models/` | The model architecture |
| `data/`, `outputs/` | Data (gitignored) and model checkpoints |
| `RUN_ME.md` | Non-technical user instructions |
| `SECURITY.md` | Threat model and operator runbook |
| `research_paper.tex` | The full research paper |

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
