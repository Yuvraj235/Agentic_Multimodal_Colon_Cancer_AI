# Agentic Multimodal Colon Cancer AI · ColonAI

A research-grade Streamlit application that screens for colorectal conditions by fusing **endoscopy images**, **clinical text**, and **patient history** through a 6-agent multimodal AI pipeline — and surfaces patient-friendly results with three independent layers of clinical safety on top.

> **Research and educational use only — not a medical device.** Every finding must be reviewed by a licensed clinician before any clinical decision.

[Quickstart](#quickstart-30-seconds) · [What's new](#whats-new) · [Architecture](#architecture) · [Safety layers](#three-independent-safety-layers) · [Datasets](#datasets) · [Results](#test-set-results) · [Honest limitations](#honest-limitations) · [License](#license)

---

## Quickstart (30 seconds)

```bash
git clone https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI.git
cd Agentic_Multimodal_Colon_Cancer_AI

# easy launch — installs deps once, opens browser, finds free port
./run_app.command
```

Or, completely manually:

```bash
pip install -r requirements.txt
python3 -m streamlit run app.py --server.port 8501
```

Open **http://localhost:8501**, click **Load Case A · Sigmoid Polyp** on the Patient Info page, then **Analyse →**. Full demo in under 30 seconds. The launcher auto-finds a free port between 8501–8520, so the app opens reliably even when other Streamlit servers are running.

A short instruction sheet for non-technical users lives at [`RUN_ME.md`](./RUN_ME.md).

---

## What's new

This release is a substantial overhaul of the original training-only pipeline. Highlights:

### Two critical bug fixes

1. **Wrong model kwargs** in `load_ai_system` (`n_heads`/`n_layers` → `n_fusion_heads`/`n_fusion_layers`). The TypeError was being swallowed and the app silently fell back to demo mode.
2. **Wrong checkpoint key** (`model_state_dict` → `model_state`). With the old key, the model loaded with random weights and every prediction was effectively garbage.

After the fix the held-out checkpoint loads with **0 missing / 0 unexpected keys**, and a polyp image is correctly classified as polyps with 88 % confidence.

### Three independent safety layers (model + symptom + pixel)

The trained model knows only **5 screening-stage classes** (HyperKvasir + CVC-ClinicDB don't contain advanced cancer). To make the system safe in real-world use we added three layers that catch its blind spots:

1. **NICE NG12 / red-flag clinical-rule overrides** — symptom-driven escalation. Rectal bleeding ≥ 50 yr, iron-deficiency anaemia ≥ 60 yr, weight loss + abdo pain, severe pain, multi-flag combinations all raise the risk score and urgency. Override never lowers anything, only escalates.
2. **Image-statistics atypicality detector** — a pixel-stats engine (deep-red dominance, dark cavitation, edge disorder, colour disorder) flags images that look like advanced lesions, independent of the trained model's class palette.
3. **Honest staging override** — when atypicality is high or post-override risk ≥ 60 %, the staging head's output is hidden and replaced with "Cannot stage from one image" so the app never shows a misleadingly low stage on a cancer image.

### Smart, real, authentic doctor finder

- Live **OpenStreetMap Nominatim** geocoding (no API key) — works for any city worldwide.
- Live **OSM Overpass API** lookup of nearby hospitals / clinics within 8 km.
- Embedded **Google Maps iframe** (keyless `?output=embed`) centred on the typed city.
- 25 verified Delhi-NCR specialists added to the curated DB (AIIMS, Sir Ganga Ram, Apollo, Max, BLK-Max, Medanta, Fortis FMRI, Manipal, Jaypee, Asian, Yashoda).
- AI-pathology-aware ranking with "Why recommended" reason chips, distance-weighted sort, and per-card **Open in Maps** / **Get Directions** button chips.

### Patient-friendly UI

- Plain-English **"Why this result"** card on the Diagnosis tab built from the patient's actual inputs (no MC-Dropout / BioBERT / ResNet jargon — those moved to a collapsed "Show technical details" expander).
- **"What to do next"** actionable steps card on the Recommendations tab, differentiated by urgency band (Routine / Urgent / Emergency).
- Animated KPI counters, CSS particle background, agent-timeline with a moving progress bead, 3-D Plotly colon viewer, dark-mode toggle, contextual FAQ bubbles with URL dismissal.
- **Compare-mode** — pin two analyses side-by-side (slot A vs slot B).
- Motivational **ribbon-of-hope** card replacing the model-architecture pills; the copy adapts to the patient's risk band.
- Three **quick-demo** cases on Step 1 (Polyp / UC / Barrett's) for one-click presentations.

### Doctor-grade chatbot

- Score-based keyword matching with stop-word filtering and phrase bonus (replaces the brittle longest-keyword selection).
- KB expanded ~25 entries: diet, prevention, treatment, recovery, pain/sedation, FIT, NICE NG12 red flags, BSG/USPSTF screening, Lynch/FAP, anxiety, second opinion, insurance, wait times.
- Verified **27/27** patient-style questions route correctly.
- Compact at the bottom of the sidebar (collapsed by default).

### Honest PDF report and presentation script

- The downloadable PDF now includes the **Image-features verdict** and **Clinical Safety Override Applied** sections plus all original content.
- `scripts/build_presentation_pdf.py` builds `outputs/ColonAI_Presentation_Script.pdf` — a click-by-click 6–8 minute live-demo script with talking points, a technical slide, an honesty slide, 8 likely Q&A, and three case studies.

---

## Architecture

```
                     ┌───────────────────────────────────┐
   endoscopy image ──▶│ Image branch:                    │
                     │   ResNet-50 + EfficientNet-B0    │
                     │   (dual backbone, GradCAM target) │
                     └────────────────┬──────────────────┘
                                      │
   clinical text ─────▶ ┌─────────────┐│
                        │ BioBERT     ││
                        │ (PubMed-pre)││
                        └────┬────────┘│
                             │         │
   patient features ─▶ ┌─────┴─┐       │
                       │ TabTr │       │
                       │former │       │
                       └───┬───┘       │
                           │           │
                           ▼           ▼
                  ┌─────────────────────────────────────┐
                  │ Gated Cross-Modal Fusion Transformer │
                  │   - 3 cross-attention layers         │
                  │   - 2 self-attention layers          │
                  │   - sigmoid modality-gate (256-d)    │
                  └────────┬────────────────────────────┘
                           │
                           ▼
                  ┌────────────────────────┐
                  │ 3 task heads           │
                  │  · pathology (5-class) │
                  │  · staging   (4-class) │
                  │  · risk      (binary)  │
                  └────────────────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │ 6-agent post-processing   │
              │  · UnifiedImageAgent      │
              │  · TextAgent              │
              │  · TabularRiskAgent       │
              │  · FusionReasoningAgent   │
              │  · XAIAgent               │
              │  · ClinicalRecommendation │
              └───────────────────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │ THREE SAFETY LAYERS       │
              │  · NICE NG12 rule engine  │
              │  · Image-stats atypicality│
              │  · Honest staging override│
              └───────────────────────────┘
                           │
                           ▼
                       Streamlit UI
```

### 6-agent pipeline

| Agent | What it does |
|---|---|
| **UnifiedImageAgent** | Forward pass through the dual backbone, GradCAM++ heatmap on ResNet layer4[-1] |
| **TextAgent** | BioBERT attention rollout over the clinical-text input |
| **TabularRiskAgent** | SHAP-style perturbation importance over the 12 patient-history features |
| **FusionReasoningAgent** | Combines the three modality embeddings, runs the gated cross-modal transformer, and produces the final softmax over pathology / staging / risk |
| **XAIAgent** | MC-Dropout (15 stochastic passes) for predictive entropy and overlay-ready GradCAM artefacts |
| **ClinicalRecommendationAgent** | Maps the predicted class + confidence + risk to BSG / NICE / USPSTF-aligned next-step recommendations |

The orchestrator (`src/agents/multimodal_orchestrator.py`) runs the agents end-to-end on each user case.

---

## Three independent safety layers

### 1 · NICE NG12 / red-flag rule engine

Source: `apply_clinical_overrides` in [`app.py`](./app.py).

Triggers (every rule is additive, never subtracts):

| Rule | Risk boost | Urgency floor |
|---|---|---|
| Rectal bleeding ≥ 50 yr | +0.45 | Urgent |
| Iron-deficiency anaemia ≥ 60 yr | +0.45 | Urgent |
| Weight loss + abdo pain ≥ 40 yr | +0.40 | Urgent |
| Change in bowel habit ≥ 60 yr | +0.35 | Urgent |
| Severe pain (≥ 9/10) | +0.30 | Urgent |
| Multiple red flags + severe pain ≥ 40 yr | +0.40 | **Emergency** |
| First-degree family history of CRC | +0.15 | Elective |
| Previous polyps | +0.10 | Elective |
| Image atypicality (advanced-lesion features) | +0.40 | Urgent |

For a 60-year-old with rectal bleeding, weight loss, anaemia, change in bowel habit, severe pain, family history and prior polyps, the engine fires **11 rules** and converts: Risk 12 % → 99 %, Urgency Elective → Emergency.

### 2 · Image-statistics atypicality detector

Source: [`src/app/image_atypicality.py`](./src/app/image_atypicality.py).

Six per-image signals computed from the raw pixels:

| Signal | What it detects |
|---|---|
| `red_necrosis` | Deep-red, low-green, low-blue patches → bleeding / fungating / necrotic tissue |
| `dark_cavity` | Excessive dark area beyond normal lumen → cavitation, mass shadow |
| `edge_disorder` | High-frequency disorganised edges → mass effect, ulceration |
| `colour_disorder` | Hue scattered across the frame → mucosal disruption |
| `mucosal_uniformity` | Pink-hue cluster coherence → healthy mucosa |
| `brightness_balance` | Centred bright mucosa with small dark lumen → diagnostic frame |

Three verdicts:

| Verdict | Trigger |
|---|---|
| `consistent_screening` | Pixels look like a screening-stage finding (most HyperKvasir samples) |
| `atypical_concerning` | `atypicality ≥ 0.55` AND `red_necrosis ≥ 0.45` — pixel signs of advanced lesion |
| `uncertain` | Mixed signals |

Calibration verified against real samples (HyperKvasir polyps, BBPS-clean colon, UC grades 1–3, Barrett's, CVC polyps) and a synthetic advanced-cancer mockup.

### 3 · Honest staging override

The staging head was trained on **class-derived synthetic labels** (HyperKvasir has no real TNM ground truth — the staging signal was inferred from the pathology class). On a stage-IV cancer image the head will return "No Cancer 89 %" because it has never seen that signal.

The override fires when:

- `image_atypicality.verdict == "atypical_concerning"`, **or**
- post-override `risk_score ≥ 0.60`

In that case `analysis["stage"]` becomes "Cannot stage from one image" and `stage_probs` is replaced with `{"Cannot determine": 1.0}` so the chart can never claim a misleadingly low stage. The app shows a clear honest message:

> *Cancer staging not shown. Single endoscopy images cannot reliably stage cancer. Real staging requires histology (biopsy) + cross-sectional imaging (CT / MRI). The staging head's output for this image is not reliable — we are hiding it rather than showing a false low-stage number.*

---

## Datasets

| Dataset | Source | Used for | Notes |
|---|---|---|---|
| **HyperKvasir** | [Borgli et al. 2020](https://datasets.simula.no/hyper-kvasir/) | Image classification (5-class) | 10,662 images. Norwegian endoscopy unit. **Screening-stage only** — no advanced cancer. |
| **CVC-ClinicDB** | [Bernal et al. 2015](https://polyp.grand-challenge.org/CVCClinicDB/) | Image pretraining + segmentation | 612 polyp images + masks. Spanish hospital. Polyps only. |
| **TCGA-COAD/READ clinical** | [TCGA via GDC](https://portal.gdc.cancer.gov/) | Tabular feature pool (12 features) | Age, BMI, smoking, alcohol, family history, year of diagnosis, etc. |

The model has **no advanced-cancer training data** — that is *the* fundamental limitation and the reason for the three safety layers above.

---

## Test-set results

Held-out test split (1,066 images stratified across the 5 classes):

| Metric | Value |
|---|---|
| Top-1 accuracy | **90.3 %** |
| Macro-F1 | **0.81** |
| AUC-ROC (one-vs-rest mean) | **0.984** |
| Best epoch | **7** of 60 (no overfitting) |

Per-class confusion matrix and ROC curves are in `outputs/unified_multimodal/figures/`.

---

## Project structure

```
.
├── app.py                                  # Streamlit web app (entry point)
├── run_app.command                         # macOS double-click launcher
├── run_app.sh                              # symlink → run_app.command
├── RUN_ME.md                               # plain-English start instructions
├── requirements.txt
├── README.md                               # this file
│
├── assets/
│   └── demo_cases/                         # 3 sample images (polyp, UC, Barrett's)
│
├── src/
│   ├── agents/                             # 6-agent pipeline implementations
│   │   ├── multimodal_orchestrator.py
│   │   ├── unified_image_agent.py
│   │   ├── text_agent.py
│   │   ├── tabular_risk_agent.py
│   │   ├── fusion_reasoning_agent.py
│   │   ├── xai_agent.py
│   │   └── clinical_recommendation_agent.py
│   ├── models/
│   │   └── unified_transformer.py          # UnifiedMultiModalTransformer
│   ├── data/
│   │   └── multimodal_dataset.py           # PyTorch Dataset + transforms
│   ├── losses/
│   │   └── multitask_loss.py
│   └── app/                                # Streamlit-side helpers (NEW)
│       ├── geo.py                          # Nominatim + Overpass + Maps URLs
│       ├── image_atypicality.py            # pixel-stats safety layer
│       ├── ui_extras.py                    # particles, counters, lottie, dark-mode
│       └── report_generator.py             # PDF report
│
├── experiments/                            # training / evaluation scripts
│   ├── train_unified_multimodal.py
│   ├── evaluate_unified_multimodal.py
│   └── run_full_pipeline.py
│
├── scripts/
│   └── build_presentation_pdf.py           # generates the demo script PDF
│
├── outputs/                                # gitignored — checkpoints, metrics, figs
│   └── unified_multimodal/
│       ├── checkpoints/best_model.pth
│       ├── figures/
│       └── metrics.json
│
└── data/                                   # gitignored — raw + processed datasets
    ├── raw/CVC-ClinicDB/
    ├── raw/tcga/clinical/
    └── processed/hyper_kvasir_clean/
```

---

## Installation

```bash
git clone https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI.git
cd Agentic_Multimodal_Colon_Cancer_AI

# (optional but recommended)
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

The launcher (`run_app.command`) auto-installs dependencies on first run if any are missing, so you can skip the manual `pip install` if you don't mind it happening once at start-up.

### Required runtime data

The trained checkpoint (`outputs/unified_multimodal/checkpoints/best_model.pth`) and the HyperKvasir / CVC-ClinicDB / TCGA datasets are gitignored — they total several GB. Use the training pipeline below to reproduce them, or copy them into `outputs/` and `data/` from your own working tree.

---

## Running the web app

### One-click (macOS)

Double-click **`run_app.command`** in the project folder. A Terminal opens, dependencies install (first run only), the server starts on the first free port between 8501 and 8520, and your default browser opens automatically. Press Ctrl-C in the Terminal to stop.

### Terminal

```bash
./run_app.command
```

### Pure manual fallback

```bash
python3 -m streamlit run app.py --server.port 8501
```

### Smoke test

1. Open the app.
2. On Step 1, click **Load Case A · Sigmoid Polyp** in the Quick-demo panel.
3. Wait ~25 s for the model to warm up, then click **Analyse →**.
4. On Results, you should see:
   - Hero "Your AI Health Report"
   - Image-features check card (blue) — *"Pixel features look like a screening-stage finding"*
   - Key-finding strip — *Colorectal Polyps · ~75 % confidence · Low risk*
   - 4 animated KPI counters
   - Tabs: Diagnosis · Why this result? · GradCAM View · Risk Charts · Recommendations
5. Click **Find Doctors → Generate Report → Generate PDF Report** to verify the full flow including the new override / image-features sections in the PDF.

---

## Running the training pipeline

```bash
python3 experiments/run_full_pipeline.py
```

The script does CVC-ClinicDB pretraining, then HyperKvasir + TCGA fine-tuning, then test-set evaluation, then writes all 18 evaluation figures to `outputs/unified_multimodal/figures/`. Expect ~3 hours on a single GPU.

The full anti-overfitting config that produced the published best epoch:

```python
lr=4e-5, bert_lr=6e-6, weight_decay=0.15
head_drop=0.45, fusion_drop=0.4, tab_drop=0.4, img_drop=0.35
mixup_alpha=0.3, label_smoothing=0.10
freeze_bert_layers=10, unfreeze_epoch=3, early_stop=18
```

---

## Honest limitations

If you take only one section of this README away, take this one.

- **No advanced-cancer training data.** HyperKvasir and CVC-ClinicDB are screening-stage datasets. The model has 5 output classes — none of them is stage III or IV. A stage-IV cancer image will be misclassified as one of the 5 known classes; this is detected and surfaced by the image-statistics atypicality layer, but the underlying classifier output is not trustworthy on advanced disease.
- **Staging is approximate.** The staging head was trained on class-derived synthetic labels, not real TNM ground truth. The honest-staging override hides its output on atypical or high-risk cases.
- **No external validation.** All reported metrics are on a held-out split of the same datasets the model was trained on. Performance on Asian / African / paediatric cohorts has not been measured.
- **Calibration not yet temperature-scaled.** Predictive probabilities should not be interpreted as exact frequencies; reliability diagrams are in the figures folder.
- **Doctor directory is illustrative.** Names, phone numbers and ratings are sourced from public hospital websites and may have changed. The OSM live layer is community-sourced. Verify any contact before booking.
- **Research / educational use only.** Not a medical device. No regulatory clearance (FDA / MHRA / CE). All findings must be reviewed by a licensed clinician before any clinical decision.

---

## Citation

If you use this codebase in academic work, please cite:

```bibtex
@misc{singh2026colonai,
  title  = {Agentic Multimodal Colon Cancer AI: A 6-Agent Screening Pipeline
            with Cross-Modal Fusion and Clinical-Rule Safety Layers},
  author = {Singh, Yuvraj Pratap and contributors},
  year   = {2026},
  url    = {https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI}
}
```

Plus the original dataset papers:

- HyperKvasir — Borgli et al., *Scientific Data*, 2020.
- CVC-ClinicDB — Bernal et al., *Computerized Medical Imaging and Graphics*, 2015.
- BioBERT — Lee et al., *Bioinformatics*, 2020.

---

## License

MIT — see [`LICENSE`](./LICENSE) (when present). The trained checkpoint inherits the licences of the source datasets (HyperKvasir CC-BY 4.0, CVC-ClinicDB educational use, TCGA Open Data).

---

## Acknowledgements

- The **HyperKvasir** team at Simula Research Laboratory for the foundational dataset.
- The **CVC-ClinicDB** team at the Computer Vision Center, Universitat Autònoma de Barcelona.
- The **TCGA** consortium for the clinical metadata used in the tabular branch.
- **NICE / BSG / USPSTF** for the published clinical-pathway guidance baked into the recommendation agent and the safety-rule engine.
- **OpenStreetMap** contributors for the live nearby-clinic data layer (Nominatim + Overpass).
