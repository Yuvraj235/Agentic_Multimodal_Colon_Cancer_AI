# Agentic Multimodal Colon Cancer AI

An end-to-end, research-grade system for early detection and staging of colorectal conditions using a 6-agent AI pipeline. It combines medical images (colonoscopy, histopathology, endoscopy), clinical text, and patient tabular data through a cross-modal transformer to produce explainable, clinician-facing diagnoses — all surfaced through an interactive Streamlit web application.

> **Research & Educational Use Only — Not a Medical Device.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [6-Agent Pipeline](#6-agent-pipeline)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Web App](#running-the-web-app)
- [Running the Training Pipeline](#running-the-training-pipeline)
- [Web App Features](#web-app-features)
- [Experiments](#experiments)
- [Configuration](#configuration)

---

## Overview

ColonAI screens for five colorectal conditions:

| Class | Description |
|---|---|
| Colorectal Polyps | Growths on the colon lining — precancerous risk |
| Ulcerative Colitis (Mild) | Inflammatory bowel disease, mild presentation |
| Ulcerative Colitis (Moderate–Severe) | Advanced IBD requiring urgent intervention |
| Barrett's Esophagus | Pre-cancerous oesophageal condition |
| Post-Therapeutic Site | Tissue following prior treatment |

The system accepts three input modalities:
- **Images** — colonoscopy frames, histopathology slides, endoscopy images
- **Clinical text** — symptom descriptions, medical notes (processed by BioBERT)
- **Tabular data** — 12 TCGA patient features (age, BMI, smoking, staging, etc.)

---

## Architecture

### Core Model — `src/models/unified_transformer.py`

```
Image branch
  ResNet50 (ImageNet pretrained, layer4 = GradCAM target)
  + EfficientNet-B4 (ImageNet pretrained, blocks[-2] = GradCAM target)
  → spatial feature maps → patch tokens → projected to d_model=256
  → dual backbone fused via learned gating per spatial position

Text branch
  BioBERT (dmis-lab/biobert-base-cased-v1.2)
  → CLS token → linear projection to d_model=256

Tabular branch
  TabTransformer (12 TCGA features, per-feature token)
  → projection to d_model=256

Fusion — 3-stage Gated Cross-Modal Transformer
  Stage A: per-modality self-attention
  Stage B: iterative bidirectional cross-attention (3 layers, 8 heads)
  Stage C: shared bottleneck self-attention + CLS pooling
  + Learned modality gate (sigmoid) for dynamic weighting

Output heads
  (a) Pathology      — 5-class GI subtype classification
  (b) Cancer staging — 4-class (No Cancer / Stage I / II / III-IV)
  (c) Cancer risk    — binary (benign / malignant)
```

**Total parameters:** ~74.7 M trainable

### Anti-Overfitting Configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 8e-5 |
| BioBERT learning rate | 1e-5 |
| Weight decay | 0.15 |
| Head dropout | 0.60 |
| Fusion dropout | 0.40 |
| Tabular dropout | 0.40 |
| Image dropout | 0.30 |
| Mixup alpha | 0.40 |
| Label smoothing | 0.15 |
| BERT frozen layers | 10 (unfrozen at epoch 8) |
| Early stopping patience | 12 epochs |
| Tabular Gaussian noise (σ) | 0.05 |

---

## 6-Agent Pipeline

Each inference pass runs six specialised agents orchestrated by `MultiModalOrchestrator`:

| # | Agent | File | What it does |
|---|---|---|---|
| 1 | Image Agent | `src/agents/unified_image_agent.py` | GradCAM++ saliency on dual ConvNeXt/ResNet backbone; returns heatmap overlay |
| 2 | Text Agent | `src/agents/text_agent.py` | BioBERT attention rollout on clinical symptom text |
| 3 | Tabular Risk Agent | `src/agents/tabular_risk_agent.py` | SHAP-style perturbation importance for 12 TCGA features |
| 4 | Fusion Reasoning Agent | `src/agents/fusion_reasoning_agent.py` | Full forward pass; computes per-modality contribution weights |
| 5 | XAI Agent | `src/agents/xai_agent.py` | MC-Dropout uncertainty quantification + counterfactual generation |
| 6 | Clinical Recommendation Agent | `src/agents/clinical_recommendation_agent.py` | BSG/NICE guideline-aligned referral & surveillance advice |
| — | Orchestrator | `src/agents/multimodal_orchestrator.py` | Coordinates all 6 agents; returns `MultiModalDiagnosticOutput` |

### Output dataclasses

```python
FusionDiagnosis:
    pathology_class       # str  — predicted condition label
    pathology_probs       # dict — per-class probabilities
    cancer_stage          # str  — No Cancer / Stage I / II / III-IV
    cancer_risk_score     # float 0–1
    image_weight          # float — modality contribution
    text_weight           # float
    tabular_weight        # float
    overall_confidence    # float 0–1

XAIReport:
    uncertainty           # float — MC-Dropout epistemic uncertainty
    gradcam_heatmap       # np.ndarray — raw heatmap
    gradcam_overlay       # np.ndarray — overlay on original image

ClinicalRecommendation:
    urgency               # str — Routine / Soon / Urgent / Emergency
    primary_action        # str
    surveillance          # str — follow-up interval
    referrals             # List[str]
    investigations        # List[str]
    lifestyle_advice      # List[str]
```

---

## Dataset

| Dataset | Source | Size | Use |
|---|---|---|---|
| HyperKvasir | `data/processed/hyper_kvasir_clean/` | 10,662 images, 4 classes | Primary classification training |
| CVC-ClinicDB | `data/raw/CVC-ClinicDB/PNG/` | 612 polyp images + masks | Segmentation training |
| TCGA Clinical | `data/raw/tcga/clinical/clinical.tsv` | 12 tabular features | Tabular branch training |

**Combined dataset split:**

| Split | Images |
|---|---|
| Train | 8,609 |
| Validation | 2,211 |
| Test | 1,066 |

### TCGA Tabular Features (12)

`age_at_index`, `bmi`, `cigarettes_per_day`, `pack_years_smoked`, `alcohol_history`, `gender`, `race_encoded`, `tumor_stage_encoded`, `morphology_encoded`, `site_of_resection_encoded`, `year_of_diagnosis`, `days_to_last_follow_up`

---

## Results

| Metric | Value |
|---|---|
| Best epoch | 7 |
| Best validation F1 | 0.9989 |
| Test Accuracy | **99.53 %** |
| Test F1 Macro | **0.9946** |
| Test AUC-ROC | **1.0000** |

Checkpoint saved to: `outputs/unified_multimodal/checkpoints/best_model.pth`
Figures (18 PNGs): `outputs/unified_multimodal/figures/`

---

## Project Structure

```
.
├── app.py                          # Streamlit web application (entry point)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Light theme, port 8501
├── .claude/
│   └── launch.json                 # Dev server config
│
├── src/
│   ├── agents/
│   │   ├── unified_image_agent.py       # Agent 1 — GradCAM++
│   │   ├── text_agent.py                # Agent 2 — BioBERT attention
│   │   ├── tabular_risk_agent.py        # Agent 3 — SHAP perturbation
│   │   ├── fusion_reasoning_agent.py    # Agent 4 — Modality fusion
│   │   ├── xai_agent.py                 # Agent 5 — Uncertainty + counterfactuals
│   │   ├── clinical_recommendation_agent.py  # Agent 6 — BSG/NICE guidelines
│   │   └── multimodal_orchestrator.py   # Coordinates all 6 agents
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   └── report_generator.py          # ReportLab PDF generator
│   │
│   ├── data/
│   │   ├── multimodal_dataset.py        # Main dataset + make_clinical_text()
│   │   ├── dataset.py
│   │   ├── cvc_seg_dataset.py
│   │   └── hyperkvasir_artifact_dataset.py
│   │
│   ├── losses/
│   │   ├── dice_loss.py
│   │   ├── masked_dice.py
│   │   └── multitask_loss.py
│   │
│   ├── models/
│   │   ├── unified_transformer.py       # Main UnifiedMultiModalTransformer
│   │   ├── convnext_unet.py             # Segmentation model
│   │   ├── multitask_convnext.py
│   │   ├── convnext_multihead.py
│   │   ├── projection_head.py
│   │   ├── artifact_multitask_convnext.py
│   │   ├── backbones/
│   │   │   ├── convnext.py
│   │   │   ├── efficientnet.py
│   │   │   └── swin.py
│   │   └── fusion/
│   │       ├── attention_fusion.py
│   │       └── fusion_model.py
│   │
│   └── utils/
│       └── artifact_mask.py
│
└── experiments/
    ├── run_full_pipeline.py         # Master training script (start here)
    ├── train_unified_multimodal.py
    ├── evaluate_unified_multimodal.py
    ├── day8b_train_multimodal.py
    ├── day9b_segmentation_train.py
    ├── day9c_multitask_train.py
    ├── day9d_evaluate_multitask.py
    ├── day9e_combined_clinical_signal.py
    ├── day10c_train_convnext_unet_strong.py
    ├── day10d_seg_guided_classification.py
    ├── day10e_evaluate_seg_guided.py
    ├── day11_clinical_explainability.py
    ├── day11_severity_module.py
    ├── day12_dataset_audit.py
    ├── gradcam_protopnet_clinical.py
    ├── pipeline_diagram.py
    ├── architecture_flow.py
    └── ... (18 scripts total)
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback supported)
- ~8 GB VRAM for full model training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Yuvraj235/Agentic_Multimodal_Colon_Cancer_AI.git
cd Agentic_Multimodal_Colon_Cancer_AI

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download datasets
#    - HyperKvasir: https://datasets.simula.no/hyper-kvasir/
#    - CVC-ClinicDB: https://polyp.grand-challenge.org/CVCClinicDB/
#    - TCGA clinical data: https://portal.gdc.cancer.gov/
#    Place them under data/raw/ and data/processed/ as described above.
```

---

## Running the Web App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

> If the trained checkpoint is not present at
> `outputs/unified_multimodal/checkpoints/best_model.pth`,
> the app automatically falls back to **Demo Mode** with realistic
> synthetic outputs so all UI features remain usable.

---

## Running the Training Pipeline

```bash
# Full end-to-end training (recommended — trains model, saves checkpoint, generates all 18 figures)
python3 experiments/run_full_pipeline.py

# Train only the unified multimodal model
python3 experiments/train_unified_multimodal.py

# Evaluate a saved checkpoint
python3 experiments/evaluate_unified_multimodal.py
```

> Always use `python3`, not `python`, to avoid encoding errors with special characters.

---

## Web App Features

The Streamlit app (`app.py`) is a **6-step wizard**:

### Step 1 — Patient Information
- Full name, age, gender
- Height/weight with automatic BMI calculation and category label
- City (used for doctor search)
- Medical history: prior colonoscopy, family history, smoking status, alcohol use, diabetes

### Step 2 — Symptoms & Upload
- 18-symptom checklist (rectal bleeding, bowel habit changes, pain, weight loss, fatigue, etc.)
- Free-text symptom description (fed to BioBERT)
- Image upload (colonoscopy / histopathology / endoscopy) — JPG, PNG, TIFF up to 50 MB
- Medical report upload (PDF/TXT — used as supplementary clinical text)

### Step 3 — AI Analysis
- Animated 6-agent pipeline progress display
- Runs `MultiModalOrchestrator.run()` in real time
- Falls back to demo output if no checkpoint or no image

### Step 4 — Results Dashboard
Four tabs:
- **Diagnosis** — predicted condition with colour-coded risk badge and urgency banner, confidence gauge (Plotly), probability breakdown bar chart
- **GradCAM View** — original image alongside GradCAM++ attention heatmap overlay (shows what the AI is looking at)
- **Risk Charts** — cancer risk gauge (0–100 %), modality contribution radar chart (image / text / tabular weights), cancer stage pie chart
- **Recommendations** — BSG/NICE-aligned clinical actions, investigations, lifestyle advice, referral list

### Step 5 — Doctor Finder
- 46+ specialist gastroenterologists and oncologists across 20+ cities:
  - India: Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata, Chandigarh, Gurgaon
  - USA: New York, Los Angeles, Chicago, Houston, Boston, San Francisco, Rochester, Baltimore
  - UK: London, Manchester, Birmingham
  - UAE, Singapore, Canada, Australia
- Filter by city or search by name/hospital
- Doctor cards show: name, specialisation, hospital, rating, contact

### Step 6 — Download Report
- Generates a professional clinical PDF using **ReportLab**
- Contents: patient details, symptom summary, AI diagnosis table, probability breakdown, GradCAM images side-by-side, clinical recommendations, doctor shortlist, legal disclaimer
- One-click download button

### Sidebar Features

**Navigation progress bar** — shows current step and completion percentage

**Site Guide button** — opens a full reference page with four tabs:
- Overview: what ColonAI is, performance metrics, who it's for
- Step-by-Step: expandable walkthrough of all 6 steps with tips
- AI Explained: model architecture, output heads, result interpretation table
- FAQ: 10 common questions answered

**AI Assistant chatbot** — collapsible expander covering 23 topics:
- How to navigate the app
- Symptoms and when to seek help
- Condition explanations (polyps, UC, Barrett's, colon cancer)
- AI model details (GradCAM, BioBERT, uncertainty, accuracy)
- Report and doctor finder help
- Medical disclaimers

---

## Experiments

| Script | Purpose |
|---|---|
| `run_full_pipeline.py` | Master script — trains, evaluates, generates all figures |
| `train_unified_multimodal.py` | Train the UnifiedMultiModalTransformer |
| `evaluate_unified_multimodal.py` | Evaluate checkpoint on test set |
| `day8b_train_multimodal.py` | Early multimodal training experiments |
| `day8b_attention_gradcam.py` | Attention + GradCAM visualisation |
| `day9b_segmentation_train.py` | ConvNeXt-UNet polyp segmentation |
| `day9b_visualize_segmentation.py` | Visualise segmentation masks |
| `day9c_multitask_train.py` | Multitask classification + segmentation |
| `day9d_evaluate_multitask.py` | Evaluate multitask model |
| `day9e_combined_clinical_signal.py` | Clinical signal fusion experiments |
| `day10c_train_convnext_unet_strong.py` | Stronger UNet with heavy augmentation |
| `day10d_seg_guided_classification.py` | Segmentation-guided classification |
| `day10e_evaluate_seg_guided.py` | Evaluate seg-guided model |
| `day11_clinical_explainability.py` | Clinical explainability module |
| `day11_severity_module.py` | Severity scoring module |
| `day12_dataset_audit.py` | Full dataset audit and quality checks |
| `gradcam_protopnet_clinical.py` | GradCAM + ProtoPNet + clinical overlay |
| `pipeline_diagram.py` | Generate architecture diagram |
| `architecture_flow.py` | Full architecture flow visualisation |

---

## Configuration

### `.streamlit/config.toml`

```toml
[theme]
base                  = "light"
primaryColor          = "#1A73E8"
backgroundColor       = "#F8FAFF"
secondaryBackgroundColor = "#FFFFFF"
textColor             = "#212121"
font                  = "sans serif"

[server]
headless     = true
port         = 8501
maxUploadSize = 50   # MB

[browser]
gatherUsageStats = false
```

### Key paths

| Path | Description |
|---|---|
| `outputs/unified_multimodal/checkpoints/best_model.pth` | Trained model checkpoint |
| `outputs/unified_multimodal/figures/` | 18 generated evaluation figures |
| `data/processed/hyper_kvasir_clean/` | HyperKvasir processed images |
| `data/raw/CVC-ClinicDB/PNG/` | CVC-ClinicDB images and masks |
| `data/raw/tcga/clinical/clinical.tsv` | TCGA clinical tabular data |

---

## Disclaimer

This system is intended for **research and educational purposes only**. It is not a certified medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment. All outputs should be reviewed by a qualified clinician before clinical action is taken.
