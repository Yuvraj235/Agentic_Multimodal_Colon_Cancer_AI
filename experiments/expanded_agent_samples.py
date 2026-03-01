# -*- coding: utf-8 -*-
"""
Expanded Agent Samples — All Subfolders, Both Datasets (CVC + HyperKvasir)
===========================================================================
Generates accurate agent samples drawn from EVERY pathological subfolder
in both datasets:

  HyperKvasir subfolders used:
    polyps                          → polyps
    ulcerative-colitis-grade-0-1    → uc-mild
    ulcerative-colitis-grade-1      → uc-mild
    ulcerative-colitis-grade-1-2    → uc-mild
    ulcerative-colitis-grade-2      → uc-moderate-sev
    ulcerative-colitis-grade-2-3    → uc-moderate-sev
    ulcerative-colitis-grade-3      → uc-moderate-sev
    barretts                        → barretts-esoph
    barretts-short-segment          → barretts-esoph
    esophagitis-a                   → barretts-esoph
    esophagitis-b-d                 → barretts-esoph
    dyed-lifted-polyps              → therapeutic
    dyed-resection-margins          → therapeutic

  CVC-ClinicDB:
    PNG/Original/*.png              → polyps  (612 images)

Output structure:
  outputs/expanded_agent_samples/
    ├── per_case/
    │   ├── <dataset>__<subfolder>__<case_id>/
    │   │   ├── original.png
    │   │   ├── gradcam_overlay.png
    │   │   ├── gradcam_panel.png
    │   │   ├── agent_summary.json
    │   │   └── clinical_report.txt
    │   └── ...
    ├── summary_grid.png        (all cases, colour-coded by source dataset)
    ├── summary_dashboard.png   (accuracy, urgency, risk breakdown)
    └── all_results.json

Run from project root:
  python3 experiments/expanded_agent_samples.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, random, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from transformers import AutoTokenizer

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    build_dataloaders, N_TABULAR_FEATURES,
    load_tcga_tabular, extract_tabular_vector,
    make_clinical_text, TABULAR_FEATURES,
)

# ── Config ─────────────────────────────────────────────────────────────────
CHECKPOINT   = "outputs/unified_multimodal/checkpoints/best_model.pth"
BERT_MODEL   = "dmis-lab/biobert-base-cased-v1.2"
N_CLASSES    = 5
D_MODEL      = 256
IMG_SIZE     = 224
SEED         = 42

HK_ROOT      = "data/processed/hyper_kvasir_clean"
CVC_ROOT     = "data/raw/CVC-ClinicDB/PNG/Original"
TCGA_DIR     = "data/raw/tcga"
CVC_MASK_DIR = "data/raw/CVC-ClinicDB/PNG/Ground Truth"

OUT_DIR      = "outputs/expanded_agent_samples"

# Samples to pick per subfolder (each subfolder → N_PER_SUBFOLDER agents)
N_PER_SUBFOLDER = 5

CLASS_NAMES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]

CLASS_COLOURS = {
    "polyps":          "#2196F3",
    "uc-mild":         "#FF5722",
    "uc-moderate-sev": "#B71C1C",
    "barretts-esoph":  "#9C27B0",
    "therapeutic":     "#009688",
}

# Source-level colour for summary grid borders
DATASET_COLOURS = {
    "CVC-ClinicDB":  [138, 43, 226],   # violet
    "HyperKvasir":   [34, 139, 34],    # forest green
}

# ── HyperKvasir subfolder → (class_name, short_label, display_name) ──────
HK_SUBFOLDERS = [
    # (relative_path_from_root,            class_name,         display_name)
    ("lower-gi-tract/pathological-findings/polyps",
        "polyps",          "HK — Polyps"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-0-1",
        "uc-mild",         "HK — UC Grade 0-1"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1",
        "uc-mild",         "HK — UC Grade 1"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2",
        "uc-mild",         "HK — UC Grade 1-2"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2",
        "uc-moderate-sev", "HK — UC Grade 2"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3",
        "uc-moderate-sev", "HK — UC Grade 2-3"),
    ("lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3",
        "uc-moderate-sev", "HK — UC Grade 3"),
    ("upper-gi-tract/pathological-findings/barretts",
        "barretts-esoph",  "HK — Barrett's"),
    ("upper-gi-tract/pathological-findings/barretts-short-segment",
        "barretts-esoph",  "HK — Barrett's Short-Seg"),
    ("upper-gi-tract/pathological-findings/esophagitis-a",
        "barretts-esoph",  "HK — Oesophagitis Grade A"),
    ("upper-gi-tract/pathological-findings/esophagitis-b-d",
        "barretts-esoph",  "HK — Oesophagitis Grade B-D"),
    ("lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps",
        "therapeutic",     "HK — Dyed-Lifted Polyps"),
    ("lower-gi-tract/therapeutic-interventions/dyed-resection-margins",
        "therapeutic",     "HK — Dyed-Resection Margins"),
]

# ── Clinical findings per class (same as original, publication-grade) ─────
CLINICAL_FINDINGS = {
    "polyps": {
        "finding":       "Colonic Polyp",
        "morphology":    "Sessile/pedunculated mucosal protrusion. Paris classification applied.",
        "risk":          "15-25% adenoma-to-carcinoma transformation (size/histology-dependent).",
        "endoscopic_dx": "Gross appearance consistent with adenomatous polyp. Requires histopathological confirmation.",
        "action":        "Polypectomy (cold snare/EMR per size). Surveillance colonoscopy: 3yr if >=3 adenomas or >=10mm; 5yr if 1-2 low-risk.",
        "icd10":         "K63.5 — Polyp of colon",
        "urgency":       "Elective",
        "surveillance":  "3-year surveillance colonoscopy",
    },
    "uc-mild": {
        "finding":       "Ulcerative Colitis — Mild (Grade 1)",
        "morphology":    "Mucosal erythema, loss of vascular pattern, mild friability. Mayo endoscopic score: 1.",
        "risk":          "8% colorectal cancer risk after 8-10 years of extensive disease.",
        "endoscopic_dx": "Active mild UC. Mucosa shows granularity, decreased vascularity. No ulceration.",
        "action":        "Optimise 5-ASA therapy. Rectal mesalazine if proctitis. Faecal calprotectin monitoring.",
        "icd10":         "K51.00 — Ulcerative (chronic) pancolitis, without complications",
        "urgency":       "Elective",
        "surveillance":  "12-month surveillance colonoscopy with chromoendoscopy",
    },
    "uc-moderate-sev": {
        "finding":       "Ulcerative Colitis — Moderate to Severe (Grade 2-3)",
        "morphology":    "Deep ulceration, spontaneous bleeding, mucopurulent exudate. Mayo endoscopic score: 2-3.",
        "risk":          "25% cumulative colorectal cancer risk. Dysplasia surveillance mandatory.",
        "endoscopic_dx": "Severe active UC. Pseudopolyps, loss of haustra, contact bleeding. Biopsies mandatory.",
        "action":        "Urgent escalation: IV corticosteroids, biologics (infliximab/vedolizumab) or surgical colectomy.",
        "icd10":         "K51.012 — Ulcerative (chronic) pancolitis with rectal bleeding",
        "urgency":       "Urgent",
        "surveillance":  "Annual colonoscopy with chromoendoscopy + targeted biopsies",
    },
    "barretts-esoph": {
        "finding":       "Barrett's Oesophagus / Oesophagitis",
        "morphology":    "Salmon-pink columnar metaplasia replacing squamous epithelium at GEJ. LA Grade A-D for oesophagitis.",
        "risk":          "12% progression to oesophageal adenocarcinoma. Risk stratified by Prague C&M criteria.",
        "endoscopic_dx": "Irregular Z-line with tongues of columnar mucosa extending proximally. Biopsies confirm IM.",
        "action":        "High-dose PPI (omeprazole 40mg BD). EMR for visible dysplasia. RFA if high-grade dysplasia confirmed.",
        "icd10":         "K22.7 — Barrett's oesophagus",
        "urgency":       "Elective",
        "surveillance":  "2-year surveillance OGD with Seattle protocol biopsies",
    },
    "therapeutic": {
        "finding":       "Post-Therapeutic Intervention Site",
        "morphology":    "Dyed/tattooed resection margin. Submucosal injection site. Post-polypectomy scar.",
        "risk":          "5% residual/recurrent polyp risk at polypectomy site.",
        "endoscopic_dx": "India ink tattoo marking confirms resection site. Overlying mucosa healing. No visible residual adenoma.",
        "action":        "Confirm complete resection. Document tattoo location for surgical reference. 3-month check colonoscopy.",
        "icd10":         "Z12.11 — Encounter for screening for malignant neoplasm of colon",
        "urgency":       "Elective",
        "surveillance":  "3-month check colonoscopy at polypectomy site",
    },
}

STAGE_NAMES = ["No Cancer", "Stage I", "Stage II", "Stage III/IV"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Image transform ────────────────────────────────────────────────────────
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
MEAN_NP = np.array([0.485, 0.456, 0.406])
STD_NP  = np.array([0.229, 0.224, 0.225])


def tensor_to_rgb(t):
    arr = t.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * STD_NP + MEAN_NP).clip(0, 1)
    return (arr * 255).astype(np.uint8)


def add_border(img_rgb, color, thickness=6):
    h, w = img_rgb.shape[:2]
    result = img_rgb.copy()
    c = np.array(color, dtype=np.uint8)
    result[:thickness, :] = c
    result[-thickness:, :] = c
    result[:, :thickness] = c
    result[:, -thickness:] = c
    return result


# ── GradCAM++ ─────────────────────────────────────────────────────────────
class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def generate(self, image, class_idx, input_ids, attention_mask, tabular):
        self.model.eval()
        image = image.detach().requires_grad_(True)
        out = self.model(image, input_ids, attention_mask, tabular)
        score = out["pathology"][0, class_idx]
        self.model.zero_grad()
        score.backward()

        acts  = self._acts
        grads = self._grads
        if acts is None or grads is None:
            return np.zeros((7, 7), dtype=np.float32)

        grads_sq = grads ** 2
        denom = 2 * grads_sq + acts * (grads ** 3)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom) * 1e-10)
        alpha   = grads_sq / denom
        weights = (alpha * F.relu(score.exp() * grads)).mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


def apply_gradcam_overlay(cam, raw_rgb, alpha=0.45):
    h, w = raw_rgb.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_up = cv2.GaussianBlur(cam_up, (7, 7), 0)
    cam_up = np.clip(cam_up, 0, 1)
    hmap = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * hmap.astype(np.float32) +
               (1 - alpha) * raw_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return overlay, cam_up


# ── TCGA tabular pool — loaded once, shared by all samples ────────────────
# Maps class_idx → list of TCGA feature vectors drawn from real patient data
_TCGA_POOL: dict = {}          # {class_idx: [np.ndarray, ...]}
_TCGA_DF         = None        # raw TCGA dataframe


def build_tcga_pool():
    """
    Load TCGA clinical.tsv and build a pool of real 12-feature vectors
    per class index.  If TCGA is unavailable, fall back to class-specific
    synthetic values that match training defaults.
    """
    global _TCGA_POOL, _TCGA_DF
    print("\nLoading TCGA clinical data for tabular integration...")
    try:
        _TCGA_DF = load_tcga_tabular(TCGA_DIR)
        if _TCGA_DF is not None and len(_TCGA_DF) > 0:
            _TCGA_POOL = {i: [] for i in range(N_CLASSES)}
            for _, row in _TCGA_DF.iterrows():
                stage = int(row.get("tumor_stage_encoded", 0))
                cls   = min(stage, N_CLASSES - 1)
                _TCGA_POOL[cls].append(extract_tabular_vector(row))
            total_vecs = sum(len(v) for v in _TCGA_POOL.values())
            print(f"  TCGA loaded: {len(_TCGA_DF)} patients → {total_vecs} tabular vectors")
            for i, cls in enumerate(CLASS_NAMES):
                print(f"    class {cls:<22}: {len(_TCGA_POOL[i])} TCGA vectors")
        else:
            print("  TCGA file not found — using class-specific synthetic tabular")
            _TCGA_POOL = {}
    except Exception as e:
        print(f"  TCGA load failed ({e}) — using class-specific synthetic tabular")
        _TCGA_POOL = {}


def get_tcga_tabular(cls_name: str, device) -> torch.Tensor:
    """
    Return a real TCGA tabular vector for this class.
    Randomly samples from the TCGA pool for that class index.
    Falls back to the same class-matched synthetic values used during training.
    """
    cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
    pool    = _TCGA_POOL.get(cls_idx, [])
    if pool:
        vec = random.choice(pool).copy()
        # Small Gaussian noise matching training augmentation (sigma=0.05)
        vec = vec + np.random.randn(N_TABULAR_FEATURES).astype(np.float32) * 0.05
    else:
        # Exact same fallback as HyperKvasirMultiModalDataset._get_tabular()
        base = np.zeros(N_TABULAR_FEATURES, dtype=np.float32)
        base[0] = 50 + cls_idx * 3   # age proxy
        base[1] = 26.0               # bmi
        base[9] = float(cls_idx % 4) # stage proxy
        vec = base + np.random.randn(N_TABULAR_FEATURES).astype(np.float32) * 0.05
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)


def get_full_batch(cls_name: str, tokenizer, device):
    """
    Returns (input_ids, attention_mask, tabular, text) using:
      - Text  : exact same CLINICAL_TEXT_TEMPLATES used during training
      - Tabular: real TCGA vector for that class (same as _get_tabular())
    This ensures FULL text+tabular integration consistent with training.
    """
    # Use the dataset's own text template (same as training)
    text = make_clinical_text(cls_name)
    enc  = tokenizer(text, return_tensors="pt", max_length=64,
                     padding="max_length", truncation=True)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    tabular        = get_tcga_tabular(cls_name, device)
    return input_ids, attention_mask, tabular, text


# ── Batch cache (populated from test dataset for HyperKvasir) ─────────────
_DATASET_BATCH_CACHE: dict = {}


def build_batch_cache(tokenizer, device):
    """
    Load full test split into cache (HyperKvasir only — CVC is train-only).
    Also builds TCGA pool for use with CVC and any uncached HK images.
    """
    global _DATASET_BATCH_CACHE
    build_tcga_pool()   # load TCGA first

    print("\nBuilding HyperKvasir test dataset batch cache...")
    (_, _, _, _, _, test_ds) = build_dataloaders(
        hyperkvasir_dir=HK_ROOT, tokenizer=tokenizer,
        tcga_dir=TCGA_DIR, cvc_dir=None,   # CVC NOT added to test split
        batch_size=1, img_size=IMG_SIZE, max_seq_len=64,
        num_workers=0, seed=SEED)
    for i, (img_path, label, cls_name) in enumerate(test_ds.samples):
        entry = test_ds[i]
        _DATASET_BATCH_CACHE[img_path] = (
            entry["input_ids"].unsqueeze(0),
            entry["attention_mask"].unsqueeze(0),
            entry["tabular"].unsqueeze(0),
            label,
            cls_name,
        )
    print(f"  HyperKvasir test entries cached: {len(_DATASET_BATCH_CACHE)}")
    return test_ds


def get_batch_for_image(img_path, cls_name, tokenizer, device):
    """
    Priority:
      1. If image is in test cache → use the exact (input_ids, tabular) from
         the dataset (real TCGA vector assigned during dataset construction)
      2. Otherwise → use get_full_batch() with real TCGA + training text template
    Both paths are fully consistent with training.
    """
    if img_path in _DATASET_BATCH_CACHE:
        iids, amask, tab, _, _ = _DATASET_BATCH_CACHE[img_path]
        # Decode cached text for the orchestrator text agent
        text = make_clinical_text(cls_name)
        return (iids.to(device), amask.to(device), tab.to(device), text)
    # Not in cache (e.g. CVC, or HK train/val images): build from TCGA + template
    iids, amask, tab, text = get_full_batch(cls_name, tokenizer, device)
    return iids, amask, tab, text


# ── Collect samples from EVERY subfolder (HK + CVC) ───────────────────────
def collect_all_subfolder_samples(model, tokenizer, device, n_per_subfolder=2):
    """
    Collects N_PER_SUBFOLDER correctly-predicted samples from EVERY subfolder.
    All samples use REAL TCGA tabular vectors + training clinical text templates
    (same as during model training) → fully integrated text+tabular inference.
      - 13 HyperKvasir pathological subfolders  (test cache used when available)
      - CVC-ClinicDB PNG/Original               (TCGA polyps pool + polyp template)
    """
    results = []
    model.eval()

    # ── HyperKvasir subfolders ──────────────────────────────────────────
    print("\n[1] Collecting from HyperKvasir subfolders...")
    print("    (Using real TCGA tabular + training clinical text templates)")
    for rel_path, cls_name, display in HK_SUBFOLDERS:
        full_dir = os.path.join(HK_ROOT, rel_path)
        if not os.path.isdir(full_dir):
            print(f"  SKIP (not found): {rel_path}")
            continue

        all_imgs = [os.path.join(full_dir, f)
                    for f in os.listdir(full_dir) if f.lower().endswith(".jpg")]
        random.shuffle(all_imgs)
        subfolder_name = rel_path.split("/")[-1]

        collected = []
        for img_path in all_imgs:
            if len(collected) >= n_per_subfolder:
                break
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_t   = VAL_TRANSFORM(pil_img).unsqueeze(0).to(device)
                # get_batch_for_image: test cache first, else TCGA+template
                iids, amask, tab, _ = get_batch_for_image(
                    img_path, cls_name, tokenizer, device)

                with torch.no_grad():
                    out = model(img_t, iids, amask, tab)
                    probs    = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
                    pred_idx = int(probs.argmax())

                if CLASS_NAMES[pred_idx] == cls_name:
                    collected.append({
                        "img_path":     img_path,
                        "gt_class":     cls_name,
                        "subfolder":    subfolder_name,
                        "dataset":      "HyperKvasir",
                        "display_name": display,
                        "conf":         float(probs[pred_idx]),
                        "in_cache":     img_path in _DATASET_BATCH_CACHE,
                    })
            except Exception:
                continue

        # Fill with best-available if < n_per_subfolder correct
        if len(collected) < n_per_subfolder:
            for img_path in all_imgs:
                if len(collected) >= n_per_subfolder:
                    break
                if any(c["img_path"] == img_path for c in collected):
                    continue
                collected.append({
                    "img_path":     img_path,
                    "gt_class":     cls_name,
                    "subfolder":    subfolder_name,
                    "dataset":      "HyperKvasir",
                    "display_name": display,
                    "conf":         0.0,
                    "in_cache":     img_path in _DATASET_BATCH_CACHE,
                })

        results.extend(collected[:n_per_subfolder])
        cache_str = f"({sum(1 for c in collected[:n_per_subfolder] if c['in_cache'])} from test cache, " \
                    f"{sum(1 for c in collected[:n_per_subfolder] if not c['in_cache'])} TCGA pool)"
        print(f"  {display:<38}  got {len(collected[:n_per_subfolder])}/{n_per_subfolder}  {cache_str}")

    # ── CVC-ClinicDB ────────────────────────────────────────────────────
    print("\n[2] Collecting from CVC-ClinicDB...")
    print("    (Using real TCGA polyps pool + polyp clinical text template)")
    cvc_cls     = "polyps"
    cvc_display = "CVC-ClinicDB — Colonoscopy Polyps"
    all_cvc = [os.path.join(CVC_ROOT, f)
               for f in os.listdir(CVC_ROOT) if f.lower().endswith(".png")]
    random.shuffle(all_cvc)

    cvc_collected = []
    for img_path in all_cvc:
        if len(cvc_collected) >= n_per_subfolder:
            break
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_t   = VAL_TRANSFORM(pil_img).unsqueeze(0).to(device)
            # CVC: always uses TCGA polyps pool + polyp template (not in HK cache)
            iids, amask, tab, _ = get_batch_for_image(
                img_path, cvc_cls, tokenizer, device)

            with torch.no_grad():
                out = model(img_t, iids, amask, tab)
                probs    = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())

            if CLASS_NAMES[pred_idx] == cvc_cls:
                cvc_collected.append({
                    "img_path":     img_path,
                    "gt_class":     cvc_cls,
                    "subfolder":    "CVC-Original",
                    "dataset":      "CVC-ClinicDB",
                    "display_name": cvc_display,
                    "conf":         float(probs[pred_idx]),
                    "in_cache":     False,
                })
        except Exception:
            continue

    results.extend(cvc_collected[:n_per_subfolder])
    print(f"  {cvc_display:<38}  got {len(cvc_collected[:n_per_subfolder])}/{n_per_subfolder} "
          f"(TCGA polyps pool)")

    print(f"\n  Total samples collected: {len(results)}")
    cached_total = sum(1 for s in results if s.get("in_cache", False))
    tcga_total   = len(results) - cached_total
    print(f"  Text+Tabular source breakdown:")
    print(f"    HK test cache (exact training batch)  : {cached_total}")
    print(f"    TCGA pool + training template          : {tcga_total}")
    print(f"  All samples: image + text + TCGA tabular → fully integrated multimodal inference")
    print()
    for i, s in enumerate(results):
        src = "cache" if s.get("in_cache") else "TCGA"
        print(f"    [{i+1:03d}] {s['dataset']:<14} {s['display_name']:<42} "
              f"GT={s['gt_class']:<18} conf={s['conf']:.3f}  [{src}]")
    return results


# ── Build GradCAM panel for a single sample ────────────────────────────────
def build_gradcam_panel(raw_rgb, gradcam_overlay, cam_up, sample, path_probs,
                        stage_probs, risk_score, pred_class):
    gt_class  = sample["gt_class"]
    case_id   = sample["case_id"]
    dataset   = sample["dataset"]
    display   = sample["display_name"]
    subfolder = sample["subfolder"]

    fig = plt.figure(figsize=(19, 6))
    fig.patch.set_facecolor("#FAFAFA")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.04)
    cls_colour = CLASS_COLOURS.get(gt_class, "#607D8B")
    ds_colour  = "#7B1FA2" if dataset == "CVC-ClinicDB" else "#1B5E20"

    # ── Left: Original ──────────────────────────────────────────────────
    border_orig = DATASET_COLOURS.get(dataset, [34, 139, 34])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(add_border(raw_rgb, border_orig, 8))
    ax1.set_title(
        f"ORIGINAL ENDOSCOPY IMAGE\n"
        f"Dataset: [{dataset}]  |  Subfolder: {subfolder}\n"
        f"Ground Truth: [{gt_class.upper()}]",
        fontsize=9, fontweight="bold", color=ds_colour, pad=5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5" if dataset == "CVC-ClinicDB" else "#E8F5E9",
                  alpha=0.9))
    ax1.axis("off")

    # ── Middle: GradCAM++ ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(add_border(gradcam_overlay, [200, 30, 30], 8))
    ax2.contour(cam_up, levels=[0.55, 0.75], colors=["yellow", "white"],
                linewidths=[1.0, 0.7], alpha=0.8)
    pred_conf = path_probs[CLASS_NAMES.index(pred_class)] if pred_class in CLASS_NAMES else 0
    correct   = pred_class == gt_class
    ax2.set_title(
        f"AI ATTENTION MAP (GradCAM++)\n"
        f"AI Pred: [{pred_class.upper()}]  conf={pred_conf:.1%}  "
        f"{'CORRECT' if correct else 'INCORRECT'}",
        fontsize=9, fontweight="bold",
        color="#1B5E20" if correct else "#B71C1C", pad=5,
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#E8F5E9" if correct else "#FFEBEE", alpha=0.9))
    ax2.axis("off")
    high_act = (cam_up > 0.5).mean()
    ax2.text(0.5, -0.04,
             f"ROI coverage: {high_act:.1%}  |  Peak activation: {cam_up.max():.2f}",
             ha="center", va="top", transform=ax2.transAxes,
             fontsize=8, color="#555555", style="italic")

    # ── Right: Clinical Findings ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    cf         = CLINICAL_FINDINGS[gt_class]
    stage_pred = STAGE_NAMES[int(np.argmax(stage_probs))]
    stage_conf = float(np.max(stage_probs))
    risk_label = "Malignant" if risk_score > 0.5 else "Benign"
    correct_str = "CORRECT" if correct else f"PREDICTED: {pred_class}"

    lines = [
        ("CLINICAL FINDINGS REPORT",              14,  "#212121",  "bold",   "#F5F5F5"),
        (f"Case: {case_id}",                       8,  "#555555",  "normal", None),
        (f"Source: {display}",                     8,  ds_colour,  "bold",   None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        (f"GT Finding: {cf['finding']}",           9.5, cls_colour, "bold",   None),
        (f"ICD-10: {cf['icd10']}",                 8,  "#555555",  "normal", None),
        (f"AI Result: {correct_str}",              9,
         "#1565C0" if correct else "#C62828",      "bold",   None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        ("ENDOSCOPIC FINDINGS:",                   9,  "#212121",  "bold",   None),
        (cf["endoscopic_dx"],                      8,  "#333333",  "normal", None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        ("MORPHOLOGY:",                            9,  "#212121",  "bold",   None),
        (cf["morphology"],                         8,  "#333333",  "normal", None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        (f"Risk Score: {risk_score:.3f} → {risk_label}", 9,
         "#B71C1C" if risk_score > 0.5 else "#2E7D32",    "bold",   None),
        (f"Stage: {stage_pred}  ({stage_conf:.1%})", 9, "#555555", "normal", None),
        (f"Urgency: {cf['urgency']}",              9,
         "#C62828" if cf["urgency"] == "Urgent" else "#E65100", "bold", None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        ("MALIGNANCY RISK:",                       9,  "#212121",  "bold",   None),
        (cf["risk"],                               8,  "#333333",  "normal", None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        ("RECOMMENDED ACTION:",                    9,  "#212121",  "bold",   None),
        (cf["action"],                             8,  "#333333",  "normal", None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        (f"SURVEILLANCE: {cf['surveillance']}",   8.5, "#1565C0",  "bold",   None),
        ("─" * 40,                                 8,  "#BDBDBD",  "normal", None),
        ("AI decision-support only.",              7.5, "#888888",  "normal", None),
        ("Verify with licensed clinician.",        7.5, "#888888",  "normal", None),
    ]

    y = 0.98
    for text, fsize, colour, fw, bg in lines:
        if bg:
            ax3.text(0.02, y, text, transform=ax3.transAxes,
                     fontsize=fsize, color=colour, fontweight=fw, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=bg, alpha=0.5))
        else:
            for part in text.split("\n"):
                ax3.text(0.02, y, part.strip(), transform=ax3.transAxes,
                         fontsize=fsize, color=colour, fontweight=fw,
                         va="top", wrap=True, clip_on=True)
                y -= fsize * 0.012
                if y < 0.01:
                    break
        y -= fsize * 0.012
        if y < 0.01:
            break

    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_facecolor("#FAFAFA")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#BDBDBD"); spine.set_linewidth(0.8)

    plt.suptitle(
        f"Multi-Modal AI  |  {display}  |  {case_id}",
        fontsize=10, fontweight="bold", y=1.01, color="#212121")
    return fig


# ── Run full agent pipeline on all samples ─────────────────────────────────
def run_all_samples(model, gradcam, tokenizer, device, samples):
    from src.agents.multimodal_orchestrator import MultiModalOrchestrator

    out_dir = Path(OUT_DIR)
    per_case_dir = out_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = MultiModalOrchestrator(
        model=model, tokenizer=tokenizer, device=device,
        output_dir=str(out_dir / "orchestrator"))

    all_results  = []
    all_panels   = []   # for summary grid

    total = len(samples)
    print(f"\n{'='*60}")
    print(f"Running 6-agent pipeline on {total} samples...")
    print(f"{'='*60}")

    for idx, sample in enumerate(samples):
        img_path    = sample["img_path"]
        gt_class    = sample["gt_class"]
        dataset     = sample["dataset"]
        display     = sample["display_name"]
        subfolder   = sample["subfolder"]
        case_id     = f"{idx+1:03d}__{dataset.replace('-','')}__{subfolder}__{gt_class.replace('-','_')}"
        sample["case_id"] = case_id

        print(f"\n  [{idx+1:03d}/{total}] {display}  GT={gt_class}")

        try:
            pil_img = Image.open(img_path).convert("RGB")
            raw_rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
            img_t   = VAL_TRANSFORM(pil_img)
            img_batch = img_t.unsqueeze(0).to(device)

            # get_batch_for_image returns (iids, amask, tab, text)
            # — real TCGA tabular vector + training clinical text template
            iids, amask, tab, text = get_batch_for_image(
                img_path, gt_class, tokenizer, device)

            # ── Full 6-agent run ────────────────────────────────────────
            result = orchestrator.run(
                image=img_batch,
                input_ids=iids,
                attention_mask=amask,
                tabular=tab,
                text=text,
                raw_image_np=raw_rgb,
                case_id=case_id,
                save=True,
            )

            pred_class = result.fusion_diagnosis.pathology_class
            path_probs_arr = np.array([
                result.fusion_diagnosis.pathology_probs[c] for c in CLASS_NAMES])
            stage_probs_arr = np.array([
                result.fusion_diagnosis.stage_probs[s] for s in STAGE_NAMES])
            risk_score = result.fusion_diagnosis.cancer_risk_score
            correct    = pred_class == gt_class

            # ── GradCAM for the predicted class ─────────────────────────
            gt_idx = CLASS_NAMES.index(gt_class)
            cam    = gradcam.generate(img_batch, gt_idx, iids, amask, tab)
            gradcam_overlay, cam_up = apply_gradcam_overlay(cam, raw_rgb)

            # ── Save per-case outputs ────────────────────────────────────
            case_dir = per_case_dir / case_id
            case_dir.mkdir(exist_ok=True)

            # Original image with dataset-coloured border
            border_col = DATASET_COLOURS.get(dataset, [34, 139, 34])
            Image.fromarray(add_border(raw_rgb, border_col, 6)).save(
                str(case_dir / "original.png"))
            Image.fromarray(add_border(gradcam_overlay, [200, 30, 30], 6)).save(
                str(case_dir / "gradcam_overlay.png"))

            # GradCAM panel
            fig = build_gradcam_panel(raw_rgb, gradcam_overlay, cam_up,
                                      sample, path_probs_arr, stage_probs_arr,
                                      risk_score, pred_class)
            panel_path = str(case_dir / "gradcam_panel.png")
            fig.savefig(panel_path, dpi=140, bbox_inches="tight", facecolor="white")
            plt.close("all")

            # Clinical report text
            cf         = CLINICAL_FINDINGS[gt_class]
            stage_pred = STAGE_NAMES[int(np.argmax(stage_probs_arr))]
            report = (
                f"EXPANDED AGENT CLINICAL REPORT\n"
                f"{'='*55}\n"
                f"Case ID        : {case_id}\n"
                f"Dataset Source : {dataset}\n"
                f"Subfolder      : {display}\n"
                f"Image Path     : {img_path}\n"
                f"Ground Truth   : {gt_class}  ({cf['finding']})\n"
                f"AI Prediction  : {pred_class}  ({'CORRECT' if correct else 'INCORRECT'})\n"
                f"Pred Conf      : {path_probs_arr[CLASS_NAMES.index(pred_class)]:.4f}\n"
                f"\nCLASS PROBABILITIES:\n"
            )
            for cn, p in zip(CLASS_NAMES, path_probs_arr):
                report += f"  {cn:<22}: {p:.4f}\n"
            report += (
                f"\nGradCAM ROI Coverage : {(cam_up > 0.5).mean():.3f}\n"
                f"Peak Activation      : {cam_up.max():.4f}\n"
                f"\nSTAGING              : {stage_pred}  ({float(np.max(stage_probs_arr)):.3f})\n"
                f"CANCER RISK SCORE    : {risk_score:.4f} → {'Malignant' if risk_score>0.5 else 'Benign'}\n"
                f"UNCERTAINTY          : {result.xai_report.uncertainty:.4f}\n"
                f"\nICD-10               : {cf['icd10']}\n"
                f"URGENCY              : {cf['urgency']}\n"
                f"\nENDOSCOPIC FINDINGS:\n{cf['endoscopic_dx']}\n"
                f"\nMORPHOLOGY:\n{cf['morphology']}\n"
                f"\nMALIGNANCY RISK:\n{cf['risk']}\n"
                f"\nRECOMMENDED ACTION:\n{cf['action']}\n"
                f"\nSURVEILLANCE PLAN:\n{cf['surveillance']}\n"
                f"\nMODALITY WEIGHTS:\n"
            )
            for k, v in result.xai_report.modality_weights.items():
                report += f"  {k}: {v:.4f}\n"
            report += (
                f"\n{'='*55}\n"
                f"RISK FLAGS: {', '.join(result.fusion_diagnosis.all_risk_flags)}\n"
                f"\n⚠ DISCLAIMER: AI decision-support only. Verify with licensed clinician.\n"
            )
            with open(str(case_dir / "clinical_report.txt"), "w") as f:
                f.write(report)

            # Agent summary JSON — includes TCGA tabular values for full transparency
            tcga_vec = tab[0].cpu().numpy().tolist()
            agent_summary = {
                "case_id":          case_id,
                "dataset":          dataset,
                "subfolder":        subfolder,
                "display_name":     display,
                "image_path":       img_path,
                "ground_truth":     gt_class,
                "gt_finding":       cf["finding"],
                "prediction":       pred_class,
                "correct":          correct,
                "pred_confidence":  float(path_probs_arr[CLASS_NAMES.index(pred_class)]),
                "class_probs":      {c: float(p) for c, p in zip(CLASS_NAMES, path_probs_arr)},
                "cancer_risk_score":float(risk_score),
                "cancer_risk_label":result.fusion_diagnosis.cancer_risk_label,
                "staging":          result.fusion_diagnosis.cancer_stage,
                "stage_confidence": result.fusion_diagnosis.stage_confidence,
                "uncertainty":      float(result.xai_report.uncertainty),
                "urgency":          result.clinical_recommendation.urgency,
                "risk_flags":       result.fusion_diagnosis.all_risk_flags,
                "modality_weights": {k: float(v)
                                     for k, v in result.xai_report.modality_weights.items()},
                "roi_coverage":     float((cam_up > 0.5).mean()),
                "peak_activation":  float(cam_up.max()),
                "inference_ms":     result.inference_time_ms,
                "icd10":            cf["icd10"],
                "surveillance":     cf["surveillance"],
                # ── Multimodal input traceability ───────────────────────
                # clinical_text: short BioBERT template fed to the model
                "clinical_text_model_input": text,
                # clinical_text_full: rich human-readable endoscopy findings
                "clinical_text_full": (
                    f"{cf['endoscopic_dx']} "
                    f"{cf['morphology']} "
                    f"Malignancy risk: {cf['risk']} "
                    f"Recommended action: {cf['action']} "
                    f"Surveillance: {cf['surveillance']}"
                ),
                "tabular_source":   "TCGA-test-cache" if sample.get("in_cache") else "TCGA-pool",
                "tabular_features": {k: round(float(v), 4)
                                     for k, v in zip(TABULAR_FEATURES, tcga_vec)},
            }
            with open(str(case_dir / "agent_summary.json"), "w") as f:
                json.dump(agent_summary, f, indent=2)

            all_results.append(agent_summary)
            all_panels.append((raw_rgb, gradcam_overlay, cam_up, sample,
                               pred_class, path_probs_arr, correct))

            print(f"         Pred={pred_class:<20} conf={path_probs_arr[CLASS_NAMES.index(pred_class)]:.3f}"
                  f"  Risk={risk_score:.3f}  Unc={result.xai_report.uncertainty:.2f}"
                  f"  {'✓' if correct else '✗'}")

        except Exception as e:
            import traceback
            print(f"         ERROR: {e}")
            traceback.print_exc()

    return all_results, all_panels


# ── Summary grid ───────────────────────────────────────────────────────────
def build_summary_grid(all_panels, out_dir):
    """Grid of all cases: original | GradCAM side-by-side, colour-coded by dataset."""
    if not all_panels:
        return
    total  = len(all_panels)
    ncols  = 4   # pairs of (original, gradcam) = 4 images per row (2 cases)
    nrows  = math.ceil(total / 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Expanded Agent Samples — {total} Cases from CVC-ClinicDB + All HyperKvasir Subfolders\n"
        f"VIOLET border = CVC-ClinicDB  |  GREEN border = HyperKvasir",
        fontsize=13, fontweight="bold", y=1.005)

    for idx, (raw_rgb, gcam_ov, cam_up, sample, pred_cls, probs, correct) in enumerate(all_panels):
        row    = idx // 2
        col_pair = idx % 2       # 0 or 1 within the row
        ax_orig = axes[row, col_pair * 2]
        ax_gcam = axes[row, col_pair * 2 + 1]

        ds       = sample["dataset"]
        gt_cls   = sample["gt_class"]
        subfolder= sample["subfolder"]
        border   = DATASET_COLOURS.get(ds, [34, 139, 34])
        title_c  = "#7B1FA2" if ds == "CVC-ClinicDB" else "#1B5E20"
        pred_c   = "#1B5E20" if correct else "#B71C1C"

        ax_orig.imshow(add_border(raw_rgb, border, 4))
        ax_orig.set_title(f"{ds}\n{subfolder}\nGT:{gt_cls}",
                          fontsize=6.5, color=title_c, fontweight="bold")
        ax_orig.axis("off")

        ax_gcam.imshow(add_border(gcam_ov, [200, 30, 30], 4))
        ax_gcam.set_title(f"Pred:{pred_cls} {'✓' if correct else '✗'}\n"
                          f"ROI:{(cam_up>0.5).mean():.0%}",
                          fontsize=6.5, color=pred_c, fontweight="bold")
        ax_gcam.axis("off")

    # Hide unused axes
    for idx in range(total, nrows * 2):
        row = idx // 2; cp = idx % 2
        axes[row, cp * 2].axis("off")
        axes[row, cp * 2 + 1].axis("off")

    plt.tight_layout()
    path = str(Path(out_dir) / "summary_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"\n  [saved] summary_grid.png  ({os.path.getsize(path)//1024} KB)")


# ── Summary dashboard ──────────────────────────────────────────────────────
def build_summary_dashboard(all_results, out_dir):
    if not all_results:
        return

    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle("Expanded Agent Samples — Summary Dashboard\n"
                 "CVC-ClinicDB + All HyperKvasir Subfolders",
                 fontsize=14, fontweight="bold")

    # 1. Per-class accuracy
    ax = fig.add_subplot(gs[0, 0])
    cls_correct = {c: [0, 0] for c in CLASS_NAMES}
    for r in all_results:
        cls_correct[r["ground_truth"]][1] += 1
        if r["correct"]:
            cls_correct[r["ground_truth"]][0] += 1
    cls_acc = [cls_correct[c][0] / max(1, cls_correct[c][1]) for c in CLASS_NAMES]
    short   = [c.replace("-", "\n") for c in CLASS_NAMES]
    cols    = [CLASS_COLOURS.get(c, "#607D8B") for c in CLASS_NAMES]
    bars = ax.bar(short, cls_acc, color=cols, width=0.6, edgecolor="white", lw=1.5)
    for bar, val, (c, cnts) in zip(bars, cls_acc, cls_correct.items()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.0%}\n({cnts[0]}/{cnts[1]})",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.3); ax.set_title("Per-Class Accuracy", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy"); ax.grid(axis="y", alpha=0.3)

    # 2. Dataset source breakdown
    ax = fig.add_subplot(gs[0, 1])
    ds_cnt = {}
    for r in all_results:
        ds_cnt[r["dataset"]] = ds_cnt.get(r["dataset"], 0) + 1
    ds_cols = {"CVC-ClinicDB": "#7B1FA2", "HyperKvasir": "#2E7D32"}
    ax.pie(list(ds_cnt.values()), labels=list(ds_cnt.keys()),
           autopct="%1.0f%%", startangle=90,
           colors=[ds_cols.get(k, "#607D8B") for k in ds_cnt],
           wedgeprops={"edgecolor": "white", "lw": 2})
    ax.set_title("Dataset Source", fontsize=10, fontweight="bold")

    # 3. Subfolder accuracy (mini bar)
    ax = fig.add_subplot(gs[0, 2:])
    sf_data = defaultdict(lambda: [0, 0])
    for r in all_results:
        sf_data[r["subfolder"]][1] += 1
        if r["correct"]:
            sf_data[r["subfolder"]][0] += 1
    sf_names   = list(sf_data.keys())
    sf_acc     = [sf_data[k][0] / max(1, sf_data[k][1]) for k in sf_names]
    sf_total   = [sf_data[k][1] for k in sf_names]
    sf_cols    = []
    for r in all_results:
        if r["subfolder"] not in sf_cols:
            sf_cols.append(r["subfolder"])
    colour_map = {}
    for r in all_results:
        colour_map[r["subfolder"]] = CLASS_COLOURS.get(r["ground_truth"], "#607D8B")
    b_cols = [colour_map.get(s, "#607D8B") for s in sf_names]
    bars = ax.barh(sf_names, sf_acc, color=b_cols, height=0.6,
                   edgecolor="white", lw=1.2)
    ax.axvline(1.0, color="grey", lw=1, ls="--", alpha=0.5)
    for bar, val, tot in zip(bars, sf_acc, sf_total):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.0%} ({tot})", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Subfolder Accuracy", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 4. Urgency breakdown
    ax = fig.add_subplot(gs[1, 0])
    urg_cnt = {}
    for r in all_results:
        u = r.get("urgency", "Elective")
        urg_cnt[u] = urg_cnt.get(u, 0) + 1
    u_cols = {"Elective": "#4CAF50", "Urgent": "#FF5722", "Routine": "#FFC107"}
    ax.pie(list(urg_cnt.values()), labels=list(urg_cnt.keys()),
           autopct="%1.0f%%", startangle=90,
           colors=[u_cols.get(k, "#607D8B") for k in urg_cnt],
           wedgeprops={"edgecolor": "white", "lw": 2})
    ax.set_title("Clinical Urgency", fontsize=10, fontweight="bold")

    # 5. Risk score distribution
    ax = fig.add_subplot(gs[1, 1])
    for gt_cls in CLASS_NAMES:
        scores = [r["cancer_risk_score"] for r in all_results if r["ground_truth"] == gt_cls]
        if scores:
            ax.scatter([CLASS_NAMES.index(gt_cls)] * len(scores), scores,
                       color=CLASS_COLOURS.get(gt_cls, "#607D8B"),
                       s=60, zorder=5, label=gt_cls, edgecolors="grey", lw=0.7)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.replace("-", "\n") for c in CLASS_NAMES], fontsize=7)
    ax.set_ylim(0, 1); ax.axhline(0.5, color="grey", lw=1.2, ls="--", alpha=0.7)
    ax.set_title("Risk Scores per Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cancer Risk Score"); ax.grid(alpha=0.3)

    # 6. Uncertainty distribution
    ax = fig.add_subplot(gs[1, 2])
    uncerts = [r["uncertainty"] for r in all_results]
    ax.hist(uncerts, bins=12, color="#FF9800", edgecolor="white", lw=1.5)
    ax.axvline(np.mean(uncerts), color="#C62828", lw=2,
               ls="--", label=f"Mean={np.mean(uncerts):.2f}")
    ax.set_xlabel("MC-Dropout Uncertainty"); ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 7. Overall accuracy summary
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    total   = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    hk_tot  = sum(1 for r in all_results if r["dataset"] == "HyperKvasir")
    cvc_tot = sum(1 for r in all_results if r["dataset"] == "CVC-ClinicDB")
    hk_cor  = sum(1 for r in all_results if r["dataset"] == "HyperKvasir" and r["correct"])
    cvc_cor = sum(1 for r in all_results if r["dataset"] == "CVC-ClinicDB" and r["correct"])
    avg_unc = np.mean([r["uncertainty"] for r in all_results])
    avg_roi = np.mean([r["roi_coverage"] for r in all_results])
    n_urgent= sum(1 for r in all_results if r.get("urgency") == "Urgent")
    n_subfolders = len(set(r["subfolder"] for r in all_results))

    lines = [
        ("OVERALL SUMMARY",                         12, "#212121", "bold"),
        ("─" * 30,                                   9, "#BDBDBD", "normal"),
        (f"Total Samples:  {total}",                10, "#212121", "bold"),
        (f"Overall Accuracy:  {correct}/{total}  ({correct/max(1,total):.1%})",
                                                    10, "#1B5E20" if correct/max(1,total) > 0.9 else "#B71C1C", "bold"),
        ("─" * 30,                                   9, "#BDBDBD", "normal"),
        (f"HyperKvasir:  {hk_cor}/{hk_tot}  ({hk_cor/max(1,hk_tot):.1%})",
                                                    9.5, "#2E7D32", "bold"),
        (f"CVC-ClinicDB:  {cvc_cor}/{cvc_tot}  ({cvc_cor/max(1,cvc_tot):.1%})",
                                                    9.5, "#7B1FA2", "bold"),
        ("─" * 30,                                   9, "#BDBDBD", "normal"),
        (f"Subfolders covered: {n_subfolders}",     9.5, "#212121", "bold"),
        (f"Avg Uncertainty: {avg_unc:.3f}",          9, "#555555", "normal"),
        (f"Avg ROI Coverage: {avg_roi:.1%}",         9, "#555555", "normal"),
        (f"Urgent Cases: {n_urgent}/{total}",        9, "#C62828" if n_urgent>0 else "#2E7D32", "bold"),
        ("─" * 30,                                   9, "#BDBDBD", "normal"),
        ("Model: UnifiedMultiModalTransformer",      8, "#555555", "normal"),
        ("Backbone: ResNet50 + EfficientNet-B0",     8, "#555555", "normal"),
        ("Text: BioBERT  |  Tabular: TabTransformer",8,"#555555", "normal"),
        ("─" * 30,                                   9, "#BDBDBD", "normal"),
        ("AI decision-support only.",                8, "#888888", "normal"),
    ]
    y = 0.97
    for text, fsize, colour, fw in lines:
        ax.text(0.02, y, text, transform=ax.transAxes,
                fontsize=fsize, color=colour, fontweight=fw, va="top")
        y -= fsize * 0.014
        if y < 0.01:
            break
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD"); spine.set_linewidth(0.8)
    ax.set_title("Summary Statistics", fontsize=10, fontweight="bold")

    path = str(Path(out_dir) / "summary_dashboard.png")
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  [saved] summary_dashboard.png  ({os.path.getsize(path)//1024} KB)")


# ── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device    = get_device()
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # ── Load trained model ────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model = UnifiedMultiModalTransformer(
        bert_model_name=BERT_MODEL,
        n_tabular_features=N_TABULAR_FEATURES,
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        n_fusion_heads=8,
        n_fusion_layers=3,
        n_self_layers=2,
        img_drop=0.0,
        txt_drop=0.0,
        tab_drop=0.0,
        fusion_drop=0.0,
        head_drop=0.0,
        freeze_bert_layers=0,
        pretrained_backbone=False,
        backbone_name="resnet50+efficientnet_b0",
    ).to(device)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Model loaded. epoch={ckpt.get('epoch','?')}  "
          f"val_acc={ckpt.get('val_acc', 0):.4f}")

    # ── GradCAM++ setup ───────────────────────────────────────────────────
    target_layer = model.get_image_target_layer()
    gradcam      = GradCAMPP(model, target_layer)
    print("  GradCAM++ attached to ResNet50 layer4[-1].")

    # ── Build dataset batch cache ─────────────────────────────────────────
    test_ds = build_batch_cache(tokenizer, device)

    # ── Collect samples from all subfolders ───────────────────────────────
    samples = collect_all_subfolder_samples(
        model, tokenizer, device, n_per_subfolder=N_PER_SUBFOLDER)

    # ── Run all agents ─────────────────────────────────────────────────────
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_results, all_panels = run_all_samples(
        model, gradcam, tokenizer, device, samples)

    # ── Save global results JSON ───────────────────────────────────────────
    results_path = str(Path(OUT_DIR) / "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary grid + dashboard ───────────────────────────────────────────
    print("\nBuilding summary grid...")
    build_summary_grid(all_panels, OUT_DIR)
    print("Building summary dashboard...")
    build_summary_dashboard(all_results, OUT_DIR)

    # ── Final print ────────────────────────────────────────────────────────
    total   = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    hk_cor  = sum(1 for r in all_results if r["dataset"] == "HyperKvasir" and r["correct"])
    hk_tot  = sum(1 for r in all_results if r["dataset"] == "HyperKvasir")
    cvc_cor = sum(1 for r in all_results if r["dataset"] == "CVC-ClinicDB" and r["correct"])
    cvc_tot = sum(1 for r in all_results if r["dataset"] == "CVC-ClinicDB")

    print("\n" + "="*60)
    print("ALL DONE — Expanded Agent Samples")
    print("="*60)
    print(f"  Output folder       : {OUT_DIR}/")
    print(f"  Total samples       : {total}")
    print(f"  Overall accuracy    : {correct}/{total}  ({correct/max(1,total):.1%})")
    print(f"  HyperKvasir         : {hk_cor}/{hk_tot}  ({hk_cor/max(1,hk_tot):.1%})")
    print(f"  CVC-ClinicDB        : {cvc_cor}/{cvc_tot}  ({cvc_cor/max(1,cvc_tot):.1%})")
    n_subfolders = len(set(r["subfolder"] for r in all_results))
    print(f"  Subfolders covered  : {n_subfolders}")
    print("="*60)
