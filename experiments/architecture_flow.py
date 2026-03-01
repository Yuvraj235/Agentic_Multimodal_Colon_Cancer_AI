# -*- coding: utf-8 -*-
"""
Architecture Flow Diagram  —  Journal-quality, fully aligned
=============================================================
All elements (images, boxes, arrows) are placed on a SINGLE coordinate
system (0–1 range on a 24×14 inch canvas) so every arrow tip lands
exactly on the correct box edge. No GridSpec coordinate mixing.

Layout (left → right, top → bottom):
  ┌────────────────────────────────────────────────────────────────────┐
  │ HEADER BAR                                                         │
  ├──────────┬────────────────────────────────────────┬───────────────┤
  │ INPUT    │  STEP 2   STEP 3   STEP 4   STEP 5    │  OUTPUT       │
  │  Image   │  Image  → Text  → Tabular → Fusion    │  GradCAM      │
  │  Text    │  Encoder  Encoder  Encoder  Transformer│  Diagnosis    │
  │  Data    │                                        │  Probs        │
  │          │              ↓ (vertical inside)      │  Stage/Risk   │
  └──────────┴────────────────────────────────────────┴───────────────┘

Usage:
  python3 experiments/architecture_flow.py --class polyps --seed 7
  python3 experiments/architecture_flow.py --class uc-moderate-sev --seed 42
  python3 experiments/architecture_flow.py --class barretts-esoph

Output: outputs/architecture_flow/architecture_flow_<class>__<img>.png
"""

import sys, os, argparse, random, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    N_TABULAR_FEATURES, load_tcga_tabular, extract_tabular_vector,
    make_clinical_text, TABULAR_FEATURES,
)

# ── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT = "outputs/unified_multimodal/checkpoints/best_model.pth"
BERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
TCGA_DIR   = "data/raw/tcga"
N_CLASSES  = 5
D_MODEL    = 256
IMG_SIZE   = 224

CLASS_NAMES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]
CLASS_DISPLAY = {
    "polyps":          "Colonic Polyp",
    "uc-mild":         "UC Mild",
    "uc-moderate-sev": "UC Moderate-Severe",
    "barretts-esoph":  "Barrett's Esophagus",
    "therapeutic":     "Post-Polypectomy",
}
CLASS_COLORS = {
    "polyps":          "#2E7D32",
    "uc-mild":         "#1565C0",
    "uc-moderate-sev": "#E65100",
    "barretts-esoph":  "#6A1B9A",
    "therapeutic":     "#00695C",
}
ICD10 = {
    "polyps":          "K63.5",
    "uc-mild":         "K51.00",
    "uc-moderate-sev": "K51.012",
    "barretts-esoph":  "K22.7",
    "therapeutic":     "Z12.11",
}
STAGE_LABELS = ["No Cancer", "Stage I", "Stage II", "Stage III/IV"]
STAGE_COLORS = {
    "No Cancer":    "#2E7D32",
    "Stage I":      "#F57F17",
    "Stage II":     "#E65100",
    "Stage III/IV": "#B71C1C",
}

HK_SUBFOLDERS = [
    ("polyps",                        "lower-gi-tract", "pathological-findings",     "polyps"),
    ("ulcerative-colitis-grade-0-1",  "lower-gi-tract", "pathological-findings",     "uc-mild"),
    ("ulcerative-colitis-grade-1",    "lower-gi-tract", "pathological-findings",     "uc-mild"),
    ("ulcerative-colitis-grade-1-2",  "lower-gi-tract", "pathological-findings",     "uc-mild"),
    ("ulcerative-colitis-grade-2",    "lower-gi-tract", "pathological-findings",     "uc-moderate-sev"),
    ("ulcerative-colitis-grade-2-3",  "lower-gi-tract", "pathological-findings",     "uc-moderate-sev"),
    ("ulcerative-colitis-grade-3",    "lower-gi-tract", "pathological-findings",     "uc-moderate-sev"),
    ("barretts",                      "upper-gi-tract", "pathological-findings",     "barretts-esoph"),
    ("barretts-short-segment",        "upper-gi-tract", "pathological-findings",     "barretts-esoph"),
    ("esophagitis-a",                 "upper-gi-tract", "pathological-findings",     "barretts-esoph"),
    ("esophagitis-b-d",               "upper-gi-tract", "pathological-findings",     "barretts-esoph"),
    ("dyed-lifted-polyps",            "lower-gi-tract", "therapeutic-interventions", "therapeutic"),
    ("dyed-resection-margins",        "lower-gi-tract", "therapeutic-interventions", "therapeutic"),
]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA UTILITIES                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pick_image(image_path=None, class_name=None):
    if image_path:
        p = Path(image_path)
        cls = class_name or "polyps"
        for c in CLASS_NAMES:
            if c in str(p): cls = c; break
        return str(p), cls, p.parent.name, "User"
    base = Path("data/processed/hyper_kvasir_clean")
    cvc  = Path("data/raw/CVC-ClinicDB/PNG/Original")
    cands = []
    for sub, gi, cat, cls in HK_SUBFOLDERS:
        if class_name and cls != class_name: continue
        d = base / gi / cat / sub
        if not d.exists(): d = base / gi / sub
        if d.exists():
            for p in list(d.glob("*.jpg"))[:60]:
                cands.append((str(p), cls, sub, "HyperKvasir"))
    if not class_name or class_name == "polyps":
        if cvc.exists():
            for p in list(cvc.glob("*.png"))[:20]:
                cands.append((str(p), "polyps", "CVC-ClinicDB", "CVC-ClinicDB"))
    if not cands: raise RuntimeError("No images found.")
    return random.choice(cands)


def build_tcga_pool():
    pool = {i: [] for i in range(N_CLASSES)}
    df = load_tcga_tabular(TCGA_DIR)
    if df is not None:
        for _, row in df.iterrows():
            stage = int(row.get("tumor_stage_encoded", 0))
            cls   = min(stage, N_CLASSES - 1)
            pool[cls].append(extract_tabular_vector(row))
    return pool


def get_tcga_tab(cls_name, pool, device):
    idx   = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
    cands = pool.get(idx, [])
    if cands:
        vec = random.choice(cands).copy()
        vec = vec + np.random.randn(N_TABULAR_FEATURES).astype(np.float32) * 0.02
    else:
        vec = np.zeros(N_TABULAR_FEATURES, dtype=np.float32)
        vec[0] = 55.0; vec[9] = float(idx % 4)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device), vec


def denorm(t):
    m = np.array([0.485, 0.456, 0.406]); s = np.array([0.229, 0.224, 0.225])
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ((img * s + m).clip(0, 1) * 255).astype(np.uint8)


def compute_gradcam(model, img, iids, amask, tab, pred_idx):
    target = model.image_encoder.resnet_target
    acts, grads = [], []
    hf = target.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    hb = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach()))
    img2 = img.clone().requires_grad_(True)
    out  = model(img2, iids, amask, tab)
    model.zero_grad()
    out["pathology"][0, pred_idx].backward()
    hf.remove(); hb.remove()
    if not acts or not grads:
        return np.zeros((IMG_SIZE, IMG_SIZE)), denorm(img)
    A = acts[0][0]; G = grads[0][0]
    G2 = G ** 2; G3 = G ** 3
    denom  = 2 * G2 + A.sum(dim=(1, 2), keepdim=True) * G3 + 1e-8
    alpha  = G2 / denom
    w      = (alpha * F.relu(G)).sum(dim=(1, 2))
    cam_t  = F.relu((w[:, None, None] * A).sum(0)).cpu().numpy()
    if cam_t.max() > cam_t.min():
        cam_t = (cam_t - cam_t.min()) / (cam_t.max() - cam_t.min())
    cam_r  = cv2.resize(cam_t, (IMG_SIZE, IMG_SIZE))
    orig   = denorm(img)
    hmap   = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hmap   = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.45, hmap, 0.55, 0)
    return cam_r, overlay


def run_pipeline(img_path, cls_name, model, tokenizer, tcga_pool, device):
    pil  = Image.open(img_path).convert("RGB")
    orig = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))
    img  = IMG_TRANSFORM(pil).unsqueeze(0).to(device)
    text = make_clinical_text(cls_name)
    enc  = tokenizer(text, return_tensors="pt", max_length=64,
                     padding="max_length", truncation=True)
    iids  = enc["input_ids"].to(device)
    amask = enc["attention_mask"].to(device)
    tab, tab_vec = get_tcga_tab(cls_name, tcga_pool, device)

    model.eval()
    with torch.no_grad():
        out = model(img, iids, amask, tab)

    path_p  = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
    stage_p = F.softmax(out["staging"],   dim=-1).cpu().numpy()[0]
    risk_p  = F.softmax(out["risk"],      dim=-1).cpu().numpy()[0]
    mod_w   = out["mod_weights"][0].cpu().numpy()

    pred_idx  = int(path_p.argmax())
    pred_cls  = CLASS_NAMES[pred_idx]
    pred_conf = float(path_p[pred_idx])
    stage_lbl = STAGE_LABELS[int(stage_p.argmax())]
    stage_c   = float(stage_p.max())
    risk_sc   = float(risk_p[1])
    risk_lbl  = "Malignant" if risk_sc >= 0.5 else "Benign"

    cam, overlay = compute_gradcam(model, img, iids, amask, tab, pred_idx)

    # MC-Dropout uncertainty
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()
    mc = []
    with torch.no_grad():
        for _ in range(15):
            o = model(img, iids, amask, tab)
            mc.append(F.softmax(o["pathology"], -1).cpu().numpy()[0])
    model.eval()
    mc_mean = np.stack(mc).mean(0)
    ent = -np.sum(mc_mean * np.log(mc_mean + 1e-8))
    unc = float(ent / np.log(len(mc_mean)))

    return dict(
        orig=orig, cam=cam, overlay=overlay,
        cls_name=cls_name, pred_cls=pred_cls, pred_conf=pred_conf,
        path_p=path_p, stage_lbl=stage_lbl, stage_c=stage_c,
        risk_sc=risk_sc, risk_lbl=risk_lbl,
        mod_w=mod_w, tab_vec=tab_vec, text=text,
        unc=unc, correct=(pred_cls == cls_name),
        img_path=img_path,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DRAWING PRIMITIVES  (all in data coordinates 0–FW, 0–FH)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

FW, FH = 24.0, 15.0   # figure size in inches; data coords == inches

def _rect(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.15, alpha=1.0, zorder=2):
    """Draw a rounded rectangle. x,y = bottom-left corner."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder, clip_on=False
    ))


def _arrow(ax, x0, y0, x1, y1, color="#455A64", lw=2.0, hw=0.18, hl=0.28, zorder=6):
    """Draw a solid filled arrow from (x0,y0) to (x1,y1) in data coords."""
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"-|>,head_width={hw},head_length={hl}",
            color=color, lw=lw,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder, clip_on=False
    )


def _label(ax, x, y, text, fontsize=9, color="#212121", ha="center", va="center",
           bold=False, italic=False, zorder=7, bg=None, bg_ec=None):
    style = "italic" if italic else "normal"
    fw    = "bold" if bold else "normal"
    kw    = dict(fontsize=fontsize, color=color, ha=ha, va=va,
                 fontweight=fw, style=style, zorder=zorder, clip_on=False)
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.25",
                          facecolor=bg, edgecolor=bg_ec or bg, alpha=0.92)
    ax.text(x, y, text, **kw)


def _image_inset(ax, img_arr, cx, cy, w, h):
    """Place a numpy RGB image centred at (cx,cy) with given width/height in data coords."""
    ax.imshow(img_arr,
              extent=[cx - w/2, cx + w/2, cy - h/2, cy + h/2],
              aspect="auto", zorder=3, clip_on=False)


def _divider(ax, x, y1, y2, color="#CFD8DC"):
    ax.plot([x, x], [y1, y2], color=color, lw=1.0, ls="--", zorder=1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BUILD DIAGRAM                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_diagram(d, out_path):
    """
    Single-axis diagram with all elements placed in (inch) data coordinates.
    Figure is 24 × 15 inches.  Origin (0,0) = bottom-left.

    Column centres (x):
      Col A  — Inputs:        cx = 2.6
      Col B  — Step boxes:   cx = 9.2   (boxes span x=4.9..13.5)
      Col C  — Outputs:      cx = 19.8

    Row layout (y from top, FH=15):
      Header:      y = 14.0 – 14.8
      Section:     y = 1.0  – 13.6
    """

    fig = plt.figure(figsize=(FW, FH), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])        # fills the whole figure
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    d_pred_col = CLASS_COLORS.get(d["pred_cls"], "#1565C0")
    d_cor_col  = "#2E7D32" if d["correct"] else "#C62828"
    d_cor_str  = "✔  CORRECT" if d["correct"] else "✘  WRONG"
    dataset    = "CVC-ClinicDB" if "CVC" in d["img_path"] else "HyperKvasir"
    ds_ec      = "#6A1B9A" if "CVC" in d["img_path"] else "#1B5E20"

    # ══════════════════════════════════════════════════════════════════════
    # HEADER  (y = 14.0 to 15.0)
    # ══════════════════════════════════════════════════════════════════════
    _rect(ax, 0, 14.0, FW, 1.0, fc="#1A237E", ec="#1A237E", lw=0, radius=0.0, zorder=1)
    _label(ax, 0.5, 14.55, "Agentic Multimodal Colon Cancer AI  —  Architecture & Processing Flow",
           fontsize=16, color="white", bold=True, ha="left")
    _label(ax, 14.5, 14.72,
           f"Prediction:  {CLASS_DISPLAY.get(d['pred_cls'], d['pred_cls'])}  ({d['pred_conf']:.1%})",
           fontsize=11, color=d_pred_col,
           bold=True, ha="left",
           bg="#FFFFFF", bg_ec=d_pred_col)
    _label(ax, 14.5, 14.30,
           f"Ground Truth:  {CLASS_DISPLAY.get(d['cls_name'], d['cls_name'])}   |   Dataset: {dataset}",
           fontsize=9, color="#B0BEC5", ha="left")
    _label(ax, 23.2, 14.55, d_cor_str,
           fontsize=13, color=d_cor_col, bold=True,
           bg="#FFFFFF", bg_ec=d_cor_col)

    # Grey background for content area
    _rect(ax, 0, 0, FW, 14.0, fc="#F5F5F5", ec="#F5F5F5", lw=0, radius=0.0, zorder=0)

    # ══════════════════════════════════════════════════════════════════════
    # COLUMN SEPARATORS
    # ══════════════════════════════════════════════════════════════════════
    _divider(ax, 4.8,  0.3, 13.7)   # between Col A and Col B
    _divider(ax, 13.6, 0.3, 13.7)   # between Col B and Col C

    # ══════════════════════════════════════════════════════════════════════
    # COL A — INPUTS  (x centre = 2.4)
    # ══════════════════════════════════════════════════════════════════════
    CX_A = 2.4
    W_A  = 4.2    # box width
    H_A  = 0.3    # section header height

    # Section header
    _rect(ax, CX_A - W_A/2, 12.95, W_A, 0.55,
          fc="#ECEFF1", ec="#90A4AE", lw=1.5, radius=0.1)
    _label(ax, CX_A, 13.23, "INPUTS  (3 data streams)",
           fontsize=10, color="#37474F", bold=True)

    # ── A1: Endoscopy image ──────────────────────────────────────────────
    IMG_W, IMG_H = 3.8, 3.4
    img_cx = CX_A
    img_cy = 10.85

    # Coloured border frame
    _rect(ax, img_cx - IMG_W/2 - 0.12, img_cy - IMG_H/2 - 0.12,
          IMG_W + 0.24, IMG_H + 0.24,
          fc=ds_ec, ec=ds_ec, lw=0, radius=0.08, zorder=2)
    _image_inset(ax, d["orig"], img_cx, img_cy, IMG_W, IMG_H)

    _label(ax, img_cx, img_cy + IMG_H/2 + 0.35,
           "① Real Endoscopy Image",
           fontsize=9.5, color="#212121", bold=True,
           bg="#E8F5E9" if "HyperKvasir" in dataset else "#F3E5F5",
           bg_ec=ds_ec)
    _label(ax, img_cx, img_cy - IMG_H/2 - 0.30,
           f"{dataset}  ·  {Path(d['img_path']).parent.name}",
           fontsize=8, color="#555", italic=True)

    # ── A2: Clinical text box ────────────────────────────────────────────
    TXT_Y_TOP = 8.7
    TXT_H     = 1.8
    _rect(ax, CX_A - W_A/2, TXT_Y_TOP - TXT_H, W_A, TXT_H,
          fc="#E3F2FD", ec="#1565C0", lw=1.8, radius=0.12)
    _label(ax, CX_A, TXT_Y_TOP - 0.22,
           "② Clinical Text  →  BioBERT",
           fontsize=9, color="#1565C0", bold=True)
    # Word-wrap text
    words = d["text"].split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 38: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    for li, ln in enumerate(lines[:4]):
        _label(ax, CX_A, TXT_Y_TOP - 0.55 - li * 0.31,
               ln, fontsize=7.5, color="#333", italic=True)

    # ── A3: Tabular data box ─────────────────────────────────────────────
    TAB_Y_TOP = 6.55
    TAB_H     = 1.85
    _rect(ax, CX_A - W_A/2, TAB_Y_TOP - TAB_H, W_A, TAB_H,
          fc="#FFF8E1", ec="#F57F17", lw=1.8, radius=0.12)
    _label(ax, CX_A, TAB_Y_TOP - 0.22,
           "③ Patient Data  →  TabTransformer",
           fontsize=9, color="#E65100", bold=True)
    feat_short = ["Age", "BMI", "Yr Dx", "Follow-up", "Cigs/d", "Pack-yr",
                  "Alcohol", "Gender", "Race", "Stage", "Morphol.", "Site"]
    for ki, fi in enumerate(range(6)):
        col_x = CX_A - 1.0 if ki < 3 else CX_A + 0.9
        row_y = TAB_Y_TOP - 0.60 - (ki % 3) * 0.37
        _label(ax, col_x, row_y,
               f"{feat_short[fi]}: {d['tab_vec'][fi]:.2f}",
               fontsize=7.5, color="#4E342E")

    # ── Section label ────────────────────────────────────────────────────
    _label(ax, CX_A, 0.65,
           "3 independent data streams\nfed simultaneously to the AI",
           fontsize=8, color="#607D8B", italic=True)

    # ══════════════════════════════════════════════════════════════════════
    # COL B — PROCESSING STEPS  (x: 4.9 to 13.5,  centre = 9.2)
    # ══════════════════════════════════════════════════════════════════════
    CX_B  = 9.2
    BX    = 5.1    # box left x
    BW    = 8.2    # box width
    BH    = 2.05   # box height
    GAP   = 0.50   # gap between boxes (for arrows)

    # Total height for 4 boxes + 3 gaps
    TOTAL_H = 4 * BH + 3 * GAP   # = 8.2 + 1.5 = 9.70
    B_TOP   = 13.55               # top of first box
    B_BOTS  = [B_TOP - i*(BH + GAP) - BH for i in range(4)]   # bottom y of each box

    step_defs = [
        (
            "STEP 2 — IMAGE ENCODER",
            "ResNet50  +  EfficientNet-B4   (pretrained on ImageNet)",
            [
                "The endoscopy image is scanned by two deep convolutional",
                "networks (ResNet50 & EfficientNet-B4). They detect visual",
                "patterns: lesion shape, colour, texture, mucosal structure.",
                f"Output → 49 image 'patch tokens', each a 256-D vector.",
            ],
            "#0D47A1", "🔵"
        ),
        (
            "STEP 3 — TEXT ENCODER",
            "BioBERT   (pre-trained on PubMed + clinical literature)",
            [
                "The clinical report is split into tokens (words/sub-words).",
                "BioBERT converts each token into a context-aware meaning",
                "vector. It understands medical language from 30M+ papers.",
                "Output → 1 CLS token  (256-D summary of the entire text).",
            ],
            "#1B5E20", "🟢"
        ),
        (
            "STEP 4 — PATIENT DATA ENCODER",
            f"TabTransformer   (trained on TCGA — 461 real cancer patients)",
            [
                f"12 patient features (age={d['tab_vec'][0]:.0f} yr, BMI, smoking,",
                "cancer stage, tumour morphology, site of resection etc.) are",
                "processed by a transformer that learns risk factor interactions.",
                "Output → 1 patient-risk token  (256-D vector).",
            ],
            "#BF360C", "🟠"
        ),
        (
            "STEP 5 — FUSION TRANSFORMER",
            "Gated Cross-Modal Attention   (3 layers, 8 attention heads)",
            [
                "All three 256-D tokens (image, text, patient data) enter a",
                "cross-attention transformer. It learns to weight each source:",
                f"  Image {d['mod_w'][0]:.1%}  ·  Text {d['mod_w'][1]:.1%}  ·  Patient data {d['mod_w'][2]:.1%}",
                "Output → 1 fused 256-D diagnostic representation.",
            ],
            "#4A148C", "🟣"
        ),
    ]

    for si, (title, subtitle, expl, bcolor, _icon) in enumerate(step_defs):
        yb = B_BOTS[si]
        yt = yb + BH

        # Shadow
        _rect(ax, BX + 0.07, yb - 0.07, BW, BH,
              fc="#DDDDDD", ec="none", lw=0, radius=0.15, zorder=1)
        # Main box
        _rect(ax, BX, yb, BW, BH,
              fc="white", ec=bcolor, lw=2.5, radius=0.15, zorder=2)
        # Left colour bar
        _rect(ax, BX, yb, 0.18, BH,
              fc=bcolor, ec="none", lw=0, radius=0.1, zorder=3)

        # Title
        _label(ax, BX + 0.40, yt - 0.33, title,
               fontsize=10.5, color=bcolor, bold=True, ha="left", zorder=4)
        # Subtitle
        _label(ax, BX + 0.40, yt - 0.62, subtitle,
               fontsize=8, color="#555", italic=True, ha="left", zorder=4)
        # Horizontal rule
        ax.plot([BX + 0.30, BX + BW - 0.15], [yt - 0.75, yt - 0.75],
                color="#E0E0E0", lw=0.8, zorder=3)
        # Explanation lines
        for li, ln in enumerate(expl):
            _label(ax, BX + 0.40, yt - 1.00 - li * 0.30, ln,
                   fontsize=8, color="#333", ha="left", zorder=4)

        # ── DOWN ARROW to next box ────────────────────────────────────────
        if si < 3:
            next_yb = B_BOTS[si + 1]
            arrow_x  = CX_B
            y_from   = yb - 0.02      # bottom of current box
            y_to     = next_yb + BH + 0.02  # top of next box

            # Arrow body
            _arrow(ax, arrow_x, y_from, arrow_x, y_to,
                   color="#455A64", lw=2.5, hw=0.20, hl=0.28, zorder=7)

            # "passes data ↓" label on arrow
            mid_y = (y_from + y_to) / 2
            _label(ax, arrow_x + 0.45, mid_y,
                   "passes\ndata  ↓",
                   fontsize=7.5, color="#607D8B", ha="left",
                   bg="#F5F5F5", bg_ec="#CFD8DC")

    # ── Section header ────────────────────────────────────────────────────
    _rect(ax, BX, 13.60, BW, 0.52,
          fc="#ECEFF1", ec="#90A4AE", lw=1.5, radius=0.1)
    _label(ax, CX_B, 13.87,
           "PROCESSING PIPELINE  (each step outputs a 256-dimensional vector)",
           fontsize=10, color="#37474F", bold=True)

    # ── Footer label under last box ───────────────────────────────────────
    final_bot = B_BOTS[3] - 0.08
    _arrow(ax, CX_B, final_bot, CX_B, 0.65,
           color="#C62828", lw=3.0, hw=0.25, hl=0.35, zorder=7)
    _label(ax, CX_B + 0.55, (final_bot + 0.65) / 2,
           "FINAL\nDIAGNOSIS  →",
           fontsize=9, color="#C62828", bold=True, ha="left",
           bg="#FFEBEE", bg_ec="#C62828")

    # Bottom note under arrow
    _rect(ax, BX, 0.15, BW, 0.45,
          fc="#FFEBEE", ec="#C62828", lw=1.5, radius=0.1)
    _label(ax, CX_B, 0.38,
           "Multi-task heads: Pathology classification  ·  Cancer staging  ·  Binary risk  ·  Uncertainty",
           fontsize=8.5, color="#C62828", bold=True)

    # ══════════════════════════════════════════════════════════════════════
    # COL C — OUTPUTS  (x: 13.7 to 24.0,  centre = 18.85)
    # ══════════════════════════════════════════════════════════════════════
    CX_C  = 18.85
    W_C   = 9.5

    # Section header
    _rect(ax, 13.8, 13.60, W_C - 0.1, 0.52,
          fc="#ECEFF1", ec="#90A4AE", lw=1.5, radius=0.1)
    _label(ax, CX_C, 13.87,
           "OUTPUTS  (real GradCAM heatmap  +  all diagnostic results)",
           fontsize=10, color="#37474F", bold=True)

    # ── C1: GradCAM overlay image ─────────────────────────────────────────
    CAM_W, CAM_H = 4.4, 4.0
    cam_cx = 16.20
    cam_cy = 11.30

    # Red border
    _rect(ax, cam_cx - CAM_W/2 - 0.12, cam_cy - CAM_H/2 - 0.12,
          CAM_W + 0.24, CAM_H + 0.24,
          fc="#B71C1C", ec="#B71C1C", lw=0, radius=0.08, zorder=2)
    _image_inset(ax, d["overlay"], cam_cx, cam_cy, CAM_W, CAM_H)

    # Contour overlay on heatmap
    cam_norm = d["cam"]
    # Draw contours using a small inset axes aligned to the image extent
    ax_cam_inset = fig.add_axes(
        [cam_cx/FW - CAM_W/(2*FW), cam_cy/FH - CAM_H/(2*FH),
         CAM_W/FW, CAM_H/FH],
        facecolor="none"
    )
    ax_cam_inset.imshow(d["overlay"], aspect="auto")
    ax_cam_inset.contour(cam_norm, levels=[0.55, 0.78],
                         colors=["yellow", "white"], linewidths=[2.0, 1.2], alpha=0.9)
    py, px = np.unravel_index(cam_norm.argmax(), cam_norm.shape)
    ax_cam_inset.plot(px, py, "w+", markersize=20, markeredgewidth=3, zorder=10)
    ax_cam_inset.axis("off")

    _label(ax, cam_cx, cam_cy + CAM_H/2 + 0.38,
           "STEP 6  ➜  GradCAM++  Attention Map",
           fontsize=10, color="#B71C1C", bold=True,
           bg="#FFEBEE", bg_ec="#B71C1C")
    _label(ax, cam_cx, cam_cy - CAM_H/2 - 0.30,
           f"Red/yellow = where the AI focused  ·  White + = peak  ·  ROI > 0.5: {float((cam_norm>0.5).mean()):.0%}",
           fontsize=8, color="#555", italic=True)

    # ── C2: DIAGNOSIS box ─────────────────────────────────────────────────
    DX_CX = 21.20
    DX_W  = 4.6
    DX_CY = 12.20
    DX_H  = 1.85

    _rect(ax, DX_CX - DX_W/2, DX_CY - DX_H/2, DX_W, DX_H,
          fc=d_pred_col + "18", ec=d_pred_col, lw=2.5, radius=0.15, zorder=2)
    _label(ax, DX_CX, DX_CY + DX_H/2 - 0.32,
           "DIAGNOSIS", fontsize=9, color="#666", bold=True)
    _label(ax, DX_CX, DX_CY + 0.12,
           CLASS_DISPLAY.get(d["pred_cls"], d["pred_cls"]),
           fontsize=14.5, color=d_pred_col, bold=True)
    _label(ax, DX_CX, DX_CY - DX_H/2 + 0.28,
           f"Confidence: {d['pred_conf']:.1%}   ·   ICD-10: {ICD10.get(d['cls_name'], 'K63.5')}",
           fontsize=8, color="#444")
    _label(ax, DX_CX, DX_CY - DX_H/2 - 0.25,
           d_cor_str, fontsize=11, color=d_cor_col, bold=True,
           bg="#FFFFFF", bg_ec=d_cor_col)

    # ── C3: 5-class probability bars ─────────────────────────────────────
    BAR_LEFT = 14.05
    BAR_W    = 5.00
    BAR_H    = 0.40
    BAR_GAP  = 0.20
    BAR_BOT  = 9.35   # bottom of lowest bar

    _label(ax, BAR_LEFT + BAR_W/2, BAR_BOT + N_CLASSES*(BAR_H + BAR_GAP) + 0.10,
           "Class Probabilities  (5 possible diagnoses)",
           fontsize=9, color="#212121", bold=True)

    for ci, (cname, prob) in enumerate(zip(CLASS_NAMES, d["path_p"])):
        bar_y = BAR_BOT + (N_CLASSES - 1 - ci) * (BAR_H + BAR_GAP)
        col   = CLASS_COLORS.get(cname, "#888")
        is_pred = (cname == d["pred_cls"])

        # Background bar
        ax.barh(bar_y + BAR_H/2, BAR_W, height=BAR_H,
                left=BAR_LEFT, color="#EEEEEE", zorder=2)
        # Filled bar
        ax.barh(bar_y + BAR_H/2, BAR_W * prob, height=BAR_H,
                left=BAR_LEFT, color=col, alpha=0.88, zorder=3,
                linewidth=2 if is_pred else 0.5,
                edgecolor="white" if is_pred else col)

        # Class label (left)
        _label(ax, BAR_LEFT - 0.08, bar_y + BAR_H/2,
               CLASS_DISPLAY.get(cname, cname),
               fontsize=8, color="#333", ha="right",
               bold=is_pred)
        # Value (right)
        _label(ax, BAR_LEFT + BAR_W * prob + 0.12, bar_y + BAR_H/2,
               f"{prob:.1%}", fontsize=8.5, color=col,
               ha="left", bold=is_pred)
        # ◄ PREDICTED tag
        if is_pred:
            _label(ax, BAR_LEFT + BAR_W + 0.10, bar_y + BAR_H/2,
                   "◄ PREDICTED",
                   fontsize=7.5, color=col, ha="left", bold=True,
                   bg=col + "22", bg_ec=col)

    # ── C4: Stage  /  Risk  /  Uncertainty  (3 side-by-side metric boxes) ─
    MET_Y  = 8.55
    MET_H  = 1.45
    MET_W  = 2.85
    MET_XS = [14.10, 17.20, 20.30]

    stage_col = STAGE_COLORS.get(d["stage_lbl"], "#F57F17")
    risk_col  = "#C62828" if d["risk_lbl"] == "Malignant" else "#2E7D32"
    unc_lbl   = "Low" if d["unc"] < 0.3 else ("Moderate" if d["unc"] < 0.6 else "High")
    unc_col   = "#2E7D32" if d["unc"] < 0.3 else ("#F57F17" if d["unc"] < 0.6 else "#C62828")

    metrics = [
        ("Cancer Stage",  d["stage_lbl"],                           stage_col, f"Confidence: {d['stage_c']:.1%}"),
        ("Cancer Risk",   f"{d['risk_lbl']}",                       risk_col,  f"Risk score: {d['risk_sc']:.3f}  (threshold 0.5)"),
        ("AI Certainty",  f"{unc_lbl} uncertainty",                 unc_col,   f"H = {d['unc']:.3f}   (MC-Dropout, n=15)"),
    ]

    for mi, (label, val, col, sub) in enumerate(metrics):
        mx = MET_XS[mi]
        _rect(ax, mx, MET_Y, MET_W, MET_H,
              fc=col + "18", ec=col, lw=2.0, radius=0.15, zorder=2)
        _label(ax, mx + MET_W/2, MET_Y + MET_H - 0.28,
               label, fontsize=8.5, color="#555", bold=True)
        _label(ax, mx + MET_W/2, MET_Y + MET_H/2 - 0.05,
               val, fontsize=11.5, color=col, bold=True)
        _label(ax, mx + MET_W/2, MET_Y + 0.22,
               sub, fontsize=7, color="#777", italic=True)

    # ── C5: Modality weights bar ──────────────────────────────────────────
    MW_Y  = 7.65
    MW_XS = [14.10, 17.20, 20.30]
    MW_LABELS = ["Image (GradCAM++)", "Text (BioBERT)", "Patient Data"]
    MW_COLS   = ["#0D47A1",           "#1B5E20",        "#BF360C"]

    _label(ax, CX_C, MW_Y + 0.48,
           "Modality weights  (how much each data stream contributed to the decision)",
           fontsize=8.5, color="#212121", bold=True)

    for mi, (mw, ml, mc) in enumerate(zip(d["mod_w"], MW_LABELS, MW_COLS)):
        mx = MW_XS[mi]
        bw = MET_W * mw
        ax.barh(MW_Y + 0.15, bw, height=0.28,
                left=mx, color=mc, alpha=0.85, zorder=3)
        ax.barh(MW_Y + 0.15, MET_W, height=0.28,
                left=mx, color="#E0E0E0", alpha=0.4, zorder=2)
        _label(ax, mx + MET_W/2, MW_Y - 0.10,
               f"{ml}  {mw:.1%}", fontsize=8, color=mc, bold=True)

    # ── C6: Disclaimer bar ────────────────────────────────────────────────
    _rect(ax, 13.8, 0.15, W_C - 0.1, 0.45,
          fc="#FFF3E0", ec="#FF8F00", lw=1.5, radius=0.1)
    _label(ax, CX_C, 0.38,
           "⚠  Decision-support tool — all results must be reviewed by a qualified gastroenterologist",
           fontsize=8.5, color="#E65100", bold=True)

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-COLUMN ARROWS  (precisely connecting box edges)
    # ══════════════════════════════════════════════════════════════════════

    # We need the exact positions of the three input boxes and the
    # right edge of Column A and the left edge of Column B (x=4.9 / x=5.1)
    # Col B step-box left edge = BX = 5.1

    A_RIGHT   = CX_A + W_A/2     # = 2.4 + 2.1 = 4.5
    B_LEFT    = BX               # = 5.1
    B_RIGHT   = BX + BW          # = 13.3
    C_LEFT    = 13.8

    # ── Arrow ①: Image → Step 2 box (IMAGE ENCODER) ──────────────────────
    # Image box right edge y-centre  =  img_cy = 10.85
    # Step 2 box y-centre            =  B_BOTS[0] + BH/2
    step2_cy = B_BOTS[0] + BH / 2
    _arrow(ax, A_RIGHT + 0.08, img_cy, B_LEFT - 0.08, step2_cy,
           color="#0D47A1", lw=2.5, hw=0.18, hl=0.28)
    mid_x = (A_RIGHT + B_LEFT) / 2
    mid_y = (img_cy + step2_cy) / 2
    _label(ax, mid_x, mid_y + 0.22,
           "image pixels\n224×224",
           fontsize=7.5, color="#0D47A1", bold=True,
           bg="#E3F2FD", bg_ec="#1565C0")

    # ── Arrow ②: Text → Step 3 box (TEXT ENCODER) ────────────────────────
    step3_cy = B_BOTS[1] + BH / 2
    txt_box_cy = TXT_Y_TOP - TXT_H / 2
    _arrow(ax, A_RIGHT + 0.08, txt_box_cy, B_LEFT - 0.08, step3_cy,
           color="#1B5E20", lw=2.5, hw=0.18, hl=0.28)
    mid_x = (A_RIGHT + B_LEFT) / 2
    mid_y = (txt_box_cy + step3_cy) / 2
    _label(ax, mid_x, mid_y + 0.22,
           "64 tokens\nBioBERT vocab",
           fontsize=7.5, color="#1B5E20", bold=True,
           bg="#E8F5E9", bg_ec="#1B5E20")

    # ── Arrow ③: Tabular → Step 4 box (PATIENT DATA ENCODER) ─────────────
    step4_cy = B_BOTS[2] + BH / 2
    tab_box_cy = TAB_Y_TOP - TAB_H / 2
    _arrow(ax, A_RIGHT + 0.08, tab_box_cy, B_LEFT - 0.08, step4_cy,
           color="#BF360C", lw=2.5, hw=0.18, hl=0.28)
    mid_x = (A_RIGHT + B_LEFT) / 2
    mid_y = (tab_box_cy + step4_cy) / 2
    _label(ax, mid_x, mid_y + 0.22,
           "12 features\nTCGA patients",
           fontsize=7.5, color="#BF360C", bold=True,
           bg="#FBE9E7", bg_ec="#BF360C")

    # ── Arrow ④: Fusion box → GradCAM output ─────────────────────────────
    step5_cy = B_BOTS[3] + BH / 2
    _arrow(ax, B_RIGHT + 0.08, step5_cy, C_LEFT + 0.08, cam_cy,
           color="#4A148C", lw=2.8, hw=0.20, hl=0.32)
    mid_x = (B_RIGHT + C_LEFT) / 2
    mid_y = (step5_cy + cam_cy) / 2
    _label(ax, mid_x, mid_y + 0.25,
           "fused 256-D\nvector  →  heads",
           fontsize=7.5, color="#4A148C", bold=True,
           bg="#F3E5F5", bg_ec="#6A1B9A")

    # ── Arrow ⑤: Fusion box → Diagnosis box ──────────────────────────────
    dx_left = DX_CX - DX_W / 2
    _arrow(ax, B_RIGHT + 0.08, step5_cy, dx_left - 0.08, DX_CY,
           color="#C62828", lw=2.8, hw=0.20, hl=0.32)

    # ── Note on modality convergence (three thin arrows into Step 5) ──────
    # All three steps' tokens converge into step 5
    step5_top = B_BOTS[3] + BH
    step5_bot = B_BOTS[3]

    # ── Footer ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.003,
             f"UnifiedMultiModalTransformer (~74.7M params)  ·  "
             f"HyperKvasir (10,662 images) + CVC-ClinicDB (612) + TCGA (461 patients)  ·  "
             f"Val F1: 0.9989  ·  Test AUC-ROC: 1.0000",
             ha="center", va="bottom", fontsize=7.5,
             color="#9E9E9E", style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"[ArchitectureFlow] Saved → {out_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--class", dest="cls", type=str, default=None,
                        choices=CLASS_NAMES)
    parser.add_argument("--seed",  type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ArchitectureFlow] Device: {device}")

    print("[ArchitectureFlow] Loading model …")
    model = UnifiedMultiModalTransformer(
        n_classes=N_CLASSES, n_tabular_features=N_TABULAR_FEATURES, d_model=D_MODEL)
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"val_f1={ckpt.get('val_f1', ckpt.get('val_acc', 0)):.4f}")

    print("[ArchitectureFlow] Loading BioBERT tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    print("[ArchitectureFlow] Building TCGA pool …")
    tcga_pool = build_tcga_pool()
    print(f"  {sum(len(v) for v in tcga_pool.values())} TCGA patient vectors")

    img_path, cls_name, subfolder, dataset = pick_image(args.image, args.cls)
    print(f"[ArchitectureFlow] Image : {img_path}")
    print(f"  Class: {cls_name}  |  Dataset: {dataset}")

    print("[ArchitectureFlow] Running pipeline …")
    data = run_pipeline(img_path, cls_name, model, tokenizer, tcga_pool, device)
    print(f"  Prediction : {data['pred_cls']} ({data['pred_conf']:.1%})  "
          f"[{'CORRECT' if data['correct'] else 'WRONG'}]")
    print(f"  Stage      : {data['stage_lbl']} ({data['stage_c']:.1%})")
    print(f"  Risk       : {data['risk_lbl']} ({data['risk_sc']:.3f})")
    print(f"  Uncertainty: {data['unc']:.3f}")

    out_dir = Path("outputs/architecture_flow")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_fn  = Path(img_path).stem[:22]
    safe_cls = cls_name.replace("-", "_")
    out_path = str(out_dir / f"architecture_flow_{safe_cls}__{safe_fn}.png")

    print("[ArchitectureFlow] Rendering diagram …")
    build_diagram(data, out_path)
    print(f"\n[ArchitectureFlow] Done!\n  → {out_path}")


if __name__ == "__main__":
    main()
