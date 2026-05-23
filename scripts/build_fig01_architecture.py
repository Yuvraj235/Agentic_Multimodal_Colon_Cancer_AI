"""Generate fig01_architecture.png — coloured ColonAI architecture diagram.

The previous fig01 was a B/W flowchart that the project guide flagged as
"simple, poor visibility". This script produces a high-contrast, colour-coded,
publication-quality version that mirrors the exact block layout of the
architecture (image / text / tabular branches → gated cross-modal fusion
→ three task heads → 6-agent pipeline).

Run:  python3 scripts/build_fig01_architecture.py
Out:  paper_figures/fig01_architecture.png  (DPI 200, ~1600 x 1200 px)
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Output -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "paper_figures" / "fig01_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Colour palette ---------------------------------------------------------
C_IMG    = "#2563EB"   # blue   – image branch
C_TXT    = "#16A34A"   # green  – text branch
C_TAB    = "#EA580C"   # orange – tabular branch
C_FUSION = "#7C3AED"   # purple – fusion transformer
C_HEAD_P = "#A855F7"   # violet – pathology head
C_HEAD_S = "#DC2626"   # red    – staging head
C_HEAD_R = "#F97316"   # orange-red – risk head
C_XAI    = "#0D9488"   # teal   – XAI / explanation row
C_BANNER = "#1E40AF"   # deep blue – top transfer-learning banner
C_BORDER = "white"
C_TEXT   = "white"


def block(ax, x, y, w, h, label, fill, text_size=10, lw=1.6, corner="round,pad=0.02"):
    """Draw one rounded coloured block with white text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=corner,
        linewidth=lw,
        edgecolor=C_BORDER, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=text_size, fontweight="bold",
            color=C_TEXT, family="DejaVu Sans")


def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.4, style="-|>",
          head_w=8, head_l=12):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw, color=color,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


# ── Figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15.0, 11.0), dpi=200)
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 100)
ax.set_aspect("auto")
ax.axis("off")

# Title
fig.text(0.50, 0.965,
         "Unified Multi-Modal Transformer v2 — Colon Cancer Detection & Staging",
         ha="center", va="top", fontsize=14, fontweight="bold", color="#0F172A")
fig.text(0.50, 0.940,
         "ResNet50 + EfficientNet-B0  |  BioBERT  |  TabTransformer  →  "
         "Gated Cross-Modal Fusion",
         ha="center", va="top", fontsize=11, color="#334155")

# Top banner — transfer-learning pipeline
block(ax, 2, 89, 96, 5,
      "TRANSFER LEARNING:  Stage-1 CVC-ClinicDB Polyp Pretrain  →  "
      "Stage-2 HyperKvasir Multimodal Finetune  →  Stage-3 TCGA Tabular+Text Fusion",
      C_BANNER, text_size=9)

# ── Row 1: input sources ──────────────────────────────────────────────────
y_in = 77
block(ax,  2, y_in, 30, 8,
      "Colonoscopy Image\n(224×224×3 RGB)\nHyperKvasir + CVC",   C_IMG, 9.5)
block(ax, 34, y_in, 30, 8,
      "Clinical Text\n(BioBERT-tokenised, ≤64 tokens)\nSymptom-templated input",
      C_TXT, 9.5)
block(ax, 66, y_in, 32, 8,
      "Patient Tabular (12 features)\nAge · BMI · Stage · Morphology\nTCGA-COAD clinical",
      C_TAB, 9.5)

# ── Row 2: backbones ──────────────────────────────────────────────────────
y_bb = 60
block(ax,  2, y_bb, 14, 9,
      "ResNet50\n(ImageNet+CVC)\nlayer4 → 7×7\n2048-dim",        C_IMG, 8.5)
block(ax, 18, y_bb, 14, 9,
      "EfficientNet-B0\n(ImageNet+CVC)\nblock6 → 7×7\n112-dim",  C_IMG, 8.5)
block(ax, 34, y_bb, 30, 9,
      "BioBERT  (dmis-lab v1.2)\nTop-2 layers fine-tuned\n[CLS] → 256-dim projection",
      C_TXT, 9)
block(ax, 66, y_bb, 32, 9,
      "TabTransformer  (4-layer · 128-dim)\nper-feature column embed → mean-pool\n"
      "→ 256-dim projection",
      C_TAB, 9)

# ── Row 3: image-branch projection ────────────────────────────────────────
y_pj = 47
block(ax,  2, y_pj, 30, 7,
      "Learned Per-Position Spatial Gate\nConcat → Project (d = 256)",
      C_IMG, 9.5)

# small caption row above fusion (token counts)
ax.text(17, 44.5, "Img tokens (49)", ha="center", va="center",
        fontsize=8, color="#475569", fontstyle="italic")
ax.text(49, 44.5, "Txt token (1)",  ha="center", va="center",
        fontsize=8, color="#475569", fontstyle="italic")
ax.text(82, 44.5, "Tab token (1)",  ha="center", va="center",
        fontsize=8, color="#475569", fontstyle="italic")

# ── Row 4: fusion transformer ─────────────────────────────────────────────
y_fu = 31
block(ax,  2, y_fu, 96, 11,
      "Gated Cross-Modal Fusion Transformer  (d_model = 256 · 8 heads)\n"
      "Stage A: per-modality self-attention   ·   "
      "Stage B: 3× bidirectional gated cross-attention (Image↔Text↔Tab)\n"
      "Stage C: shared bottleneck self-attention  +  Learnable CLS token\n"
      "Learned Modality Importance Gates:  σ(W·img) · σ(W·txt) · σ(W·tab)  →  "
      "softmax-normalised weights",
      C_FUSION, text_size=9)

# ── Row 5: three task heads ───────────────────────────────────────────────
y_hd = 17
block(ax,  2, y_hd, 30, 8,
      "Pathology Head  (5-class)\nPolyps · UC mild · UC mod-sev\n"
      "Barrett's · Therapeutic",
      C_HEAD_P, 9)
block(ax, 34, y_hd, 30, 8,
      "Staging Head  (4-class)\nNo Cancer · Stage I · II · III/IV",
      C_HEAD_S, 9)
block(ax, 66, y_hd, 32, 8,
      "Risk Head  (binary)\nBenign vs Malignant\nP(malignant) confidence",
      C_HEAD_R, 9)

# ── Row 6: XAI / explanation row ──────────────────────────────────────────
y_xa = 4
block(ax,  2, y_xa, 30, 8,
      "GradCAM++ XAI Agent\nResNet50 layer4 + EfficientNet-B0\nSpatial saliency",
      C_XAI, 9)
block(ax, 34, y_xa, 30, 8,
      "BioBERT Attention Agent\nToken attention rollout\nClinical keyword emphasis",
      C_XAI, 9)
block(ax, 66, y_xa, 32, 8,
      "SHAP + MC-Dropout Agent\nTabular feature importance\n"
      "Uncertainty (T = 15)",
      C_XAI, 9)

# ── Arrows ────────────────────────────────────────────────────────────────
# inputs → backbones
arrow(ax, 17, y_in,    9,  y_bb + 9)
arrow(ax, 17, y_in,    25, y_bb + 9)
arrow(ax, 49, y_in,    49, y_bb + 9)
arrow(ax, 82, y_in,    82, y_bb + 9)

# backbones → image-projection
arrow(ax,  9, y_bb,   12, y_pj + 7)
arrow(ax, 25, y_bb,   22, y_pj + 7)

# branches → fusion
arrow(ax, 17, y_pj,   17, y_fu + 11)
arrow(ax, 49, y_bb,   49, y_fu + 11)
arrow(ax, 82, y_bb,   82, y_fu + 11)

# fusion → heads
arrow(ax, 17, y_fu,   17, y_hd + 8)
arrow(ax, 49, y_fu,   49, y_hd + 8)
arrow(ax, 82, y_fu,   82, y_hd + 8)

# heads → XAI agents
arrow(ax, 17, y_hd,   17, y_xa + 8)
arrow(ax, 49, y_hd,   49, y_xa + 8)
arrow(ax, 82, y_hd,   82, y_xa + 8)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
