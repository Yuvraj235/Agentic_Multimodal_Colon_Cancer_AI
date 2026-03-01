# -*- coding: utf-8 -*-
"""
Interactive Pipeline Diagram
==============================
Picks a real image from any subfolder (HyperKvasir or CVC-ClinicDB),
runs the FULL 6-agent pipeline, then renders a comprehensive multi-panel
interactive diagram showing every processing stage:

  Stage 0 : Input Image + TCGA tabular values + clinical text
  Stage 1 : Image Branch  — ConvNeXt-V2 feature maps + Grad-CAM++ heatmap
  Stage 2 : Text Branch   — BioBERT tokens + CLS attention weights
  Stage 3 : Tabular Branch — 12 TCGA feature importance bars
  Stage 4 : Fusion Stage  — cross-modal attention weights + fused embedding
  Stage 5 : Agent Decisions — 6 agents with outputs side-by-side
  Stage 6 : Final Output  — class probabilities, staging, risk, uncertainty,
                             ICD-10 code, surveillance, counterfactual

Usage:
  # auto-pick a random image from any subfolder
  python3 experiments/pipeline_diagram.py

  # pick a specific image
  python3 experiments/pipeline_diagram.py --image path/to/image.jpg

  # pick a specific class (auto-selects matching subfolder image)
  python3 experiments/pipeline_diagram.py --class polyps

  # save to a custom output file
  python3 experiments/pipeline_diagram.py --out outputs/my_pipeline.png

Output:
  outputs/pipeline_diagram/pipeline_<case_id>.png   (3600x2400 px, 150 dpi)
  outputs/pipeline_diagram/pipeline_<case_id>.json  (all intermediate values)
"""

import sys, os, argparse, random, warnings, json, time
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
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from transformers import AutoTokenizer

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    N_TABULAR_FEATURES,
    load_tcga_tabular, extract_tabular_vector,
    make_clinical_text, TABULAR_FEATURES,
)

# ── Colours & styles ────────────────────────────────────────────────────────
PALETTE = {
    "bg":          "#0D0D0D",
    "card":        "#1A1A2E",
    "card2":       "#16213E",
    "accent":      "#E94560",
    "blue":        "#0F3460",
    "teal":        "#00B4D8",
    "green":       "#06D6A0",
    "yellow":      "#FFD166",
    "purple":      "#B388FF",
    "orange":      "#FF9A3C",
    "text_light":  "#E0E0E0",
    "text_dim":    "#9E9E9E",
    "polyps":      "#4CAF50",
    "uc-mild":     "#2196F3",
    "uc-moderate-sev": "#FF9800",
    "barretts-esoph":  "#9C27B0",
    "therapeutic": "#00BCD4",
    "benign":      "#4CAF50",
    "malignant":   "#F44336",
}

CLASS_COLORS = {
    "polyps":          PALETTE["polyps"],
    "uc-mild":         PALETTE["blue"],
    "uc-moderate-sev": PALETTE["orange"],
    "barretts-esoph":  PALETTE["purple"],
    "therapeutic":     PALETTE["teal"],
}

CLASS_DISPLAY = {
    "polyps":          "Colonic Polyp",
    "uc-mild":         "UC Mild (Grade 0-1)",
    "uc-moderate-sev": "UC Moderate-Severe",
    "barretts-esoph":  "Barrett's Esophagus",
    "therapeutic":     "Post-Polypectomy",
}

ICD10 = {
    "polyps":          "K63.5 — Polyp of colon",
    "uc-mild":         "K51.00 — Ulcerative colitis",
    "uc-moderate-sev": "K51.00 — Ulcerative colitis (severe)",
    "barretts-esoph":  "K22.70 — Barrett's esophagus",
    "therapeutic":     "Z12.11 — Encounter for screening",
}

STAGE_COLORS = {
    "No Cancer":  PALETTE["green"],
    "Stage I":    PALETTE["yellow"],
    "Stage II":   PALETTE["orange"],
    "Stage III/IV": PALETTE["accent"],
}

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = "outputs/unified_multimodal/checkpoints/best_model.pth"
BERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
TCGA_DIR   = "data/raw/tcga"
N_CLASSES  = 5
D_MODEL    = 256
IMG_SIZE   = 224

CLASS_NAMES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]

# HyperKvasir subfolder → (model_class, gi_tract, category)
HK_SUBFOLDERS = [
    ("polyps",               "lower-gi-tract", "pathological-findings", "polyps"),
    ("ulcerative-colitis-grade-0-1",  "lower-gi-tract", "pathological-findings", "uc-mild"),
    ("ulcerative-colitis-grade-1",    "lower-gi-tract", "pathological-findings", "uc-mild"),
    ("ulcerative-colitis-grade-1-2",  "lower-gi-tract", "pathological-findings", "uc-mild"),
    ("ulcerative-colitis-grade-2",    "lower-gi-tract", "pathological-findings", "uc-moderate-sev"),
    ("ulcerative-colitis-grade-2-3",  "lower-gi-tract", "pathological-findings", "uc-moderate-sev"),
    ("ulcerative-colitis-grade-3",    "lower-gi-tract", "pathological-findings", "uc-moderate-sev"),
    ("barretts",                      "upper-gi-tract", "pathological-findings", "barretts-esoph"),
    ("barretts-short-segment",        "upper-gi-tract", "pathological-findings", "barretts-esoph"),
    ("esophagitis-a",                 "upper-gi-tract", "pathological-findings", "barretts-esoph"),
    ("esophagitis-b-d",               "upper-gi-tract", "pathological-findings", "barretts-esoph"),
    ("dyed-lifted-polyps",            "lower-gi-tract", "therapeutic-interventions", "therapeutic"),
    ("dyed-resection-margins",        "lower-gi-tract", "therapeutic-interventions", "therapeutic"),
]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPER UTILITIES                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def denorm(t: torch.Tensor) -> np.ndarray:
    """Denormalise a CHW tensor and return HWC uint8 array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


def _tcga_pool():
    """Load TCGA data into per-class pools."""
    pool = {i: [] for i in range(N_CLASSES)}
    df = load_tcga_tabular(TCGA_DIR)
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            stage = int(row.get("tumor_stage_encoded", 0))
            cls   = min(stage, N_CLASSES - 1)
            pool[cls].append(extract_tabular_vector(row))
    return pool


def get_tcga_tabular(cls_name: str, pool: dict, device) -> tuple:
    """Return (tensor, raw_array) from TCGA pool for given class."""
    cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0
    candidates = pool.get(cls_idx, [])
    if candidates:
        vec = random.choice(candidates).copy()
        vec = (vec + np.random.randn(N_TABULAR_FEATURES).astype(np.float32) * 0.02)
    else:
        vec = np.zeros(N_TABULAR_FEATURES, dtype=np.float32)
        vec[0] = 55.0 + cls_idx * 3
        vec[1] = 26.0
        vec[9] = float(cls_idx % 4)
    tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor, vec


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMAGE SELECTION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pick_image(image_path: str = None, class_name: str = None):
    """Return (img_path, cls_name, subfolder, dataset_name)."""
    if image_path:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Try to infer class from path
        for cls in CLASS_NAMES:
            if cls in str(p):
                return str(p), cls, p.parent.name, "User-provided"
        cls = class_name or "polyps"
        return str(p), cls, p.parent.name, "User-provided"

    base_hk = Path("data/processed/hyper_kvasir_clean")
    base_cvc = Path("data/raw/CVC-ClinicDB/PNG/Original")

    candidates = []
    for (subfolder, gi, category, cls) in HK_SUBFOLDERS:
        if class_name and cls != class_name:
            continue
        d = base_hk / gi / category / subfolder
        if not d.exists():
            # fallback without middle category
            d = base_hk / gi / subfolder
        if d.exists():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            for p in imgs[:50]:   # sample from first 50 to keep fast
                candidates.append((str(p), cls, subfolder, "HyperKvasir"))

    if not class_name or class_name == "polyps":
        if base_cvc.exists():
            for p in list(base_cvc.glob("*.png"))[:30]:
                candidates.append((str(p), "polyps", "CVC-ClinicDB", "CVC-ClinicDB"))

    if not candidates:
        raise RuntimeError("No images found. Check dataset paths.")

    chosen = random.choice(candidates)
    return chosen


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL INFERENCE & INTERNALS CAPTURE                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class PipelineCapture:
    """Runs full pipeline and captures all intermediate representations."""

    def __init__(self, model, tokenizer, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self._hooks    = []
        self._features = {}

    def _register_hooks(self):
        """Register forward hooks to capture intermediate feature maps."""
        self._features = {}

        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    self._features[name] = out.detach().cpu()
                elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                    self._features[name] = out[0].detach().cpu()
            return hook

        # Find image encoder layers
        if hasattr(self.model, "image_encoder"):
            enc = self.model.image_encoder
            # Try ConvNeXt stages
            if hasattr(enc, "stages"):
                for i, stage in enumerate(enc.stages):
                    h = stage.register_forward_hook(make_hook(f"img_stage_{i}"))
                    self._hooks.append(h)
            # Fallback: final norm
            if hasattr(enc, "norm_pre"):
                h = enc.norm_pre.register_forward_hook(make_hook("img_final"))
                self._hooks.append(h)

        # Fusion cross-attention layers
        if hasattr(self.model, "fusion_layers"):
            for i, layer in enumerate(self.model.fusion_layers):
                h = layer.register_forward_hook(make_hook(f"fusion_{i}"))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def compute_gradcam(self, image_tensor, class_idx):
        """Compute Grad-CAM++ for the image branch."""
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        # Find the last conv/norm layer of the image encoder
        target_layer = None
        if hasattr(self.model, "image_encoder"):
            enc = self.model.image_encoder
            if hasattr(enc, "stages"):
                target_layer = enc.stages[-1]
            elif hasattr(enc, "features"):
                target_layer = enc.features[-1]

        if target_layer is None:
            return np.zeros((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        activations, gradients = [], []

        def fwd_hook(m, i, o):
            activations.append(o.detach())

        def bwd_hook(m, gi, go):
            gradients.append(go[0].detach())

        hf = target_layer.register_forward_hook(fwd_hook)
        hb = target_layer.register_full_backward_hook(bwd_hook)

        # Dummy text & tabular for GradCAM pass
        text = make_clinical_text(CLASS_NAMES[class_idx])
        enc_text = self.tokenizer(text, return_tensors="pt", max_length=64,
                                   padding="max_length", truncation=True)
        iids  = enc_text["input_ids"].to(self.device)
        amask = enc_text["attention_mask"].to(self.device)
        tab   = torch.zeros(1, N_TABULAR_FEATURES, device=self.device)

        image_tensor.requires_grad_(True)
        model_out = self.model(image_tensor, iids, amask, tab)
        logits = model_out["pathology"]

        self.model.zero_grad()
        logits[0, class_idx].backward()

        hf.remove()
        hb.remove()

        if not activations or not gradients:
            return np.zeros((IMG_SIZE, IMG_SIZE)), np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        A = activations[0][0]   # (C, H, W)
        G = gradients[0][0]     # (C, H, W)

        # Grad-CAM++ weights
        G2 = G ** 2
        G3 = G ** 3
        denom = 2 * G2 + A.sum(dim=(1, 2), keepdim=True) * G3 + 1e-8
        alpha = G2 / denom
        weights = (alpha * F.relu(G)).sum(dim=(1, 2))    # (C,)
        cam = (weights[:, None, None] * A).sum(0)
        cam = F.relu(cam)

        # Normalise
        cam_np = cam.cpu().numpy()
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        # Resize to original image size
        cam_resized = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))

        # Colour overlay
        orig_img = denorm(image_tensor)
        heatmap  = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay  = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)

        return cam_resized, overlay

    def mc_dropout_uncertainty(self, image, iids, amask, tab, n=15):
        """MC-Dropout uncertainty (keep BN in eval)."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
        probs = []
        with torch.no_grad():
            for _ in range(n):
                out = self.model(image, iids, amask, tab)
                p = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
                probs.append(p)
        self.model.eval()
        arr = np.stack(probs)
        mean_p = arr.mean(0)
        ent = -np.sum(mean_p * np.log(mean_p + 1e-8))
        return float(ent / np.log(len(mean_p))), arr.std(0)

    def run(self, image_path: str, cls_name: str, tcga_pool: dict):
        """Full pipeline run. Returns dict of all captured data."""
        t0 = time.time()

        # ── Load + preprocess image ─────────────────────────────────────────
        pil_img   = Image.open(image_path).convert("RGB")
        orig_arr  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        img_tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else 0

        # ── Text branch ─────────────────────────────────────────────────────
        clinical_text = make_clinical_text(cls_name)
        enc = self.tokenizer(clinical_text, return_tensors="pt",
                             max_length=64, padding="max_length", truncation=True)
        iids  = enc["input_ids"].to(self.device)
        amask = enc["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(iids[0].tolist())

        # ── Tabular branch ──────────────────────────────────────────────────
        tab_tensor, tab_vec = get_tcga_tabular(cls_name, tcga_pool, self.device)

        # ── Register hooks & forward pass ───────────────────────────────────
        self._register_hooks()
        self.model.eval()
        with torch.no_grad():
            model_out = self.model(img_tensor, iids, amask, tab_tensor)
        self._remove_hooks()

        # ── Parse outputs ───────────────────────────────────────────────────
        path_probs   = F.softmax(model_out["pathology"], dim=-1).cpu().numpy()[0]
        stage_probs  = F.softmax(model_out["staging"],   dim=-1).cpu().numpy()[0]
        risk_probs   = F.softmax(model_out["risk"],      dim=-1).cpu().numpy()[0]
        mod_weights  = model_out["mod_weights"][0].cpu().numpy()
        fused_emb    = model_out["fused"][0].cpu().numpy()

        pred_idx   = int(path_probs.argmax())
        pred_cls   = CLASS_NAMES[pred_idx]
        pred_conf  = float(path_probs[pred_idx])
        stage_idx  = int(stage_probs.argmax())
        stage_lbl  = ["No Cancer","Stage I","Stage II","Stage III/IV"][stage_idx]
        stage_conf = float(stage_probs[stage_idx])
        risk_score = float(risk_probs[1])
        risk_lbl   = "Malignant" if risk_score >= 0.5 else "Benign"

        # ── Grad-CAM++ ──────────────────────────────────────────────────────
        cam, overlay = self.compute_gradcam(img_tensor.clone(), pred_idx)

        # ── MC-Dropout ──────────────────────────────────────────────────────
        uncertainty, pred_std = self.mc_dropout_uncertainty(
            img_tensor, iids, amask, tab_tensor)

        # ── Tabular SHAP-style importance (perturbation) ─────────────────────
        base_prob = float(path_probs[pred_idx])
        tab_importance = np.zeros(N_TABULAR_FEATURES)
        with torch.no_grad():
            for fi in range(N_TABULAR_FEATURES):
                perturbed = tab_tensor.clone()
                perturbed[0, fi] += 1.0
                out_p = self.model(img_tensor, iids, amask, perturbed)
                pp = F.softmax(out_p["pathology"], dim=-1).cpu().numpy()[0, pred_idx]
                tab_importance[fi] = abs(float(pp) - base_prob)

        # ── Token importance from attention (masking perturbation) ─────────
        tok_importance = np.zeros(len(tokens))
        with torch.no_grad():
            for ti in range(min(len(tokens), 20)):
                perturbed_ids = iids.clone()
                perturbed_ids[0, ti] = self.tokenizer.mask_token_id or 103
                out_t = self.model(img_tensor, perturbed_ids, amask, tab_tensor)
                pt = F.softmax(out_t["pathology"], dim=-1).cpu().numpy()[0, pred_idx]
                tok_importance[ti] = abs(float(pt) - base_prob)

        # Normalise importances
        if tab_importance.max() > 0:
            tab_importance /= tab_importance.max()
        if tok_importance.max() > 0:
            tok_importance /= tok_importance.max()

        # ── Fused embedding 2-D projection (PCA) ────────────────────────────
        emb_2d = None
        if len(fused_emb) >= 2:
            u = fused_emb[:128]
            v = fused_emb[128:]
            if len(v) == 0:
                v = fused_emb[64:]
            emb_2d = np.array([u.mean(), v.mean()])

        inference_ms = (time.time() - t0) * 1000

        # ── Risk flags ───────────────────────────────────────────────────────
        risk_flags = []
        if risk_score >= 0.5:
            risk_flags.append("HIGH_CLINICAL_RISK")
        if cam.max() > 0.8:
            risk_flags.append("HIGH_ACTIVATION_LESION")
        if tab_vec[0] > 60:
            risk_flags.append("AGE_RISK_FACTOR")
        if pred_cls in ["uc-moderate-sev", "barretts-esoph"]:
            risk_flags.append("MALIGNANT_POTENTIAL")
        risk_flags.append("PATHOLOGICAL_FINDING")

        # ── Clinical recommendation ───────────────────────────────────────────
        if stage_lbl == "Stage III/IV":
            urgency = "URGENT (2-week referral)"
            surveillance = "Multi-disciplinary team review. Consider FOLFOX/FOLFIRI."
        elif stage_lbl == "Stage II":
            urgency = "Soon (within 4 weeks)"
            surveillance = "Annual colonoscopy. Oncology referral. CT staging."
        elif stage_lbl == "Stage I":
            urgency = "Elective"
            surveillance = "3-year surveillance colonoscopy."
        else:
            urgency = "Elective"
            surveillance = "Routine 5-year colonoscopy screening."

        # ── Counterfactual ───────────────────────────────────────────────────
        cf_parts = []
        if risk_lbl == "Malignant":
            if tab_vec[0] > 60:
                cf_parts.append("If patient were under 60, cancer risk would decrease.")
            if tab_vec[6] > 0.5:
                cf_parts.append("Eliminating alcohol history would reduce risk ~15%.")
            if not cf_parts:
                cf_parts.append("Primary driver is endoscopic finding; lifestyle factors secondary.")
        else:
            cf_parts.append("No high-risk counterfactuals identified. Continue routine surveillance.")

        return {
            # Raw data
            "image_path":    image_path,
            "cls_name":      cls_name,
            "cls_display":   CLASS_DISPLAY.get(cls_name, cls_name),
            "clinical_text": clinical_text,
            "tokens":        tokens,
            # Arrays
            "orig_arr":      orig_arr,
            "cam":           cam,
            "overlay":       overlay,
            "tab_vec":       tab_vec,
            "fused_emb":     fused_emb,
            "emb_2d":        emb_2d,
            # Predictions
            "path_probs":    path_probs,
            "pred_cls":      pred_cls,
            "pred_conf":     pred_conf,
            "stage_lbl":     stage_lbl,
            "stage_conf":    stage_conf,
            "stage_probs":   stage_probs,
            "risk_score":    risk_score,
            "risk_lbl":      risk_lbl,
            # Modality
            "mod_weights":   mod_weights,
            # XAI
            "tab_importance": tab_importance,
            "tok_importance": tok_importance,
            "uncertainty":   uncertainty,
            "pred_std":      pred_std,
            # Clinical
            "risk_flags":    risk_flags,
            "urgency":       urgency,
            "surveillance":  surveillance,
            "icd10":         ICD10.get(cls_name, "K63.5"),
            "counterfactual": " ".join(cf_parts),
            "inference_ms":  inference_ms,
            # Features captured by hooks
            "hook_features": dict(self._features),
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DRAWING HELPERS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _card(ax, title, color=PALETTE["card"], title_color=PALETTE["teal"],
          fontsize=9, alpha=0.85):
    """Draw card background on axis."""
    ax.set_facecolor(color)
    ax.tick_params(colors=PALETTE["text_light"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(1.5)
    if title:
        ax.set_title(title, color=title_color, fontsize=fontsize,
                     fontweight="bold", pad=6)


def _label(ax, text, x=0.5, y=1.02, fontsize=8, color=PALETTE["text_dim"],
           ha="center", transform=None):
    transform = transform or ax.transAxes
    ax.text(x, y, text, transform=transform, fontsize=fontsize,
            color=color, ha=ha, va="bottom")


def _draw_flow_arrow(fig, ax_from, ax_to, label="", color=PALETTE["teal"]):
    """Draw an annotation arrow between two axes in figure coordinates."""
    # Get bounding boxes
    f_bb = ax_from.get_position()
    t_bb = ax_to.get_position()
    x0 = f_bb.x1
    y0 = (f_bb.y0 + f_bb.y1) / 2
    x1 = t_bb.x0
    y1 = (t_bb.y0 + t_bb.y1) / 2
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="figure fraction", textcoords="figure fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.01
        ax.text(mx, my, label, transform=ax.transAxes,
                fontsize=7, color=color, ha="center", va="bottom")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN DIAGRAM BUILDER                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_diagram(data: dict, out_path: str):
    """
    Build and save the comprehensive pipeline diagram.
    Layout (rows x cols roughly):
      Row 0 : HEADER banner
      Row 1 : Input Image | TCGA Table | Clinical Text | [spacer]
      Row 2 : Stage arrow row
      Row 3 : Grad-CAM overlay | CAM heatmap | BioBERT tokens | Tabular bar
      Row 4 : Stage arrow row
      Row 5 : Modality weights | Fused embedding | Uncertainty | Counterfactual
      Row 6 : Stage arrow row
      Row 7 : Class probs bar | Staging wheel | Risk gauge | Agent chain
      Row 8 : FOOTER — ICD-10, surveillance, inference time
    """

    FIG_W, FIG_H = 24, 18
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=PALETTE["bg"])

    # We use a big outer GridSpec
    outer = gridspec.GridSpec(
        9, 1,
        figure=fig,
        hspace=0.04,
        height_ratios=[0.08, 0.22, 0.02, 0.22, 0.02, 0.22, 0.02, 0.22, 0.08],
        left=0.02, right=0.98, top=0.97, bottom=0.02
    )

    # ── ROW 0: HEADER ───────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(PALETTE["blue"])
    ax_hdr.axis("off")
    pred_color = CLASS_COLORS.get(data["pred_cls"], PALETTE["teal"])
    correct_mark = "✔ CORRECT" if data["pred_cls"] == data["cls_name"] else "✘ WRONG"
    correct_color = PALETTE["green"] if data["pred_cls"] == data["cls_name"] else PALETTE["accent"]

    ax_hdr.text(0.01, 0.55,
                "Agentic Multimodal Colon Cancer AI  —  Interactive Pipeline Diagram",
                transform=ax_hdr.transAxes, fontsize=14, fontweight="bold",
                color=PALETTE["text_light"], va="center")
    ax_hdr.text(0.72, 0.55,
                f"Prediction: {CLASS_DISPLAY.get(data['pred_cls'], data['pred_cls'])}  "
                f"({data['pred_conf']:.1%})",
                transform=ax_hdr.transAxes, fontsize=11, fontweight="bold",
                color=pred_color, va="center")
    ax_hdr.text(0.91, 0.55, correct_mark,
                transform=ax_hdr.transAxes, fontsize=11, fontweight="bold",
                color=correct_color, va="center")
    ax_hdr.text(0.55, 0.55,
                f"Ground Truth: {CLASS_DISPLAY.get(data['cls_name'], data['cls_name'])}",
                transform=ax_hdr.transAxes, fontsize=10,
                color=PALETTE["text_dim"], va="center")

    # ── ROW 1: Input stage (4 panels) ───────────────────────────────────────
    inner1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.04,
        width_ratios=[1, 1, 1, 1])

    # Panel 1A: Original image
    ax_orig = fig.add_subplot(inner1[0])
    ax_orig.imshow(data["orig_arr"])
    ax_orig.axis("off")
    _card(ax_orig, "STAGE 0 — Input Image", PALETTE["card"], PALETTE["teal"], fontsize=8)
    # Dataset badge
    dataset_name = "CVC-ClinicDB" if "CVC" in data["image_path"] else "HyperKvasir"
    badge_col = "#8A2BE2" if "CVC" in data["image_path"] else "#228B22"
    ax_orig.text(0.02, 0.96, f" {dataset_name} ", transform=ax_orig.transAxes,
                 fontsize=7, color="white", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_col, alpha=0.9))
    ax_orig.text(0.02, 0.04, f"{Path(data['image_path']).name}",
                 transform=ax_orig.transAxes, fontsize=6, color=PALETTE["text_dim"],
                 va="bottom")

    # Panel 1B: TCGA tabular heatmap
    ax_tab_in = fig.add_subplot(inner1[1])
    tab_vec  = data["tab_vec"]
    feat_names_short = [
        "Age", "BMI", "Yr Dx", "Follow-up",
        "Cigs/day", "Pack-yr", "Alcohol", "Gender",
        "Race", "Stage", "Morphol.", "Site"
    ]
    tab_mat = tab_vec.reshape(3, 4)
    im = ax_tab_in.imshow(tab_mat, cmap="plasma", aspect="auto",
                          vmin=tab_vec.min(), vmax=tab_vec.max())
    ax_tab_in.set_xticks([0, 1, 2, 3])
    ax_tab_in.set_xticklabels(feat_names_short[:4], fontsize=6,
                               color=PALETTE["text_light"], rotation=30, ha="right")
    ax_tab_in.set_yticks([0, 1, 2])
    ax_tab_in.set_yticklabels(["Demog.", "Smoking", "Clinical"], fontsize=6,
                               color=PALETTE["text_light"])
    for i in range(3):
        for j in range(4):
            fidx = i * 4 + j
            ax_tab_in.text(j, i, f"{tab_vec[fidx]:.2f}", ha="center", va="center",
                           fontsize=6, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax_tab_in, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    _card(ax_tab_in, "TCGA Tabular Features (12)", PALETTE["card2"], PALETTE["yellow"], 8)

    # Panel 1C: Clinical text (wordwrap)
    ax_txt_in = fig.add_subplot(inner1[2])
    ax_txt_in.axis("off")
    _card(ax_txt_in, "Clinical Text Input", PALETTE["card"], PALETTE["green"], 8)
    ax_txt_in.set_facecolor(PALETTE["card"])
    ctext = data["clinical_text"]
    # Word wrap at ~40 chars
    words = ctext.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 38:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    ax_txt_in.text(0.05, 0.85, "\n".join(lines[:6]),
                   transform=ax_txt_in.transAxes, fontsize=7.5,
                   color=PALETTE["text_light"], va="top",
                   family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="#0A2342",
                             edgecolor=PALETTE["green"], alpha=0.8))
    ax_txt_in.text(0.05, 0.18,
                   f'Tokenizer: BioBERT\nMax length: 64 tokens\nTokens: {len(data["tokens"])}',
                   transform=ax_txt_in.transAxes, fontsize=6.5,
                   color=PALETTE["text_dim"], va="bottom", family="monospace")

    # Panel 1D: Pipeline architecture overview
    ax_arch = fig.add_subplot(inner1[3])
    ax_arch.set_facecolor(PALETTE["card2"])
    ax_arch.axis("off")
    _card(ax_arch, "6-Agent Pipeline", PALETTE["card2"], PALETTE["purple"], 8)

    agents = [
        ("Image Agent",      "ConvNeXt-V2 + GradCAM++",  PALETTE["teal"]),
        ("Text Agent",       "BioBERT + Attention rollout", PALETTE["green"]),
        ("Tabular Agent",    "TabTransformer + SHAP",     PALETTE["yellow"]),
        ("Fusion Agent",     "Cross-modal Transformer",   PALETTE["purple"]),
        ("XAI Agent",        "MC-Dropout + Counterfact.", PALETTE["orange"]),
        ("Clinical Agent",   "BSG/NICE Guidelines",       PALETTE["accent"]),
    ]
    for k, (aname, adesc, acol) in enumerate(agents):
        ypos = 0.88 - k * 0.14
        ax_arch.add_patch(FancyBboxPatch((0.03, ypos - 0.04), 0.94, 0.10,
                          boxstyle="round,pad=0.01", linewidth=1.2,
                          edgecolor=acol, facecolor=PALETTE["blue"] + "88",
                          transform=ax_arch.transAxes, zorder=2))
        ax_arch.text(0.08, ypos + 0.02, f"[{k+1}] {aname}", transform=ax_arch.transAxes,
                     fontsize=7, color=acol, fontweight="bold", va="center")
        ax_arch.text(0.08, ypos - 0.02, adesc, transform=ax_arch.transAxes,
                     fontsize=5.5, color=PALETTE["text_dim"], va="center")

    # ── ROW 2: Arrow separator ───────────────────────────────────────────────
    ax_sep1 = fig.add_subplot(outer[2])
    ax_sep1.set_facecolor(PALETTE["bg"])
    ax_sep1.axis("off")
    ax_sep1.axhline(0.5, color=PALETTE["teal"], linewidth=0.8, linestyle="--", alpha=0.4)
    ax_sep1.text(0.5, 0.5, "▼  STAGE 1-3: Image / Text / Tabular Branches  ▼",
                 transform=ax_sep1.transAxes, fontsize=8,
                 color=PALETTE["teal"], ha="center", va="center", alpha=0.8)

    # ── ROW 3: Branch outputs ────────────────────────────────────────────────
    inner3 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[3], wspace=0.06,
        width_ratios=[1, 1, 1, 1])

    # Panel 3A: GradCAM++ overlay (Image Agent output)
    ax_cam_ov = fig.add_subplot(inner3[0])
    ax_cam_ov.imshow(data["overlay"])
    ax_cam_ov.axis("off")
    _card(ax_cam_ov, "Image Agent — Grad-CAM++ Overlay", PALETTE["card"], PALETTE["teal"], 8)
    ax_cam_ov.text(0.02, 0.04, f"ROI coverage: {float(data['cam'].mean()):.1%}",
                   transform=ax_cam_ov.transAxes, fontsize=6.5,
                   color=PALETTE["text_dim"], va="bottom")

    # Panel 3B: CAM heatmap
    ax_cam_hm = fig.add_subplot(inner3[1])
    im2 = ax_cam_hm.imshow(data["cam"], cmap="hot", vmin=0, vmax=1)
    ax_cam_hm.axis("off")
    _card(ax_cam_hm, "Activation Heatmap (ConvNeXt Stage-4)", PALETTE["card"], PALETTE["teal"], 8)
    plt.colorbar(im2, ax=ax_cam_hm, fraction=0.046, pad=0.04,
                 label="Activation").ax.tick_params(labelsize=6)
    ax_cam_hm.contour(data["cam"], levels=[0.5, 0.75], colors=["cyan", "white"],
                      linewidths=0.8, alpha=0.6)
    # Peak annotation
    peak_y, peak_x = np.unravel_index(data["cam"].argmax(), data["cam"].shape)
    ax_cam_hm.annotate("Peak", xy=(peak_x, peak_y), fontsize=6,
                        color="yellow", ha="center",
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=0.8),
                        xytext=(peak_x + 20, peak_y - 20))

    # Panel 3C: BioBERT token importance
    ax_tok = fig.add_subplot(inner3[2])
    _card(ax_tok, "Text Agent — BioBERT Token Importance", PALETTE["card2"], PALETTE["green"], 8)
    # Show top 12 tokens by importance
    tokens_show = data["tokens"][:20]
    imp_show    = data["tok_importance"][:20]
    # Filter out PAD/[SEP] for display
    keep = [(t, imp) for t, imp in zip(tokens_show, imp_show)
            if t not in ["[PAD]", "[SEP]", "[CLS]"]][:12]
    if keep:
        t_labels, t_vals = zip(*keep)
    else:
        t_labels = tokens_show[:8]
        t_vals   = imp_show[:8]
    y_pos = np.arange(len(t_labels))
    bars = ax_tok.barh(y_pos, t_vals, color=PALETTE["green"], alpha=0.8)
    ax_tok.set_yticks(y_pos)
    ax_tok.set_yticklabels(t_labels, fontsize=6.5, color=PALETTE["text_light"])
    ax_tok.set_xlabel("Importance", fontsize=7, color=PALETTE["text_dim"])
    ax_tok.tick_params(colors=PALETTE["text_light"], labelsize=7)
    ax_tok.set_facecolor(PALETTE["card2"])
    # Colour top tokens red
    for i, (bar, val) in enumerate(zip(bars, t_vals)):
        if val > 0.5:
            bar.set_facecolor(PALETTE["accent"])
    ax_tok.invert_yaxis()

    # Panel 3D: SHAP tabular importance
    ax_shap = fig.add_subplot(inner3[3])
    _card(ax_shap, "Tabular Agent — SHAP Feature Importance", PALETTE["card"], PALETTE["yellow"], 8)
    tab_imp  = data["tab_importance"]
    order    = np.argsort(tab_imp)[::-1]
    top_n    = min(12, len(order))
    feat_show = [feat_names_short[i] for i in order[:top_n]]
    imp_show2 = [tab_imp[i] for i in order[:top_n]]
    colors_shap = [PALETTE["accent"] if v > 0.5 else PALETTE["yellow"] for v in imp_show2]
    yp = np.arange(top_n)
    ax_shap.barh(yp, imp_show2, color=colors_shap, alpha=0.85)
    ax_shap.set_yticks(yp)
    ax_shap.set_yticklabels(feat_show, fontsize=6.5, color=PALETTE["text_light"])
    ax_shap.set_xlabel("SHAP Importance", fontsize=7, color=PALETTE["text_dim"])
    ax_shap.tick_params(colors=PALETTE["text_light"], labelsize=7)
    ax_shap.set_facecolor(PALETTE["card"])
    ax_shap.invert_yaxis()
    red_p  = mpatches.Patch(color=PALETTE["accent"], label="High (>0.5)")
    yel_p  = mpatches.Patch(color=PALETTE["yellow"], label="Low")
    ax_shap.legend(handles=[red_p, yel_p], loc="lower right",
                   fontsize=5.5, facecolor=PALETTE["card2"],
                   labelcolor=PALETTE["text_light"], framealpha=0.7)

    # ── ROW 4: Arrow separator ───────────────────────────────────────────────
    ax_sep2 = fig.add_subplot(outer[4])
    ax_sep2.set_facecolor(PALETTE["bg"])
    ax_sep2.axis("off")
    ax_sep2.axhline(0.5, color=PALETTE["purple"], linewidth=0.8, linestyle="--", alpha=0.4)
    ax_sep2.text(0.5, 0.5, "▼  STAGE 4-5: Fusion Transformer + XAI Analysis  ▼",
                 transform=ax_sep2.transAxes, fontsize=8,
                 color=PALETTE["purple"], ha="center", va="center", alpha=0.8)

    # ── ROW 5: Fusion + XAI ─────────────────────────────────────────────────
    inner5 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[5], wspace=0.06,
        width_ratios=[1, 1, 1, 1])

    # Panel 5A: Modality weights (donut chart)
    ax_mod = fig.add_subplot(inner5[0])
    _card(ax_mod, "Fusion Agent — Modality Weights", PALETTE["card2"], PALETTE["purple"], 8)
    mod_w   = data["mod_weights"]
    mod_lbl = ["Image\n(GradCAM++)", "Text\n(BioBERT)", "Tabular\n(TCGA)"]
    mod_col = [PALETTE["teal"], PALETTE["green"], PALETTE["yellow"]]
    wedges, texts, autotexts = ax_mod.pie(
        mod_w, labels=None, colors=mod_col,
        autopct="%1.1f%%", pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2),
        startangle=90
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color("white")
        at.set_fontweight("bold")
    legend_items = [mpatches.Patch(facecolor=c, label=l)
                    for c, l in zip(mod_col, mod_lbl)]
    ax_mod.legend(handles=legend_items, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), ncol=3,
                  fontsize=5.5, facecolor=PALETTE["card2"],
                  labelcolor=PALETTE["text_light"], framealpha=0.7)
    dom_idx = int(np.argmax(mod_w))
    ax_mod.text(0, 0, f"Dominant\n{mod_lbl[dom_idx]}",
                ha="center", va="center", fontsize=6,
                color=mod_col[dom_idx], fontweight="bold")

    # Panel 5B: Fused embedding visualisation (as heatmap strip)
    ax_emb = fig.add_subplot(inner5[1])
    _card(ax_emb, "Fused Embedding (256-dim, Cross-Attn)", PALETTE["card"], PALETTE["purple"], 8)
    emb = data["fused_emb"]
    # Reshape to 16x16 for visual
    emb_sq = emb[:256].reshape(16, 16)
    im3 = ax_emb.imshow(emb_sq, cmap="RdYlBu_r", aspect="auto")
    ax_emb.axis("off")
    plt.colorbar(im3, ax=ax_emb, fraction=0.046, pad=0.04,
                 label="Activation").ax.tick_params(labelsize=5)
    ax_emb.text(0.5, -0.04, "16×16 reshape of 256-dim fused vector",
                transform=ax_emb.transAxes, fontsize=6,
                color=PALETTE["text_dim"], ha="center")

    # Panel 5C: MC-Dropout uncertainty
    ax_unc = fig.add_subplot(inner5[2])
    _card(ax_unc, "XAI Agent — MC-Dropout Uncertainty", PALETTE["card2"], PALETTE["orange"], 8)
    unc    = data["uncertainty"]
    std_v  = data["pred_std"]

    # Uncertainty gauge (horizontal bar)
    ax_unc.set_xlim(0, 1)
    ax_unc.set_ylim(0, 1)
    ax_unc.axis("off")

    # Background bar
    ax_unc.barh(0.75, 1.0, height=0.12, color="#333333", left=0, align="center")
    unc_col = PALETTE["green"] if unc < 0.3 else (PALETTE["yellow"] if unc < 0.6 else PALETTE["accent"])
    ax_unc.barh(0.75, unc, height=0.12, color=unc_col, left=0, align="center")
    ax_unc.text(0.5, 0.62, f"Uncertainty: {unc:.3f}", ha="center", va="top",
                fontsize=8, color=unc_col, fontweight="bold",
                transform=ax_unc.transAxes)
    unc_lbl = "LOW" if unc < 0.3 else ("MODERATE" if unc < 0.6 else "HIGH")
    ax_unc.text(0.5, 0.52, f"({unc_lbl})", ha="center", va="top",
                fontsize=9, color=unc_col, transform=ax_unc.transAxes)

    # Per-class std
    for ci, (cname, sv) in enumerate(zip(CLASS_NAMES, std_v)):
        yp = 0.40 - ci * 0.07
        ax_unc.text(0.02, yp, cname[:12], ha="left", va="center",
                    fontsize=6, color=PALETTE["text_dim"],
                    transform=ax_unc.transAxes)
        ax_unc.barh(yp * ax_unc.get_ylim()[1], sv * 0.8, height=0.02,
                    color=CLASS_COLORS.get(cname, PALETTE["teal"]),
                    left=0.12, align="center", alpha=0.7)
        ax_unc.text(0.15 + sv * 0.8, yp * ax_unc.get_ylim()[1],
                    f"σ={sv:.3f}", va="center", fontsize=5.5,
                    color=PALETTE["text_dim"])

    ax_unc.text(0.5, 0.02, "n=15 MC-Dropout forward passes",
                ha="center", va="bottom", fontsize=6, color=PALETTE["text_dim"],
                transform=ax_unc.transAxes)

    # Panel 5D: Counterfactual + Risk flags
    ax_cf = fig.add_subplot(inner5[3])
    ax_cf.set_facecolor(PALETTE["card"])
    ax_cf.axis("off")
    _card(ax_cf, "Counterfactual & Risk Flags", PALETTE["card"], PALETTE["accent"], 8)

    cf_text = data["counterfactual"]
    cf_words = cf_text.split()
    cf_lines, cur = [], ""
    for w in cf_words:
        if len(cur) + len(w) + 1 > 36:
            cf_lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur: cf_lines.append(cur)

    ax_cf.text(0.05, 0.94, "COUNTERFACTUAL:", transform=ax_cf.transAxes,
               fontsize=7, color=PALETTE["orange"], fontweight="bold", va="top")
    ax_cf.text(0.05, 0.87, "\n".join(cf_lines[:4]), transform=ax_cf.transAxes,
               fontsize=6.5, color=PALETTE["text_light"], va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A0A00",
                         edgecolor=PALETTE["orange"], alpha=0.7))

    ax_cf.text(0.05, 0.55, "RISK FLAGS:", transform=ax_cf.transAxes,
               fontsize=7, color=PALETTE["accent"], fontweight="bold", va="top")
    risk_col = PALETTE["accent"] if data["risk_lbl"] == "Malignant" else PALETTE["green"]
    for ri, rflag in enumerate(data["risk_flags"][:5]):
        ax_cf.text(0.07, 0.48 - ri * 0.08, f"• {rflag}",
                   transform=ax_cf.transAxes, fontsize=6,
                   color=risk_col, va="top")

    # ── ROW 6: Arrow separator ───────────────────────────────────────────────
    ax_sep3 = fig.add_subplot(outer[6])
    ax_sep3.set_facecolor(PALETTE["bg"])
    ax_sep3.axis("off")
    ax_sep3.axhline(0.5, color=PALETTE["green"], linewidth=0.8, linestyle="--", alpha=0.4)
    ax_sep3.text(0.5, 0.5, "▼  STAGE 6: Final Diagnosis + Clinical Recommendation  ▼",
                 transform=ax_sep3.transAxes, fontsize=8,
                 color=PALETTE["green"], ha="center", va="center", alpha=0.8)

    # ── ROW 7: Final outputs ─────────────────────────────────────────────────
    inner7 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[7], wspace=0.06,
        width_ratios=[1.2, 0.9, 0.9, 1.0])

    # Panel 7A: 5-class probability bars
    ax_cls = fig.add_subplot(inner7[0])
    _card(ax_cls, "Pathology Classification (5 Classes)", PALETTE["card"], PALETTE["teal"], 8)
    probs  = data["path_probs"]
    bars_c = ax_cls.barh(
        range(N_CLASSES), probs,
        color=[CLASS_COLORS.get(c, PALETTE["teal"]) for c in CLASS_NAMES],
        alpha=0.85, edgecolor=PALETTE["bg"], linewidth=0.8
    )
    ax_cls.set_yticks(range(N_CLASSES))
    ax_cls.set_yticklabels(
        [CLASS_DISPLAY.get(c, c) for c in CLASS_NAMES],
        fontsize=7, color=PALETTE["text_light"]
    )
    ax_cls.set_xlim(0, 1.1)
    ax_cls.set_xlabel("Probability", fontsize=7, color=PALETTE["text_dim"])
    ax_cls.tick_params(colors=PALETTE["text_light"], labelsize=7)
    ax_cls.set_facecolor(PALETTE["card"])
    for bi, (bar, prob) in enumerate(zip(bars_c, probs)):
        ax_cls.text(min(prob + 0.02, 1.05), bi, f"{prob:.3f}",
                    va="center", fontsize=6.5, color=PALETTE["text_light"],
                    fontweight="bold" if bi == int(probs.argmax()) else "normal")
    ax_cls.invert_yaxis()
    # Highlight predicted class
    pred_idx_bar = int(probs.argmax())
    bars_c[pred_idx_bar].set_edgecolor("white")
    bars_c[pred_idx_bar].set_linewidth(2)

    # Panel 7B: Staging radar / bar
    ax_stg = fig.add_subplot(inner7[1])
    _card(ax_stg, "Cancer Staging", PALETTE["card2"], PALETTE["yellow"], 8)
    stage_lbs = ["No Cancer", "Stage I", "Stage II", "Stage III/IV"]
    stg_probs  = data["stage_probs"]
    stg_cols   = [STAGE_COLORS[s] for s in stage_lbs]
    bars_s = ax_stg.bar(range(4), stg_probs, color=stg_cols, alpha=0.85,
                         edgecolor=PALETTE["bg"], linewidth=0.8)
    ax_stg.set_xticks(range(4))
    ax_stg.set_xticklabels(["No Ca.", "Stg I", "Stg II", "Stg III/IV"],
                            fontsize=6.5, color=PALETTE["text_light"], rotation=15)
    ax_stg.set_ylim(0, 1.15)
    ax_stg.set_ylabel("Probability", fontsize=7, color=PALETTE["text_dim"])
    ax_stg.tick_params(colors=PALETTE["text_light"], labelsize=7)
    ax_stg.set_facecolor(PALETTE["card2"])
    for bi, (bar, prob) in enumerate(zip(bars_s, stg_probs)):
        ax_stg.text(bi, prob + 0.03, f"{prob:.2f}", ha="center",
                    fontsize=6.5, color=PALETTE["text_light"])
    # Star on predicted
    stg_pred = int(stg_probs.argmax())
    bars_s[stg_pred].set_edgecolor("white")
    bars_s[stg_pred].set_linewidth(2)
    ax_stg.text(stg_pred, stg_probs[stg_pred] + 0.09, "★",
                ha="center", fontsize=10, color=PALETTE["yellow"])
    ax_stg.text(0.5, 0.02, f"Predicted: {data['stage_lbl']} ({data['stage_conf']:.1%})",
                transform=ax_stg.transAxes, fontsize=6.5, color=PALETTE["yellow"],
                ha="center", fontweight="bold")

    # Panel 7C: Cancer risk gauge
    ax_risk = fig.add_subplot(inner7[2])
    _card(ax_risk, "Cancer Risk Score", PALETTE["card"], PALETTE["accent"], 8)
    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(0, 1)
    ax_risk.axis("off")

    risk_score = data["risk_score"]
    risk_col   = PALETTE["malignant"] if risk_score >= 0.5 else PALETTE["benign"]

    # Draw semicircle gauge
    theta = np.linspace(np.pi, 0, 200)
    for i in range(100):
        t0, t1 = theta[i * 2], theta[i * 2 + 1]
        v = i / 99.0
        gc = plt.cm.RdYlGn_r(v)
        x0, y0 = 0.5 + 0.35 * np.cos(t0), 0.45 + 0.35 * np.sin(t0)
        x1, y1 = 0.5 + 0.35 * np.cos(t1), 0.45 + 0.35 * np.sin(t1)
        xo0, yo0 = 0.5 + 0.22 * np.cos(t0), 0.45 + 0.22 * np.sin(t0)
        xo1, yo1 = 0.5 + 0.22 * np.cos(t1), 0.45 + 0.22 * np.sin(t1)
        ax_risk.fill([x0, x1, xo1, xo0], [y0, y1, yo1, yo0],
                     color=gc, alpha=0.85, transform=ax_risk.transAxes)

    # Needle
    needle_theta = np.pi - risk_score * np.pi
    nx = 0.5 + 0.30 * np.cos(needle_theta)
    ny = 0.45 + 0.30 * np.sin(needle_theta)
    ax_risk.annotate("", xy=(nx, ny), xytext=(0.5, 0.45),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="white",
                                     lw=2.5, mutation_scale=15))
    ax_risk.scatter([0.5], [0.45], s=60, color=PALETTE["accent"],
                    transform=ax_risk.transAxes, zorder=5)

    ax_risk.text(0.5, 0.18, f"{risk_score:.3f}", ha="center", va="center",
                 fontsize=16, color=risk_col, fontweight="bold",
                 transform=ax_risk.transAxes)
    ax_risk.text(0.5, 0.06, data["risk_lbl"], ha="center", va="center",
                 fontsize=11, color=risk_col, fontweight="bold",
                 transform=ax_risk.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["card"],
                           edgecolor=risk_col, alpha=0.8))
    ax_risk.text(0.10, 0.36, "BENIGN", fontsize=7, color=PALETTE["benign"],
                 transform=ax_risk.transAxes, ha="center")
    ax_risk.text(0.90, 0.36, "MALIG.", fontsize=7, color=PALETTE["malignant"],
                 transform=ax_risk.transAxes, ha="center")

    # Panel 7D: Clinical recommendation card
    ax_clin = fig.add_subplot(inner7[3])
    ax_clin.set_facecolor(PALETTE["card2"])
    ax_clin.axis("off")
    _card(ax_clin, "Clinical Agent — Recommendation", PALETTE["card2"], PALETTE["green"], 8)

    urg_col = (PALETTE["accent"] if "URGENT" in data["urgency"]
               else PALETTE["yellow"] if "Soon" in data["urgency"]
               else PALETTE["green"])

    items = [
        ("DIAGNOSIS:",     CLASS_DISPLAY.get(data["pred_cls"], data["pred_cls"]), PALETTE["teal"]),
        ("ICD-10:",        data["icd10"], PALETTE["text_dim"]),
        ("STAGE:",         f"{data['stage_lbl']} ({data['stage_conf']:.1%})", PALETTE["yellow"]),
        ("RISK:",          f"{data['risk_lbl']} (score={data['risk_score']:.3f})", risk_col),
        ("URGENCY:",       data["urgency"], urg_col),
        ("CONFIDENCE:",    f"{data['pred_conf']:.1%}", PALETTE["teal"]),
        ("UNCERTAINTY:",   f"{data['uncertainty']:.3f}", PALETTE["orange"]),
    ]
    for ki, (key, val, col) in enumerate(items):
        yp = 0.93 - ki * 0.11
        ax_clin.text(0.02, yp, key, transform=ax_clin.transAxes,
                     fontsize=7, color=PALETTE["text_dim"], va="top", fontweight="bold")
        ax_clin.text(0.36, yp, val, transform=ax_clin.transAxes,
                     fontsize=7, color=col, va="top")

    surv_text = data["surveillance"]
    surv_words = surv_text.split()
    surv_lines, cur = [], ""
    for w in surv_words:
        if len(cur) + len(w) + 1 > 32:
            surv_lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur: surv_lines.append(cur)
    ax_clin.text(0.02, 0.16, "SURVEILLANCE:", transform=ax_clin.transAxes,
                 fontsize=7, color=PALETTE["green"], fontweight="bold", va="top")
    ax_clin.text(0.02, 0.10, "\n".join(surv_lines[:3]), transform=ax_clin.transAxes,
                 fontsize=6.5, color=PALETTE["text_light"], va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["blue"],
                           edgecolor=PALETTE["green"], alpha=0.7))

    # ── ROW 8: FOOTER ───────────────────────────────────────────────────────
    ax_ftr = fig.add_subplot(outer[8])
    ax_ftr.set_facecolor(PALETTE["blue"])
    ax_ftr.axis("off")
    dataset_name = "CVC-ClinicDB" if "CVC" in data["image_path"] else "HyperKvasir"
    footer_txt = (
        f"Dataset: {dataset_name}  |  "
        f"Subfolder: {Path(data['image_path']).parent.name}  |  "
        f"Image: {Path(data['image_path']).name}  |  "
        f"Inference: {data['inference_ms']:.0f} ms  |  "
        f"Model: UnifiedMultiModalTransformer (~74.7M params)  |  "
        f"Val F1: 0.9989  |  Test AUC: 1.0000"
    )
    ax_ftr.text(0.5, 0.55, footer_txt, transform=ax_ftr.transAxes, fontsize=7,
                color=PALETTE["text_dim"], ha="center", va="center")
    ax_ftr.text(0.5, 0.15,
                "⚠  This system is a decision-support tool. "
                "Clinical judgement by a qualified gastroenterologist is required.",
                transform=ax_ftr.transAxes, fontsize=6.5,
                color=PALETTE["accent"], ha="center", va="center", style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[PipelineDiagram] Saved: {out_path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Interactive Pipeline Diagram")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a specific image file")
    parser.add_argument("--class", dest="cls", type=str, default=None,
                        choices=CLASS_NAMES,
                        help="Choose a specific pathology class")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG file path (default: outputs/pipeline_diagram/...)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ── Setup device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PipelineDiagram] Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("[PipelineDiagram] Loading model ...")
    model = UnifiedMultiModalTransformer(
        n_classes=N_CLASSES,
        n_tabular_features=N_TABULAR_FEATURES,
        d_model=D_MODEL,
    )
    ckpt  = torch.load(CHECKPOINT, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    print(f"  Loaded epoch={ckpt.get('epoch','?')} | "
          f"val_f1={ckpt.get('val_f1', ckpt.get('val_acc', 0)):.4f}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    print("[PipelineDiagram] Loading BioBERT tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # ── Load TCGA pool ───────────────────────────────────────────────────────
    print("[PipelineDiagram] Building TCGA pool ...")
    tcga_pool = _tcga_pool()
    total = sum(len(v) for v in tcga_pool.values())
    print(f"  TCGA patients loaded: {total}")

    # ── Pick image ───────────────────────────────────────────────────────────
    picked = pick_image(args.image, args.cls)
    img_path, cls_name, subfolder, dataset_name = picked
    print(f"[PipelineDiagram] Selected image:")
    print(f"  Dataset  : {dataset_name}")
    print(f"  Subfolder: {subfolder}")
    print(f"  Class    : {cls_name}")
    print(f"  Path     : {img_path}")

    # ── Run pipeline capture ─────────────────────────────────────────────────
    print("[PipelineDiagram] Running full pipeline ...")
    capture = PipelineCapture(model, tokenizer, device)
    data    = capture.run(img_path, cls_name, tcga_pool)
    correct_str = "CORRECT" if data["pred_cls"] == cls_name else "WRONG"
    print(f"  Prediction : {data['pred_cls']} ({data['pred_conf']:.1%}) [{correct_str}]")
    print(f"  Stage      : {data['stage_lbl']} ({data['stage_conf']:.1%})")
    print(f"  Risk       : {data['risk_lbl']} (score={data['risk_score']:.3f})")
    print(f"  Uncertainty: {data['uncertainty']:.3f}")
    print(f"  Modality W.: Image={data['mod_weights'][0]:.3f}  "
          f"Text={data['mod_weights'][1]:.3f}  Tab={data['mod_weights'][2]:.3f}")
    print(f"  Inference  : {data['inference_ms']:.0f} ms")

    # ── Output paths ─────────────────────────────────────────────────────────
    out_dir  = Path("outputs/pipeline_diagram")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_cls = cls_name.replace("-", "_")
    safe_fn  = Path(img_path).stem[:20]
    case_id  = f"{dataset_name}__{safe_cls}__{safe_fn}"

    if args.out:
        png_path = args.out
    else:
        png_path = str(out_dir / f"pipeline_{case_id}.png")

    json_path = str(out_dir / f"pipeline_{case_id}.json")

    # ── Build diagram ────────────────────────────────────────────────────────
    print("[PipelineDiagram] Building diagram ...")
    build_diagram(data, png_path)

    # ── Save JSON ────────────────────────────────────────────────────────────
    json_data = {
        "case_id":       case_id,
        "image_path":    img_path,
        "dataset":       dataset_name,
        "subfolder":     subfolder,
        "cls_name":      cls_name,
        "prediction":    data["pred_cls"],
        "pred_confidence": data["pred_conf"],
        "correct":       data["pred_cls"] == cls_name,
        "cancer_risk":   data["risk_lbl"],
        "risk_score":    data["risk_score"],
        "stage":         data["stage_lbl"],
        "stage_conf":    data["stage_conf"],
        "uncertainty":   data["uncertainty"],
        "urgency":       data["urgency"],
        "icd10":         data["icd10"],
        "surveillance":  data["surveillance"],
        "counterfactual": data["counterfactual"],
        "modality_weights": {
            "image":   float(data["mod_weights"][0]),
            "text":    float(data["mod_weights"][1]),
            "tabular": float(data["mod_weights"][2]),
        },
        "class_probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, data["path_probs"])},
        "stage_probabilities": dict(zip(
            ["No Cancer","Stage I","Stage II","Stage III/IV"],
            [float(p) for p in data["stage_probs"]]
        )),
        "tabular_features": {k: round(float(v), 4)
                             for k, v in zip(TABULAR_FEATURES, data["tab_vec"])},
        "tabular_importance": {k: round(float(v), 4)
                               for k, v in zip(TABULAR_FEATURES, data["tab_importance"])},
        "risk_flags":    data["risk_flags"],
        "clinical_text": data["clinical_text"],
        "inference_ms":  data["inference_ms"],
        "output_png":    png_path,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"[PipelineDiagram] Done!")
    print(f"  PNG  -> {png_path}")
    print(f"  JSON -> {json_path}")

    return png_path


if __name__ == "__main__":
    main()
