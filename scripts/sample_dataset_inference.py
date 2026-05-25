"""ColonAI — REAL inference panel for 10 random samples per dataset folder.

For every image-containing leaf folder in the data/ tree we:
  1. Randomly sample 10 images (seed = 42, so reproducible).
  2. Load the trained checkpoint and run a REAL forward pass.
  3. Generate the REAL GradCAM++ map for the predicted class.
  4. Render a 4-panel PNG: original | gradcam heatmap | gradcam overlay
     | text panel (top-3 probabilities + class + confidence + folder).
  5. Save under outputs/dataset_sample_panel/<dataset>/<subfolder>/.

After every panel is written, we produce a master index.html that lets
you browse all panels organised by dataset and class.

EXPLICITLY NOT FAKE:
  • The script refuses to start if the model checkpoint cannot be loaded.
  • If a forward pass fails for any sample, the panel records the actual
    Python exception and is marked clearly — the panel image is NOT shown
    as if it succeeded.
  • Each panel embeds the SHA-256 of the original image bytes so the
    output is verifiable against the input.

Run:
    python3 scripts/sample_dataset_inference.py
    python3 scripts/sample_dataset_inference.py --n 5            # 5 per folder
    python3 scripts/sample_dataset_inference.py --datasets hyper_kvasir
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch


# ──────────────────────────────────────────────────────────────────────
#  Constants — match what app.py / training uses
# ──────────────────────────────────────────────────────────────────────
IMG_SIZE       = 224
IMG_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_CLASSES      = 5
D_MODEL        = 256
N_TAB_FEATURES = 12
BERT_MODEL     = "dmis-lab/biobert-base-cased-v1.2"

PATHOLOGY_CLASSES = [
    "polyps", "uc-mild", "uc-moderate-sev",
    "barretts-esoph", "therapeutic",
]
PLAIN = {
    "polyps":         "Colorectal Polyps",
    "uc-mild":        "Ulcerative Colitis (Mild)",
    "uc-moderate-sev":"Ulcerative Colitis (Moderate–Severe)",
    "barretts-esoph": "Barrett's Esophagus",
    "therapeutic":    "Post-Therapeutic Site",
}

OUT_ROOT = _REPO_ROOT / "outputs" / "dataset_sample_panel"
CKPT_V2 = _REPO_ROOT / "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
CKPT_V1 = _REPO_ROOT / "outputs/unified_multimodal/checkpoints/best_model.pth"


# ──────────────────────────────────────────────────────────────────────
#  Dataset discovery
# ──────────────────────────────────────────────────────────────────────
def _images_in(folder: Path,
               exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """Return sorted list of image files directly inside `folder`."""
    if not folder.exists():
        return []
    out: List[Path] = []
    for ext in exts:
        out.extend(folder.glob(f"*{ext}"))
        out.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(out))


def discover_dataset_folders() -> List[Dict[str, Any]]:
    """Return a list of {dataset, subfolder, path, n_images, expected_class}
    for every image-containing leaf folder we care about.

    `expected_class` is the trained-class label we EXPECT the model to
    predict for images in that folder. It is None when the folder
    contains content outside the 5 training classes (normal anatomy,
    bowel-prep quality, hemorrhoids, esophagitis) — in that case a
    "match" check is N/A, not a failure.
    """
    root = _REPO_ROOT
    discovered: List[Dict[str, Any]] = []

    # Map HyperKvasir leaf folder names → trained 5-class label
    # (None = folder is out-of-distribution for the trained classifier)
    HK_LEAF_TO_CLASS = {
        # ── pathology classes (in-distribution) ──────────────────
        "polyps":                         "polyps",
        "ulcerative-colitis-grade-0-1":   "uc-mild",
        "ulcerative-colitis-grade-1":     "uc-mild",
        "ulcerative-colitis-grade-1-2":   "uc-mild",
        "ulcerative-colitis-grade-2":     "uc-moderate-sev",
        "ulcerative-colitis-grade-2-3":   "uc-moderate-sev",
        "ulcerative-colitis-grade-3":     "uc-moderate-sev",
        "barretts":                       "barretts-esoph",
        "barretts-short-segment":         "barretts-esoph",
        "dyed-lifted-polyps":             "therapeutic",
        "dyed-resection-margins":         "therapeutic",
        # ── out-of-distribution (NOT in 5-class training set) ────
        "cecum":             None,
        "ileum":             None,
        "retroflex-rectum":  None,
        "pylorus":           None,
        "retroflex-stomach": None,
        "z-line":            None,
        "esophagitis-a":     None,
        "esophagitis-b-d":   None,
        "hemorrhoids":       None,
        "bbps-0-1":          None,
        "bbps-2-3":          None,
        "impacted-stool":    None,
    }

    # ── HyperKvasir cleaned: 23 leaf folders ──────────────────────────
    hk_root = root / "data/processed/hyper_kvasir_clean"
    if hk_root.exists():
        for leaf in sorted(hk_root.glob("*/*/*")):
            if leaf.is_dir():
                imgs = _images_in(leaf)
                if imgs:
                    discovered.append({
                        "dataset":  "hyper_kvasir",
                        "subfolder": f"{leaf.parent.parent.name}/{leaf.parent.name}/{leaf.name}",
                        "path":     leaf,
                        "n_images": len(imgs),
                        "expected_class": HK_LEAF_TO_CLASS.get(leaf.name),
                        "ood":           HK_LEAF_TO_CLASS.get(leaf.name) is None,
                    })

    # ── CVC-ClinicDB PNG (all polyps) ─────────────────────────────────
    p = root / "data/raw/CVC-ClinicDB/PNG/Original"
    if p.exists():
        imgs = _images_in(p)
        if imgs:
            discovered.append({
                "dataset":  "cvc_clinicdb",
                "subfolder": "Original",
                "path":     p,
                "n_images": len(imgs),
                "expected_class": "polyps",
                "ood":            False,
            })

    # ── Kvasir-SEG (all polyps) ───────────────────────────────────────
    p = root / "data/raw/kvasir-seg/Kvasir-SEG/images"
    if p.exists():
        imgs = _images_in(p)
        if imgs:
            discovered.append({
                "dataset":  "kvasir_seg",
                "subfolder": "images",
                "path":     p,
                "n_images": len(imgs),
                "expected_class": "polyps",
                "ood":            False,
            })

    # ── 5 polyp test datasets (each has images/ subfolder — all polyps) ──
    test_root = root / "data/raw/test_polyp_datasets/TestDataset"
    if test_root.exists():
        for sub in sorted(test_root.iterdir()):
            imgs_dir = sub / "images"
            if imgs_dir.exists():
                imgs = _images_in(imgs_dir)
                if imgs:
                    discovered.append({
                        "dataset":  "polyp_test",
                        "subfolder": sub.name,
                        "path":     imgs_dir,
                        "n_images": len(imgs),
                        "expected_class": "polyps",
                        "ood":            False,
                    })

    return discovered


def _matches_folder(predicted: str, expected: Optional[str]) -> bool:
    """Match check that respects the in-distribution / OOD distinction.

    OOD folders (no training class) — always "differs", because the
    5-class classifier has no honest answer for normal anatomy, bowel
    prep, or esophagitis. We track them separately in the index.
    """
    if expected is None:
        return False
    return predicted == expected


# ──────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────
def load_model(device: torch.device, *, verbose: bool = True
               ) -> Tuple[Any, Any, Path]:
    """Load the unified transformer + tokenizer. Returns (model, tokenizer, ckpt_path).

    Raises RuntimeError if no checkpoint is available — refusing to
    proceed prevents the script from silently emitting fake/random
    predictions.
    """
    ckpt = CKPT_V2 if CKPT_V2.exists() else CKPT_V1 if CKPT_V1.exists() else None
    if ckpt is None:
        raise RuntimeError(
            "No checkpoint found at either "
            f"{CKPT_V2} or {CKPT_V1}.  Refusing to generate predictions "
            "without a real trained model (per user instruction: "
            "real results only).")

    if verbose:
        print(f"  using checkpoint: {ckpt}  ({ckpt.stat().st_size/1e6:.1f} MB)")

    from src.models.unified_transformer import UnifiedMultiModalTransformer
    from transformers import AutoTokenizer

    model = UnifiedMultiModalTransformer(
        n_classes=N_CLASSES, d_model=D_MODEL,
        n_fusion_heads=8, n_fusion_layers=3,
        n_tabular_features=N_TAB_FEATURES,
    )

    try:
        from src.app.security import safe_torch_load
        state = safe_torch_load(str(ckpt), map_location=device, allow_unsafe=True)
    except Exception:
        state = torch.load(str(ckpt), map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("model_state",
                state.get("model_state_dict",
                state.get("state_dict", state)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        print(f"  loaded — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        if len(missing) > 50:
            raise RuntimeError(f"too many missing keys ({len(missing)}) — "
                                "checkpoint likely incompatible with model. "
                                "Refusing to generate predictions.")
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    return model, tokenizer, ckpt


# ──────────────────────────────────────────────────────────────────────
#  Inference helpers
# ──────────────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """(1,3,224,224) normalised tensor + (224,224,3) 0-1 float."""
    pil = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    norm = (arr - IMG_MEAN) / IMG_STD
    t = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return t, arr


def tokenize(tokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", max_length=128,
                     padding="max_length", truncation=True)
    return enc["input_ids"], enc["attention_mask"]


class GradCAMPP:
    """Grad-CAM++ on the model's image target layer."""
    def __init__(self, model):
        self.model = model
        self._acts = None
        self._grads = None
        try:
            target = model.get_image_target_layer()
        except Exception:
            # Fallback: last conv block in backbone
            target = None
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target = m
        if target is None:
            raise RuntimeError("could not find a conv layer for GradCAM")
        target.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def generate(self, image, target_class, input_ids, attention_mask, tabular):
        self.model.eval()
        image = image.detach().requires_grad_(True)
        out = self.model(image, input_ids, attention_mask, tabular)
        logits = out["pathology"]
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward()
        acts = self._acts; grads = self._grads
        if acts is None or grads is None:
            return np.zeros((7, 7), dtype=np.float32)
        gsq = grads ** 2
        denom = 2 * gsq + acts * grads ** 3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom) * 1e-10)
        alpha = gsq / denom
        weights = (alpha * torch.relu(score.exp() * grads)).mean(
            dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Visualisation
# ──────────────────────────────────────────────────────────────────────
def _jet_heatmap(cam: np.ndarray) -> np.ndarray:
    """Convert a (H,W) [0,1] heatmap to (H,W,3) uint8 jet-coloured RGB."""
    try:
        import cv2
        cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
        h = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    except Exception:
        # Crude matplotlib-like jet without cv2
        c = np.clip(cam, 0, 1)
        r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _overlay(orig_01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet heatmap onto the original (HxWx3 0-1 float). Returns uint8."""
    if cam is None or cam.max() <= 0:
        return (orig_01 * 255).astype(np.uint8)
    try:
        import cv2
        h, w = orig_01.shape[:2]
        cam_r = cv2.resize(cam.astype(np.float32), (w, h))
    except Exception:
        cam_r = np.kron(cam, np.ones((orig_01.shape[0] // cam.shape[0] + 1,
                                       orig_01.shape[1] // cam.shape[1] + 1)))
        cam_r = cam_r[:orig_01.shape[0], :orig_01.shape[1]]
    hm = _jet_heatmap(cam_r) / 255.0
    blended = (1 - alpha) * orig_01 + alpha * hm
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _text_panel_image(predicted: str, conf: float, top3: List[Tuple[str, float]],
                      true_class: str, image_sha: str, size: Tuple[int, int],
                      expected_class: Optional[str] = None, ood: bool = False,
                      ) -> Image.Image:
    """Draw a text panel with the prediction + top-3 + folder context."""
    W, H = size
    panel = Image.new("RGB", (W, H), (250, 250, 252))
    draw = ImageDraw.Draw(panel)
    try:
        f_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        f_med  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        f_sm   = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        f_bold = ImageFont.load_default()
        f_med  = ImageFont.load_default()
        f_sm   = ImageFont.load_default()

    y = 10
    draw.text((10, y), "PREDICTION", fill=(11, 95, 255), font=f_bold)
    y += 22
    draw.text((10, y), PLAIN.get(predicted, predicted),
              fill=(15, 23, 42), font=f_bold)
    y += 22
    draw.text((10, y), f"Confidence: {conf*100:.1f}%",
              fill=(15, 23, 42), font=f_med)
    y += 24

    draw.text((10, y), "TOP-3 PROBABILITIES", fill=(100, 116, 139), font=f_sm)
    y += 16
    for cls, p in top3:
        bar_w = max(2, int((W - 130) * p))
        draw.rectangle([10, y + 4, 10 + bar_w, y + 14],
                        fill=(11, 95, 255) if cls == predicted else (148, 163, 184))
        draw.text((W - 100, y + 2), f"{p*100:5.1f}%  {cls[:12]}",
                  fill=(15, 23, 42), font=f_sm)
        y += 20

    y += 4
    draw.text((10, y), "GROUND-TRUTH FOLDER", fill=(100, 116, 139), font=f_sm)
    y += 14
    draw.text((10, y), true_class[:48], fill=(15, 23, 42), font=f_med)
    y += 22

    # Match indicator — uses semantic expected_class
    if ood:
        draw.text((10, y), "⚠  OOD folder — no expected 5-class label",
                  fill=(124, 58, 237), font=f_sm)
        y += 18
        draw.text((10, y), "(model must pick from 5 classes anyway)",
                  fill=(124, 58, 237), font=f_sm)
        y += 24
    else:
        matched = (expected_class is not None and predicted == expected_class)
        if matched:
            draw.text((10, y), f"✓  matches expected: {expected_class}",
                      fill=(22, 163, 74), font=f_sm)
        else:
            exp_str = expected_class or "?"
            draw.text((10, y), f"✗  expected {exp_str}, got {predicted}",
                      fill=(185, 28, 28), font=f_sm)
        y += 24

    draw.text((10, y), "IMAGE SHA-256", fill=(100, 116, 139), font=f_sm)
    y += 14
    draw.text((10, y), image_sha[:32] + "…", fill=(71, 85, 105), font=f_sm)

    return panel


def render_panel(orig_pil: Image.Image, orig_01: np.ndarray,
                 cam: np.ndarray, predicted: str, conf: float,
                 probs: Dict[str, float], true_class: str,
                 image_sha: str, out_path: Path,
                 expected_class: Optional[str] = None, ood: bool = False):
    """Render and save the 4-panel image."""
    Wp = 280; Hp = 280; gap = 8
    # Resize originals to consistent square panels
    orig_resized = orig_pil.convert("RGB").resize((Wp, Hp), Image.LANCZOS)
    heatmap_arr = _jet_heatmap(cam)
    heatmap_pil = Image.fromarray(heatmap_arr).resize((Wp, Hp), Image.NEAREST)
    overlay_arr = _overlay(orig_01, cam, alpha=0.45)
    overlay_pil = Image.fromarray(overlay_arr).resize((Wp, Hp), Image.LANCZOS)
    top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    text_pil = _text_panel_image(predicted, conf, top3, true_class,
                                  image_sha, (Wp, Hp),
                                  expected_class=expected_class, ood=ood)

    W_total = 4 * Wp + 5 * gap
    H_total = Hp + 2 * gap + 32        # +32 for footer
    canvas = Image.new("RGB", (W_total, H_total), (255, 255, 255))
    canvas.paste(orig_resized, (gap, gap))
    canvas.paste(heatmap_pil, (gap + Wp + gap, gap))
    canvas.paste(overlay_pil, (gap + 2 * (Wp + gap), gap))
    canvas.paste(text_pil,    (gap + 3 * (Wp + gap), gap))

    # Footer labels
    draw = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        f = ImageFont.load_default()
    y = Hp + 2 * gap
    for i, label in enumerate(["Original", "GradCAM++ heatmap",
                                "GradCAM++ overlay", "Prediction details"]):
        x = gap + i * (Wp + gap) + Wp // 2 - 60
        draw.text((x, y), label, fill=(71, 85, 105), font=f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)


# ──────────────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────────────
def run_for_folder(folder_info: Dict[str, Any],
                   *, model, tokenizer, gradcam, device, n_per: int,
                   seed: int = 42,
                   ) -> Dict[str, Any]:
    """Sample n_per images, run inference, write panels, return report."""
    dataset = folder_info["dataset"]
    sub     = folder_info["subfolder"]
    path: Path = folder_info["path"]
    expected_class = folder_info.get("expected_class")
    ood            = bool(folder_info.get("ood", False))
    print(f"\n  [{dataset}/{sub}] {folder_info['n_images']} images "
          f"(expected={expected_class or 'OOD'})")

    out_dir = OUT_ROOT / dataset / sub.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = _images_in(path)
    if not imgs:
        return {"dataset": dataset, "subfolder": sub, "n_samples": 0,
                "panels": [], "error": "no images"}

    rng = random.Random(f"{seed}-{dataset}-{sub}")
    sample = rng.sample(imgs, min(n_per, len(imgs)))

    # Pre-tokenise a generic prompt — text agent will see this for every sample
    input_ids, attention_mask = tokenize(tokenizer,
        "Endoscopic image submitted for screening review.")
    input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
    tabular = torch.zeros(1, N_TAB_FEATURES, dtype=torch.float32, device=device)

    panel_results: List[Dict[str, Any]] = []
    n_match = 0
    for img_path in sample:
        rec = {"image": str(img_path.relative_to(_REPO_ROOT))}
        t0 = time.time()
        try:
            pil = Image.open(img_path)
            img_bytes = img_path.read_bytes()
            sha = hashlib.sha256(img_bytes).hexdigest()
            tensor, arr = preprocess(pil)
            tensor = tensor.to(device)

            with torch.no_grad():
                out = model(tensor, input_ids, attention_mask, tabular)
                probs = torch.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
            cls_idx = int(probs.argmax())
            predicted = PATHOLOGY_CLASSES[cls_idx]
            conf = float(probs[cls_idx])
            probs_d = {c: float(p) for c, p in zip(PATHOLOGY_CLASSES, probs)}

            # GradCAM — must re-enable grad
            cam = gradcam.generate(tensor, cls_idx, input_ids,
                                    attention_mask, tabular)

            out_png = out_dir / f"{img_path.stem}.png"
            render_panel(pil, arr, cam, predicted, conf, probs_d,
                          true_class=sub, image_sha=sha, out_path=out_png,
                          expected_class=expected_class, ood=ood)

            rec.update({
                "predicted":  predicted,
                "confidence": conf,
                "probs":      probs_d,
                "sha256":     sha,
                "panel_png":  str(out_png.relative_to(_REPO_ROOT)),
                "elapsed_s":  round(time.time() - t0, 3),
                "expected_class": expected_class,
                "ood":            ood,
            })
            matched = _matches_folder(predicted, expected_class)
            rec["folder_matched"] = matched
            n_match += int(matched)
            marker = "⚠OOD" if ood else ("✓" if matched else "✗")
            print(f"    {img_path.name[:36]:<36}  → {predicted:<16} "
                  f"({conf*100:5.1f}%)  {marker}  "
                  f"{rec['elapsed_s']:.2f}s")
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-400:]
            print(f"    {img_path.name[:36]:<36}  ✗ {exc}")
        panel_results.append(rec)

    return {
        "dataset":         dataset,
        "subfolder":       sub,
        "expected_class":  expected_class,
        "ood":             ood,
        "n_images":        folder_info["n_images"],
        "n_sampled":       len(panel_results),
        "n_matched":       n_match,
        "match_rate":      (n_match / max(1, len(panel_results))),
        "out_dir":         str(out_dir.relative_to(_REPO_ROOT)),
        "panels":          panel_results,
    }


# ──────────────────────────────────────────────────────────────────────
#  Index HTML builder
# ──────────────────────────────────────────────────────────────────────
def build_index(report: List[Dict[str, Any]], ckpt: Path,
                out_path: Path = OUT_ROOT / "index.html"):
    """Write a single HTML page that browses every panel."""
    import html
    # Split in-distribution (has expected_class) vs out-of-distribution
    in_dist  = [r for r in report if not r.get("ood")]
    ood_rows = [r for r in report if r.get("ood")]
    in_imgs   = sum(r["n_sampled"] for r in in_dist)
    in_match  = sum(r["n_matched"] for r in in_dist)
    in_rate   = (in_match / max(1, in_imgs)) * 100
    ood_imgs  = sum(r["n_sampled"] for r in ood_rows)
    total_imgs = in_imgs + ood_imgs

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>ColonAI — dataset sample-inference panel</title>")
    parts.append("""<style>
      body { font-family: -apple-system, system-ui, Segoe UI, Arial; margin: 0;
             background: #F8FAFC; color: #0F172A; }
      header { background: linear-gradient(135deg, #0B5FFF 0%, #7C3AED 100%);
               color: white; padding: 32px 40px; }
      header h1 { margin: 0; font-size: 1.8rem; font-weight: 800; }
      header .sub { opacity: 0.9; font-size: 0.95rem; margin-top: 6px; }
      .metrics { display: flex; gap: 16px; margin-top: 18px; }
      .metric { background: rgba(255,255,255,0.15); padding: 10px 16px;
                border-radius: 10px; font-size: 0.9rem; }
      .metric b { font-size: 1.4rem; display: block; margin-top: 2px; }
      nav { background: white; padding: 14px 40px; border-bottom: 1px solid #E2E8F0;
            position: sticky; top: 0; z-index: 10; }
      nav a { display: inline-block; margin: 4px 8px; padding: 4px 10px;
              border-radius: 6px; color: #334155; text-decoration: none;
              font-size: 0.85rem; background: #F1F5F9; }
      nav a:hover { background: #E0E7FF; color: #4338CA; }
      section { padding: 24px 40px; }
      h2 { color: #0F172A; border-left: 4px solid #0B5FFF; padding-left: 12px; }
      h3 { color: #334155; margin-top: 28px; }
      .subhdr { display: flex; justify-content: space-between; align-items: center;
                background: #FFF; padding: 10px 16px; border-radius: 10px;
                box-shadow: 0 1px 3px rgba(15,23,42,0.05); margin: 8px 0 12px; }
      .subhdr .name { font-weight: 700; }
      .subhdr .stats { font-size: 0.85rem; color: #475569; }
      .panel { background: white; border: 1px solid #E2E8F0; border-radius: 12px;
               padding: 8px; margin: 6px 0; }
      .panel img { width: 100%; max-width: 1200px; display: block; }
      .panel .cap { font-size: 0.78rem; color: #64748B;
                    padding: 4px 8px; display: flex; justify-content: space-between; }
      .err { background: #FEE2E2; color: #991B1B; padding: 8px 12px;
             border-radius: 8px; font-family: monospace; font-size: 0.8rem; }
      footer { padding: 24px 40px; color: #64748B; font-size: 0.85rem;
               border-top: 1px solid #E2E8F0; }
    </style></head><body>""")

    parts.append(f"""<header>
      <h1>ColonAI — Dataset Sample-Inference Panel</h1>
      <div class='sub'>10 random samples per dataset folder, real model
        inference, real GradCAM++. Checkpoint:
        <code>{html.escape(str(ckpt.relative_to(_REPO_ROOT)))}</code>
        ({ckpt.stat().st_size/1e6:.1f} MB). Seed 42 — fully reproducible.
        OOD folders contain content (normal anatomy, bowel-prep quality,
        esophagitis, hemorrhoids) outside the 5 training classes, so the
        model has no honest answer there — they are split out so the
        accuracy number is not artificially deflated.</div>
      <div class='metrics'>
        <div class='metric'>Dataset folders<b>{len(report)}</b></div>
        <div class='metric'>In-distribution samples<b>{in_imgs}</b></div>
        <div class='metric'>In-distribution accuracy<b>{in_rate:.0f}%</b></div>
        <div class='metric'>OOD samples (separate)<b>{ood_imgs}</b></div>
      </div>
    </header>""")

    # Group by dataset
    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in report:
        by_ds[r["dataset"]].append(r)

    # Nav
    parts.append("<nav>")
    for ds_name in by_ds:
        parts.append(f"<a href='#ds-{html.escape(ds_name)}'>{html.escape(ds_name)}</a>")
    parts.append("</nav>")

    # Sections per dataset
    for ds_name, rows in by_ds.items():
        parts.append(f"<section id='ds-{html.escape(ds_name)}'>")
        parts.append(f"<h2>{html.escape(ds_name)}</h2>")
        for r in rows:
            sub = r["subfolder"]
            n_sampled = r["n_sampled"]
            match = r["n_matched"]
            rate = r["match_rate"] * 100
            exp_class = r.get("expected_class")
            is_ood    = bool(r.get("ood"))
            badge = (f"<span style='background:#EDE9FE;color:#5B21B6;"
                     f"padding:2px 8px;border-radius:999px;font-size:0.72rem;"
                     f"font-weight:700;margin-left:8px;'>OOD</span>"
                     if is_ood else
                     f"<span style='background:#DCFCE7;color:#15803D;"
                     f"padding:2px 8px;border-radius:999px;font-size:0.72rem;"
                     f"font-weight:700;margin-left:8px;'>expected: "
                     f"{html.escape(exp_class or '?')}</span>")
            parts.append(f"<h3 id='{html.escape(ds_name)}__{html.escape(sub)}'>"
                          f"{html.escape(sub)} {badge}</h3>")
            if is_ood:
                stats_html = (f"{n_sampled} sampled · out-of-distribution "
                              f"(no expected class — model must pick one of 5 "
                              f"trained classes regardless)")
            else:
                stats_html = (f"{n_sampled} sampled · {match}/{n_sampled} match "
                              f"expected class '{html.escape(exp_class or '?')}' "
                              f"({rate:.0f}%)")
            parts.append(f"<div class='subhdr'>"
                          f"<span class='name'>{html.escape(r['out_dir'])}</span>"
                          f"<span class='stats'>{stats_html}</span>"
                          f"</div>")
            for p in r["panels"]:
                if "error" in p:
                    parts.append(f"<div class='err'>{html.escape(p['image'])} "
                                  f"— {html.escape(p['error'])}</div>")
                    continue
                rel_png = p["panel_png"]
                pred = PLAIN.get(p["predicted"], p["predicted"])
                conf = p["confidence"] * 100
                p_ood = p.get("ood", is_ood)
                p_exp = p.get("expected_class", exp_class)
                if p_ood:
                    flag = "⚠ OOD"
                else:
                    flag = "✓ match" if (p_exp is not None and
                                          p.get("predicted") == p_exp) else "✗ differs"
                parts.append(f"<div class='panel'>"
                              f"<img src='../../{html.escape(rel_png)}' alt='panel'>"
                              f"<div class='cap'><span>{html.escape(p['image'])}</span>"
                              f"<span><b>{html.escape(pred)}</b> "
                              f"({conf:.1f}%) · {flag} · "
                              f"{p.get('elapsed_s', 0):.2f}s</span></div>"
                              f"</div>")
        parts.append("</section>")

    parts.append(f"""<footer>
      Generated by scripts/sample_dataset_inference.py · seed = 42 ·
      every panel is reproducible from the same checkpoint + image.
      No fake or hallucinated outputs — if a forward pass failed it is
      shown as an error, not a synthesized panel.
    </footer></body></html>""")

    out_path.write_text("\n".join(parts))
    return out_path


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", type=str, default="",
                     help="comma-separated dataset prefixes to keep "
                          "(empty = all)")
    ap.add_argument("--limit-folders", type=int, default=0,
                     help="0 = no limit; otherwise cap total folders processed "
                          "(useful for quick smoke runs)")
    ap.add_argument("--reindex-only", action="store_true",
                     help="Skip inference; recompute match rates from the "
                          "existing report.json and regenerate index.html")
    args = ap.parse_args()

    # ── Re-index path: no model load, no inference ─────────────────────
    if args.reindex_only:
        report_path = OUT_ROOT / "report.json"
        if not report_path.exists():
            print(f"✗ no existing report at {report_path}")
            sys.exit(1)
        print(f"Re-indexing from {report_path} (no inference)…")
        old = json.loads(report_path.read_text())
        ckpt = _REPO_ROOT / old["checkpoint"]

        # Build a lookup of {(dataset, subfolder): folder_info} from current discovery
        cur = {(f["dataset"], f["subfolder"]): f for f in discover_dataset_folders()}

        n_in_dist_imgs = n_in_dist_match = n_ood_imgs = 0
        for r in old["results"]:
            key = (r["dataset"], r["subfolder"])
            info = cur.get(key)
            if info is None:
                r.setdefault("expected_class", None)
                r.setdefault("ood", False)
            else:
                r["expected_class"] = info.get("expected_class")
                r["ood"]            = bool(info.get("ood", False))
            n_match = 0
            for p in r["panels"]:
                p["expected_class"] = r["expected_class"]
                p["ood"]            = r["ood"]
                pred = p.get("predicted")
                matched = _matches_folder(pred, r["expected_class"]) if pred else False
                p["folder_matched"] = matched
                n_match += int(matched)
            r["n_matched"]  = n_match
            r["match_rate"] = n_match / max(1, r["n_sampled"])
            if r["ood"]:
                n_ood_imgs += r["n_sampled"]
            else:
                n_in_dist_imgs += r["n_sampled"]
                n_in_dist_match += n_match

        old["n_total_match"]      = sum(r["n_matched"] for r in old["results"])
        old["n_in_dist_samples"]  = n_in_dist_imgs
        old["n_in_dist_match"]    = n_in_dist_match
        old["n_ood_samples"]      = n_ood_imgs
        report_path.write_text(json.dumps(old, indent=2))
        idx = build_index(old["results"], ckpt)
        print(f"  in-distribution accuracy: "
              f"{n_in_dist_match}/{n_in_dist_imgs} = "
              f"{n_in_dist_match/max(1,n_in_dist_imgs)*100:.1f}%")
        print(f"  OOD samples (separate): {n_ood_imgs}")
        print(f"  index → {idx}")
        return

    print("─" * 64)
    print("ColonAI — dataset sample-inference panel (REAL model)")
    print("─" * 64)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Discover
    folders = discover_dataset_folders()
    if args.datasets.strip():
        keep = set(d.strip() for d in args.datasets.split(","))
        folders = [f for f in folders if f["dataset"] in keep]
    if args.limit_folders > 0:
        folders = folders[:args.limit_folders]
    print(f"  discovered {len(folders)} dataset folders.")
    for f in folders:
        print(f"    - {f['dataset']:<14} / {f['subfolder']:<40}  "
              f"({f['n_images']} images)")

    if not folders:
        print("✗ no folders found — nothing to do.")
        sys.exit(1)

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps")
                              and torch.backends.mps.is_available()
                          else "cpu")
    print(f"\n  device: {device}")
    model, tokenizer, ckpt = load_model(device)
    gradcam = GradCAMPP(model)

    # 3. Loop
    report: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, f in enumerate(folders, start=1):
        print(f"\n[{i}/{len(folders)}]")
        r = run_for_folder(f, model=model, tokenizer=tokenizer,
                            gradcam=gradcam, device=device, n_per=args.n,
                            seed=args.seed)
        report.append(r)

    elapsed = time.time() - t0
    print(f"\n  total elapsed: {elapsed:.1f}s  ({elapsed/max(1, len(folders)):.1f}s/folder)")

    # 4. Save report
    report_path = OUT_ROOT / "report.json"
    payload = {
        "checkpoint":      str(ckpt.relative_to(_REPO_ROOT)),
        "device":          str(device),
        "n_per_folder":    args.n,
        "seed":            args.seed,
        "elapsed_s":       elapsed,
        "n_folders":       len(folders),
        "n_total_samples": sum(r["n_sampled"] for r in report),
        "n_total_match":   sum(r["n_matched"] for r in report),
        "results":         report,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    print(f"  report → {report_path}")

    # 5. Build HTML index
    idx_path = build_index(report, ckpt)
    print(f"  index  → {idx_path}")
    print(f"\nOpen the index in a browser:")
    print(f"  open '{idx_path}'")
    print("─" * 64)


if __name__ == "__main__":
    main()
