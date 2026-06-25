"""ColonAI — CT rectal-tumor segmentation specialist (CARE).

A SEPARATE modality path: outlines the rectal tumour on a pelvic/abdominal CT
slice using a U-Net (resnet34 encoder) trained on the CARE dataset (398
patients). Honest held-out (CARE *official* test split, patient-disjoint):

    IoU 0.7698  (95% CI 0.7532–0.786) · Dice 0.8497 · sens@0.5 0.895 · n=600

CC BY-NC 4.0 — research / non-commercial use only.

────────────────────────────────────────────────────────────────────────────
HONESTY NOTES — read before trusting any output (these are not boilerplate):

  • This is a SEGMENTER, not a detector / triage tool. Every CARE training
    slice contained a tumour, so the model assumes the input IS a rectal /
    pelvic CT slice that contains disease. It will outline the most
    tumour-like region on essentially ANY grayscale image — it CANNOT decide
    whether a scan contains cancer.
  • Absence of a segmented region is NOT exclusion of disease.
  • In-distribution to CARE (CT, that cohort / those scanners). It is NOT
    externally validated on other centres or modalities. Decision-support
    only — a radiologist must confirm every finding.
  • Does NOT touch the colonoscopy pipeline. Fully fail-open: if the weights
    or the deps (segmentation-models-pytorch / albumentations) are missing,
    every entry point returns "unavailable" and the caller keeps its existing
    behaviour (the radiology rejection).

The result ALWAYS carries requires_human_review=True.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Where the trained weights live (same dir as the other specialist heads).
_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "unified_multimodal_v2" / "care_ct_seg.pt"
)
_METRICS_PATH = _DEFAULT_WEIGHTS.with_name("care_ct_seg_metrics.json")

# ImageNet normalisation — must match train_care_ct_colab.py exactly.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_CAVEATS = [
    "This is a tumour SEGMENTER, not a cancer detector — it assumes the image "
    "is a rectal/pelvic CT slice and outlines the most tumour-like region. It "
    "cannot tell you whether a scan actually contains cancer.",
    "Absence of a highlighted region is NOT exclusion of disease.",
    "Validated in-distribution on the CARE cohort only (held-out IoU ≈ 0.77); "
    "not externally validated on other scanners or centres.",
    "Decision-support only — a radiologist must confirm every finding.",
]

# Lazy, thread-safe singleton so we don't load 90 MB unless a CT image arrives.
_lock = threading.Lock()
_model = None          # the loaded torch model (or None)
_load_attempted = False
_meta: Dict[str, Any] = {}   # checkpoint metadata (encoder, imgsz, iou…)


def metrics() -> Dict[str, Any]:
    """Return the honest held-out metrics dict (from the metrics JSON), or {}."""
    try:
        import json
        if _METRICS_PATH.exists():
            return json.loads(_METRICS_PATH.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CT seg metrics unreadable: %s", exc)
    return {}


def ct_segmenter_available(weights: Optional[Path] = None) -> bool:
    """True iff the weights file exists (cheap check; does not import torch)."""
    return Path(weights or _DEFAULT_WEIGHTS).exists()


def load_ct_segmenter(weights: Optional[Path] = None, device: str = "cpu"):
    """Load (and cache) the CARE U-Net. Returns the model, or None if anything
    is missing. Never raises — fail-open by contract."""
    global _model, _load_attempted, _meta
    with _lock:
        if _model is not None:
            return _model
        if _load_attempted:
            return _model  # previous attempt failed; don't retry every call
        _load_attempted = True

        wpath = Path(weights or _DEFAULT_WEIGHTS)
        if not wpath.exists():
            logger.info("CT segmenter weights not found at %s — disabled.", wpath)
            return None
        try:
            import torch
            import segmentation_models_pytorch as smp
        except Exception as exc:
            logger.warning("CT segmenter deps missing (%s) — disabled.", exc)
            return None
        try:
            ckpt = torch.load(str(wpath), map_location=device, weights_only=False)
            encoder = ckpt.get("encoder", "resnet34")
            _meta = {
                "encoder": encoder,
                "imgsz": int(ckpt.get("imgsz", 384)),
                "iou": float(ckpt.get("iou", 0.0)),
                "tumor_labels": ckpt.get("tumor_labels", "non-zero"),
            }
            net = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
            net.load_state_dict(ckpt["state_dict"])
            net.eval().to(device)
            _model = net
            logger.info("CT segmenter loaded (encoder=%s imgsz=%d iou=%.4f).",
                        encoder, _meta["imgsz"], _meta["iou"])
            return _model
        except Exception as exc:
            logger.warning("CT segmenter failed to load (%s) — disabled.", exc)
            return None


def _to_ct_uint8_3ch(image) -> Optional[np.ndarray]:
    """Coerce any CT slice (PIL / uint8 / float / RGB) to a uint8 3-channel
    array via per-image min-max — mirroring the training-time conversion of HU
    floats to display range."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)        # CT is grayscale; collapse any RGB
    elif a.ndim != 2:
        return None
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo + 1e-6) * 255.0 if hi > lo else np.zeros_like(a)
    u = a.astype(np.uint8)
    return np.stack([u, u, u], axis=-1)


def _overlay(base_u8_3ch: np.ndarray, mask: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    """Red tumour overlay + contour on the grayscale slice (red distinguishes it
    from the colonoscopy pipeline's green polyp overlay)."""
    img = np.ascontiguousarray(base_u8_3ch.astype(np.uint8))
    binm = (mask >= 0.5).astype(np.uint8)
    if binm.sum() == 0:
        return img
    try:
        import cv2
        red = np.zeros_like(img); red[..., 0] = 255
        out = np.where(binm[..., None] == 1,
                       (alpha * red + (1 - alpha) * img).astype(np.uint8), img)
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 0, 0), 2)
        return out
    except Exception:
        return img


def segment_ct(image, device: str = "cpu",
               weights: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Segment the rectal tumour on a single CT slice.

    `image` : PIL.Image | np.ndarray (H,W) or (H,W,3), any dtype.
    Returns a JSON-ish dict, or None if the specialist is unavailable
    (fail-open — the caller then keeps its existing behaviour).

    The returned dict (when available):
        {
          available, tumor_area_frac, tumor_present_hint, mean_prob, max_prob,
          overlay (uint8 HxWx3), mask (float32 HxW), metrics, caveats,
          requires_human_review (always True), provenance
        }
    """
    model = load_ct_segmenter(weights=weights, device=device)
    if model is None:
        return None

    u3 = _to_ct_uint8_3ch(image)
    if u3 is None or u3.size < 100:
        return None
    H, W = u3.shape[:2]

    try:
        import torch
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except Exception as exc:
        logger.warning("CT segmenter inference deps missing (%s).", exc)
        return None

    try:
        imgsz = int(_meta.get("imgsz", 384))
        tf = A.Compose([
            A.Resize(imgsz, imgsz),
            A.CLAHE(p=1.0),
            A.Normalize(_MEAN, _STD),
            ToTensorV2(),
        ])
        x = tf(image=u3)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob_small = torch.sigmoid(model(x))[0, 0].cpu().numpy().astype(np.float32)

        import cv2
        prob_full = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)
        mask_full = (prob_full >= 0.5).astype(np.float32)

        area_frac = float(mask_full.mean())
        flagged = mask_full >= 0.5
        mean_prob = float(prob_full[flagged].mean()) if flagged.any() else 0.0
        max_prob = float(prob_full.max())
        # A *hint* only — never a diagnosis. A meaningful region (≥0.3% of slice).
        present_hint = bool(area_frac >= 0.003)

        # Tight bounding box of the detected region (for a zoom inset). None if empty.
        ys, xs = np.where(flagged)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1) \
            if ys.size else None

        return {
            "available": True,
            "tumor_area_frac": area_frac,
            "tumor_present_hint": present_hint,
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "overlay": _overlay(u3, mask_full),
            "mask": mask_full,
            "prob_map": prob_full,        # full-res probability (for the heatmap)
            "bbox": bbox,                 # (y0,x0,y1,x1) of the detected region or None
            "base_rgb": u3,               # the grayscale CT as 3ch (for zoom crops)
            "metrics": metrics() or {"mean_iou": _meta.get("iou")},
            "caveats": list(_CAVEATS),
            "requires_human_review": True,
            "provenance": {
                "model": "CARE U-Net (resnet34) — CT rectal-tumour segmentation",
                "encoder": _meta.get("encoder", "resnet34"),
                "imgsz": imgsz,
                "held_out_iou": _meta.get("iou"),
                "dataset": "CARE rectal CT (CC BY-NC 4.0)",
                "modality": "CT",
            },
        }
    except Exception as exc:
        logger.warning("CT segmentation failed (%s) — returning None.", exc)
        return None


def reset_cache() -> None:
    """Test helper — clear the cached model so a fresh load can be exercised."""
    global _model, _load_attempted, _meta
    with _lock:
        _model, _load_attempted, _meta = None, False, {}


if __name__ == "__main__":
    # Self-test against the REAL weights (no fabricated data — a synthetic
    # grayscale slice with a bright blob just exercises the plumbing).
    import sys
    print("weights present:", ct_segmenter_available(), "→", _DEFAULT_WEIGHTS)
    print("metrics:", metrics())

    if not ct_segmenter_available():
        print("SKIP — weights not present; cannot run inference self-test.")
        sys.exit(0)

    m = load_ct_segmenter()
    assert m is not None, "model should load when weights + deps are present"

    # synthetic 320x256 grayscale 'slice' with a soft bright lesion-like blob
    rng = np.random.default_rng(0)
    base = (rng.normal(90, 18, (320, 256)).clip(0, 255)).astype(np.uint8)
    yy, xx = np.mgrid[0:320, 0:256]
    blob = np.exp(-(((yy - 160) ** 2) / (2 * 30 ** 2) + ((xx - 128) ** 2) / (2 * 28 ** 2)))
    base = np.clip(base + (blob * 120), 0, 255).astype(np.uint8)

    out = segment_ct(base)
    assert out is not None, "segment_ct should return a dict when available"
    assert out["available"] is True
    assert out["requires_human_review"] is True
    assert out["overlay"].shape == (320, 256, 3), out["overlay"].shape
    assert out["mask"].shape == (320, 256), out["mask"].shape
    assert 0.0 <= out["tumor_area_frac"] <= 1.0
    assert isinstance(out["caveats"], list) and len(out["caveats"]) >= 3
    print(f"OK — segment_ct ran. area_frac={out['tumor_area_frac']:.4f} "
          f"mean_prob={out['mean_prob']:.3f} max_prob={out['max_prob']:.3f} "
          f"present_hint={out['tumor_present_hint']}")
    print("held-out IoU (provenance):", out["provenance"]["held_out_iou"])
    print("CT specialist self-test passed.")
