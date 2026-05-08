"""Image-statistics atypicality detector for ColonAI.

The trained classifier knows 5 screening-stage classes only.  It cannot output
'advanced cancer' because that class doesn't exist in its training distribution
(HyperKvasir + CVC-ClinicDB are screening-stage only).  Instead, this module
inspects the raw pixels and looks for visual signatures that are *clinically*
suspicious for advanced lesions or, conversely, signatures that suggest a
clean / normal-looking mucosa.

Signals measured (all in [0,1]):
  • dark_cavity      — fraction of very dark pixels (cavitation, sloughed tissue)
  • red_necrosis     — fraction of intensely red, low-blue pixels (bleeding /
                       fungating tumour / necrotic ulcer)
  • edge_disorder    — Sobel edge density × randomness (mass effect, irregularity)
  • colour_disorder  — variance of hue across the visible mucosa
  • mucosal_uniformity — coherence of the dominant pink colour (clean colon)
  • brightness_balance — central illumination is well-distributed (a good shot)

These produce two opposing scores:
  • atypicality      [0..1]  — higher = more like advanced lesion / OOD
  • normal_score     [0..1]  — higher = more like a clean, healthy colon

The thresholds are conservative — better to flag a normal image as "review"
than to miss a concerning one.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass
class ImageReadout:
    atypicality: float       # 0..1
    normal_score: float      # 0..1
    verdict: str             # one of: "clean", "screening_finding", "atypical_concerning"
    confidence: float        # 0..1 — how sure we are about the verdict
    signals: Dict[str, float]
    reasons: list            # human-readable bullets

    def to_dict(self) -> Dict:
        return {**asdict(self)}


# ──────────────────────────────────────────────────────────────────
# Helpers — vectorised numpy / cv2 (cv2 already in app deps)
# ──────────────────────────────────────────────────────────────────

def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce a float-or-uint8 numpy array (H,W,3) into uint8 RGB."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim != 3:
        return None
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    return a


def _hsv(arr_rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)


def _gray(arr_rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)


# ──────────────────────────────────────────────────────────────────
# Individual signal computations
# ──────────────────────────────────────────────────────────────────

def _dark_cavity(rgb: np.ndarray) -> float:
    """Fraction of pixels with very low value (V<0.18 in HSV).
    Endoscopy normally has central darkness from the lumen — but excessive
    dark area (>22%) suggests cavitation / mass shadow / poor mucosa."""
    hsv = _hsv(rgb)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    frac = float((v < 0.18).mean())
    # Map [0.05 .. 0.40] -> [0..1] (some darkness is normal, lots is bad)
    return float(np.clip((frac - 0.05) / 0.35, 0.0, 1.0))


def _red_necrosis(rgb: np.ndarray) -> float:
    """Fraction of pixels that are *clinically* red — i.e. deeply red with
    almost no green and almost no blue (bleeding / necrotic / fungating tissue).

    Healthy mucosa is *pinkish* — the green channel is typically 35–55 % of
    red and the blue channel is 25–45 % of red. Necrotic / clotted-blood
    tissue has green and blue both very low (<15 %), which is a reliable
    discriminator vs normal mucosa.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    # Very-deep-red pixels: red dominant with green and blue both very low.
    # Endoscopy mucosa virtually never satisfies all three of these together.
    deep_red = (r > 0.45) & (g < 0.18) & (b < 0.18)
    frac_deep = float(deep_red.mean())

    # Bright-red bleeding: high red, dramatically lower green/blue
    bright_red = (r > 0.65) & (g < 0.30) & (b < 0.30) & (r - g > 0.40)
    frac_bright = float(bright_red.mean())

    score = 0.55 * frac_deep + 0.45 * frac_bright
    # Map [0.05 .. 0.40] -> [0..1] — needs a meaningful patch, not stray pixels
    return float(np.clip((score - 0.05) / 0.35, 0.0, 1.0))


def _edge_disorder(rgb: np.ndarray) -> float:
    """High-frequency, randomly-oriented edges suggest disorganized tissue
    (mass effect, ulceration).  Use Sobel magnitude variance + entropy."""
    import cv2
    gray = _gray(rgb).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((mag > 0.10).mean())
    edge_var = float(mag.std())
    # Map [0.10 .. 0.45] for density, [0.04 .. 0.18] for variance
    d_score = float(np.clip((edge_density - 0.10) / 0.35, 0.0, 1.0))
    v_score = float(np.clip((edge_var - 0.04) / 0.14, 0.0, 1.0))
    return 0.6 * d_score + 0.4 * v_score


def _colour_disorder(rgb: np.ndarray) -> float:
    """Hue variance across the image.  Healthy mucosa has tight hue clustering
    (pink); diseased tissue mixes red, white (ulcer) and dark-red (bleeding)."""
    hsv = _hsv(rgb).astype(np.float32)
    h = hsv[:, :, 0]
    # Use circular variance because hue wraps
    angles = h / 180.0 * 2 * math.pi
    cx = float(np.cos(angles).mean())
    sx = float(np.sin(angles).mean())
    R = math.sqrt(cx * cx + sx * sx)
    # 1 - R is circular variance; healthy mucosa often has R ~ 0.85+
    var_circular = 1.0 - R
    # Map [0.15 .. 0.55] -> [0..1]
    return float(np.clip((var_circular - 0.15) / 0.40, 0.0, 1.0))


def _mucosal_uniformity(rgb: np.ndarray) -> float:
    """Inverse of colour disorder, in [0..1].  Healthy mucosa scores high."""
    return float(1.0 - _colour_disorder(rgb))


def _brightness_balance(rgb: np.ndarray) -> float:
    """Endoscopy with a well-positioned scope shows centrally bright mucosa
    with a small dark lumen — a healthy frame.  Score the central-edge
    brightness ratio."""
    g = _gray(rgb).astype(np.float32) / 255.0
    h, w = g.shape
    cx, cy = w // 2, h // 2
    rr = min(h, w) // 6
    centre = g[cy - rr:cy + rr, cx - rr:cx + rr].mean()
    edges = (g[:rr].mean() + g[-rr:].mean() +
             g[:, :rr].mean() + g[:, -rr:].mean()) / 4.0
    diff = abs(edges - centre)  # well-balanced ~ 0.05–0.20
    # Score 1.0 when diff is tiny, 0.0 when diff is extreme
    return float(np.clip(1.0 - (diff / 0.45), 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────
# Top-level: compute readout
# ──────────────────────────────────────────────────────────────────

def compute_image_readout(arr: np.ndarray) -> ImageReadout:
    """Compute the full image-statistics readout for an endoscopy image.

    `arr` is expected as a numpy array, either uint8 (H,W,3) or float in
    [0,1]. The function is robust to either.
    """
    rgb = _to_uint8_rgb(arr)
    if rgb is None or rgb.size < 100:
        return ImageReadout(
            atypicality=0.0, normal_score=0.0, verdict="clean",
            confidence=0.0,
            signals={}, reasons=["No image available."],
        )

    sig = {
        "dark_cavity":         _dark_cavity(rgb),
        "red_necrosis":        _red_necrosis(rgb),
        "edge_disorder":       _edge_disorder(rgb),
        "colour_disorder":     _colour_disorder(rgb),
        "mucosal_uniformity":  _mucosal_uniformity(rgb),
        "brightness_balance":  _brightness_balance(rgb),
    }

    # Atypicality — weighted sum of "this looks bad" signals.
    # Endoscopy normally has a dark central lumen, so weight dark_cavity LOW;
    # red_necrosis (the most cancer-specific cue) is weighted heaviest.
    atyp = (
        0.50 * sig["red_necrosis"] +
        0.20 * sig["edge_disorder"] +
        0.15 * sig["colour_disorder"] +
        0.15 * sig["dark_cavity"]
    )
    # Normal score — clean colon signature
    normal = (
        0.55 * sig["mucosal_uniformity"] +
        0.25 * sig["brightness_balance"] +
        0.20 * (1.0 - sig["red_necrosis"])
    )

    # Verdict bands — calibrated against HyperKvasir samples.
    # Note: every HyperKvasir image is a *finding* (polyps, colitis, Barrett's,
    # etc.). The model has NO 'normal anatomy' class. So this readout is NOT
    # 'is the patient healthy' — it is 'do the raw pixels show signs of an
    # ADVANCED lesion that might be outside the model's screening-stage scope'.
    if atyp >= 0.55 and sig["red_necrosis"] >= 0.45:
        verdict = "atypical_concerning"
        confidence = float(min(1.0, (atyp - 0.55) / 0.25 + 0.55))
    elif normal >= 0.78 and atyp < 0.40 and sig["red_necrosis"] < 0.30:
        # Pixel signature consistent with normal mucosa or a small focal
        # screening finding — the AI's *class* prediction still matters.
        verdict = "consistent_screening"
        confidence = float(min(1.0, (normal - 0.78) / 0.15 + 0.55))
    else:
        verdict = "uncertain"
        confidence = 0.55

    # Human-readable bullets
    reasons = []
    if sig["dark_cavity"] > 0.55:
        reasons.append(("amber",
            f"Large dark area in the image ({sig['dark_cavity']*100:.0f}%) — "
            f"could indicate cavitation, mass shadow, or poor lumen visualisation."))
    elif sig["dark_cavity"] < 0.25:
        reasons.append(("green",
            "Lumen and mucosa are well-distinguished — no abnormal dark cavitation."))
    if sig["red_necrosis"] > 0.55:
        reasons.append(("red",
            f"Strong red / low-blue dominance ({sig['red_necrosis']*100:.0f}%) — "
            f"clinically suspicious for ulceration, fungating tissue or bleeding."))
    elif sig["red_necrosis"] < 0.25:
        reasons.append(("green",
            "No abnormally bleeding or necrotic-looking tissue picked up."))
    if sig["edge_disorder"] > 0.55:
        reasons.append(("amber",
            f"Highly disorganised edge pattern ({sig['edge_disorder']*100:.0f}%) — "
            f"could reflect mass effect, ulceration or scope artefact."))
    if sig["mucosal_uniformity"] > 0.65:
        reasons.append(("green",
            "Mucosa shows a coherent pink hue — typical of healthy bowel lining."))
    if sig["brightness_balance"] > 0.65:
        reasons.append(("green",
            "Image is well-illuminated and centred — a clean diagnostic frame."))
    if not reasons:
        reasons.append(("amber",
            "Image features are mixed — falls between normal and concerning."))

    return ImageReadout(
        atypicality=float(atyp),
        normal_score=float(normal),
        verdict=verdict,
        confidence=float(confidence),
        signals={k: float(v) for k, v in sig.items()},
        reasons=reasons,
    )
