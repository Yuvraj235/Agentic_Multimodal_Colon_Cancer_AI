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
import cv2


@dataclass
class ImageReadout:
    atypicality: float       # 0..1
    normal_score: float      # 0..1
    verdict: str             # one of: "not_endoscopy", "consistent_screening",
                             #         "atypical_concerning", "uncertain"
    confidence: float        # 0..1 — how sure we are about the verdict
    signals: Dict[str, float]
    reasons: list            # human-readable bullets
    is_endoscopy: bool = True              # NEW — hard gate before the model runs
    endoscopy_score: float = 1.0           # NEW — 0..1 endoscopy-ness score

    def to_dict(self) -> Dict:
        return {**asdict(self)}


# ──────────────────────────────────────────────────────────────────
# ENDOSCOPY-IMAGE GATE  — rejects non-medical / non-colonoscopy inputs
# ──────────────────────────────────────────────────────────────────

def _endoscopy_score(rgb: np.ndarray) -> Dict[str, float]:
    """Score how 'endoscopy-like' an image is, on six robust signals:

      1. red_dominance   — fraction of pixels where R is clearly the strongest channel
      2. low_blue        — fraction of pixels with low blue (no sky, no UI chrome)
      3. tissue_hue      — fraction with hue in red-pink-orange band (HSV)
      4. moderate_satur  — mean saturation in tissue range (0.20–0.85)
      5. natural_texture — Sobel std in tissue range (not flat, not extreme noise)
      6. not_grayscale   — RGB channels are not nearly identical (rejects B&W)

    Each is in [0,1].  Their average is the endoscopy_score — high = looks like a
    real colonoscopy frame, low = looks like something else (cat photo,
    landscape, screenshot, X-ray, drawing, etc.).
    """
    import cv2
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return {"red_dominance": 0.0, "low_blue": 0.0, "tissue_hue": 0.0,
                "moderate_satur": 0.0, "natural_texture": 0.0, "not_grayscale": 0.0}

    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    # 1. Red dominance — endoscopy is almost always R > G and R > B
    red_dom_mask = (r > g + 0.04) & (r > b + 0.04)
    red_dominance = float(red_dom_mask.mean())
    # Map [0.35 .. 0.85] -> [0..1].  Real endoscopy is usually 0.70+.
    s1 = float(np.clip((red_dominance - 0.35) / 0.50, 0.0, 1.0))

    # 2. Low-blue — endoscopy almost never has bright blue (no sky / sea / UI)
    low_blue_mask = b < 0.55
    low_blue = float(low_blue_mask.mean())
    s2 = float(np.clip((low_blue - 0.60) / 0.35, 0.0, 1.0))

    # 3. Tissue hue band — H in [0..25] or [150..180] (OpenCV hue 0..180)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    tissue_band = ((hue <= 25) | (hue >= 150)) & (sat > 0.15) & (val > 0.10)
    tissue_hue = float(tissue_band.mean())
    s3 = float(np.clip((tissue_hue - 0.30) / 0.50, 0.0, 1.0))

    # 4. Moderate saturation — endoscopy is saturated but not neon
    mean_sat = float(sat.mean())
    if 0.20 <= mean_sat <= 0.85:
        s4 = 1.0
    elif mean_sat < 0.20:
        s4 = float(mean_sat / 0.20)         # too washed out
    else:
        s4 = float(max(0.0, 1.0 - (mean_sat - 0.85) / 0.15))   # too neon

    # 5. Natural tissue texture — Sobel std in [0.06 .. 0.20]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    tex = float(mag.std())
    if 0.04 <= tex <= 0.22:
        s5 = 1.0
    elif tex < 0.04:
        s5 = float(tex / 0.04)              # too flat (solid colour / drawing)
    else:
        s5 = float(max(0.0, 1.0 - (tex - 0.22) / 0.15))   # too noisy (text/UI)

    # 6. Not grayscale — channel differences average must be > 0.04
    chan_diff = float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)
    s6 = float(np.clip((chan_diff - 0.02) / 0.10, 0.0, 1.0))

    return {
        "red_dominance":   s1,
        "low_blue":        s2,
        "tissue_hue":      s3,
        "moderate_satur":  s4,
        "natural_texture": s5,
        "not_grayscale":   s6,
    }


def _modality_label(rgb: np.ndarray) -> Dict:
    """Best-effort image-modality classification from pixel statistics.

    Heuristic (NOT a learned classifier) used to (a) give an HONEST message about
    what kind of image was uploaded and (b) avoid wrongly rejecting narrow-band
    imaging (NBI) colonoscopy, which is green/teal and would fail a red-dominance
    gate. The robust replacement is a learned modality classifier trained on real
    CT/MRI/NBI samples (see plan Part 1b). Returns {modality, detail, stats}.

    Categories:
      colonoscopy_wl       white-light colonoscopy (red/pink, textured)
      colonoscopy_nbi      narrow-band imaging (dark, green/teal dominant, textured)
      radiology_grayscale  X-ray / CT / MRI (near-grayscale)
      photo_or_screenshot  bright / blue-heavy / flat or noisy (not internal tissue)
      unknown              matches none of the colonoscopy signatures confidently
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    mean_val   = float(rgb_f.mean())
    chan_diff  = float((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0)
    red_dom    = float(((r > g + 0.04) & (r > b + 0.04)).mean())
    green_dom  = float(((g > r + 0.04) & (g >= b - 0.02)).mean())   # NBI: green/teal over red
    blue_bright = float(((b > 0.55) & (b > r)).mean())              # sky / UI chrome
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tex = float(np.sqrt(gx * gx + gy * gy).std())
    stats = {"mean_val": round(mean_val, 3), "chan_diff": round(chan_diff, 3),
             "red_dom": round(red_dom, 3), "green_dom": round(green_dom, 3),
             "blue_bright": round(blue_bright, 3), "texture": round(tex, 3)}

    # 1. Near-grayscale → radiology (X-ray / CT / MRI). 0.06 tolerates the slight
    #    channel drift from JPEG compression of a genuinely grayscale scan, while
    #    colour endoscopy sits well above (chan_diff ~0.15+).
    if chan_diff < 0.06:
        return {"modality": "radiology_grayscale", "stats": stats,
                "detail": "near-grayscale image — looks like a radiology scan "
                          "(X-ray / CT / MRI), not a colour endoscopy frame"}
    # 2. NBI colonoscopy — dark, green/teal-dominant over red, with real texture
    if mean_val < 0.50 and green_dom > 0.45 and red_dom < 0.30 and 0.03 <= tex <= 0.30:
        return {"modality": "colonoscopy_nbi", "stats": stats,
                "detail": "dark, green/teal-dominant, textured frame — consistent "
                          "with narrow-band imaging (NBI) colonoscopy"}
    # 3. White-light colonoscopy — red/pink-dominant, naturally textured
    if red_dom > 0.45 and 0.03 <= tex <= 0.30:
        return {"modality": "colonoscopy_wl", "stats": stats,
                "detail": "red/pink, textured frame — consistent with white-light colonoscopy"}
    # 4. Bright / blue-heavy / flat or noisy → photo, drawing, or screenshot
    if blue_bright > 0.20 or mean_val > 0.78 or tex < 0.02 or tex > 0.35:
        return {"modality": "photo_or_screenshot", "stats": stats,
                "detail": "bright / blue-heavy / flat or noisy image — looks like a "
                          "regular photo, drawing, or screenshot, not internal tissue"}
    return {"modality": "unknown", "stats": stats,
            "detail": "does not confidently match colonoscopy (white-light or NBI) signatures"}


def is_endoscopy_image(arr: np.ndarray, threshold: float = 0.55) -> Dict:
    """Hard gate that decides if an uploaded image is a colonoscopy frame.

    Returns a dict with:
       is_endoscopy    : bool   — accepts white-light AND NBI colonoscopy
       score           : float in [0..1]  (red-based endoscopy-ness)
       modality        : str    — best-effort modality label (see _modality_label)
       modality_detail : str    — human-readable explanation of the modality
       signals         : dict of the 6 sub-signals
       reasons         : human-readable bullets explaining the decision

    Use BEFORE calling compute_image_readout().  If is_endoscopy is False,
    the caller should refuse to run the trained classifier on this image.
    """
    rgb = _to_uint8_rgb(arr)
    if rgb is None or rgb.size < 100:
        return {"is_endoscopy": False, "score": 0.0, "modality": "invalid",
                "modality_detail": "no valid image data",
                "signals": {}, "reasons": ["No valid image data."]}

    sig = _endoscopy_score(rgb)
    score = float(np.mean(list(sig.values())))
    mod = _modality_label(rgb)
    modality = mod["modality"]

    # Accept white-light OR NBI colonoscopy. White-light is also accepted via the
    # red-based score gate (covers atypical frames the modality heuristic misses).
    is_endo = (modality in ("colonoscopy_wl", "colonoscopy_nbi")) or (score >= threshold)

    # Build human-readable reasons
    reasons = [mod["detail"]]
    if is_endo:
        if modality == "colonoscopy_nbi":
            reasons.append("Accepted as narrow-band imaging (NBI) colonoscopy.")
        else:
            reasons.append("Input looks like a colonoscopy frame.")
    else:
        # Explain, by modality, why it was not accepted (honest, not just 'rejected')
        if modality == "radiology_grayscale":
            reasons.append("ColonAI analyses colonoscopy images, not radiology scans — "
                           "an X-ray/CT/MRI cannot be assessed for polyps here.")
        elif modality == "photo_or_screenshot":
            reasons.append("This does not look like internal tissue — please upload a "
                           "real colonoscopy/endoscopy frame.")
        else:
            if sig["red_dominance"] < 0.30:
                reasons.append("Not red-dominant — colonoscopy frames are usually reddish/pink.")
            if sig["not_grayscale"] < 0.30:
                reasons.append("Image appears nearly grayscale — endoscopy is in colour.")
            if sig["natural_texture"] < 0.40:
                reasons.append("Texture is too flat or too noisy for endoscopy.")

    return {
        "is_endoscopy":    is_endo,
        "score":           score,
        "modality":        modality,
        "modality_detail": mod["detail"],
        "stats":           mod["stats"],
        "signals":         sig,
        "reasons":         reasons,
    }


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
            is_endoscopy=False, endoscopy_score=0.0,
        )

    # ── HARD GATE: is this even an endoscopy image? ─────────────────
    gate = is_endoscopy_image(rgb, threshold=0.55)
    if not gate["is_endoscopy"]:
        endoscopy_reasons = [("red", r) for r in gate["reasons"]]
        endoscopy_reasons.append(("red",
            f"Endoscopy-likeness score is only {gate['score']*100:.0f}% — below the 55% threshold "
            f"required to run the trained model."))
        endoscopy_reasons.append(("amber",
            "The trained classifier is only valid on real colonoscopy images. "
            "Please upload a colonoscopy / endoscopy frame."))
        return ImageReadout(
            atypicality=0.0,
            normal_score=0.0,
            verdict="not_endoscopy",
            confidence=float(1.0 - gate["score"]),  # confident it's NOT endoscopy
            signals=gate["signals"],
            reasons=endoscopy_reasons,
            is_endoscopy=False,
            endoscopy_score=float(gate["score"]),
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
        is_endoscopy=True,
        endoscopy_score=float(gate["score"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invasive / advanced-disease detector (pixel-statistics override)
# ─────────────────────────────────────────────────────────────────────────────
def detect_advanced_lesion(arr: np.ndarray) -> Dict:
    """Detect visual signs of advanced colorectal disease that the
    5-class classifier (polyps / UC / Barrett's / therapeutic) was NEVER
    trained on — fungating mass, deep ulceration, heavy contact bleeding,
    nodular irregular surface.

    When this returns is_advanced=True, the safety policy should OVERRIDE
    the classifier's confident call with "Atypical lesion — urgent
    endoscopist review required" regardless of what class was predicted.
    This stops the model from confidently mis-labelling an invasive
    tumour as 'polyps' just because polyps is the closest class it knows.

    Returns:
        {
          is_advanced:    bool,
          severity:       float in [0..1]   higher = more advanced
          signals:        dict of sub-scores
          reasons:        list[str]         human-readable bullets
        }
    """
    rgb = _to_uint8_rgb(arr)
    if rgb is None or rgb.size < 100:
        return {"is_advanced": False, "severity": 0.0,
                "signals": {}, "reasons": []}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 1. Deep ulceration / cavitation: large CONNECTED dark patches
    #    (not just shadows — sustained low-V regions ≥ 1 % of frame)
    dark = (V < 60).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    img_area = float(rgb.shape[0] * rgb.shape[1])
    biggest_dark = (float(stats[1:, cv2.CC_STAT_AREA].max())
                    if n > 1 else 0.0) / img_area
    s_ulceration = float(min(1.0, biggest_dark * 8))   # 12.5 %+ → 1.0

    # 2. Heavy contact bleeding: BRIGHT red-saturated pixels (fresh blood
    #    is different from healthy mucosa — high saturation + low hue + high value)
    blood = (((H < 8) | (H > 170)) & (S > 130) & (V > 100)).astype(np.uint8)
    s_bleeding = float(min(1.0, blood.mean() * 4))     # 25 %+ → 1.0

    # 3. Nodular / irregular surface: high local-variance texture
    #    Compute variance of Laplacian in 16×16 blocks; take the 90th-pct.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    blocks = []
    for y in range(0, gray.shape[0] - 16, 16):
        for x in range(0, gray.shape[1] - 16, 16):
            blocks.append(float(lap[y:y+16, x:x+16].var()))
    if blocks:
        nodular = float(np.percentile(blocks, 90))
        s_nodular = float(min(1.0, nodular / 5000.0))  # >5000 var → 1.0
    else:
        s_nodular = 0.0

    # 4. Mass effect: very low-saturation white-ish reflection patch
    #    (large saturation drop indicates fungating tissue / reflective surface)
    bright_dull = ((V > 200) & (S < 40)).astype(np.uint8)
    s_mass = float(min(1.0, bright_dull.mean() * 6))    # 16 %+ → 1.0

    signals = {
        "deep_ulceration": s_ulceration,
        "contact_bleeding": s_bleeding,
        "nodular_surface": s_nodular,
        "mass_reflection": s_mass,
    }

    # Composite severity = max signal (any ONE strong sign is enough to flag)
    severity = float(max(signals.values()))

    # Trigger if severity ≥ 0.55 OR any TWO signals ≥ 0.35
    n_moderate = sum(1 for v in signals.values() if v >= 0.35)
    is_advanced = (severity >= 0.55) or (n_moderate >= 2)

    reasons: list = []
    if s_ulceration >= 0.35:
        reasons.append(f"Large dark region ({s_ulceration*100:.0f}% severity) — "
                       f"could indicate deep ulceration or cavitation.")
    if s_bleeding >= 0.35:
        reasons.append(f"Heavy red-saturated patches ({s_bleeding*100:.0f}%) — "
                       f"suggests contact bleeding or fresh blood.")
    if s_nodular >= 0.35:
        reasons.append(f"Highly nodular/irregular surface texture "
                       f"({s_nodular*100:.0f}%) — possible mass effect.")
    if s_mass >= 0.35:
        reasons.append(f"Large reflective fungating-looking patches "
                       f"({s_mass*100:.0f}%) — manual review recommended.")

    return {
        "is_advanced": is_advanced,
        "severity":    severity,
        "signals":     signals,
        "reasons":     reasons,
    }
