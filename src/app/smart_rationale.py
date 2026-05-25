"""ColonAI — smart rationale generator.

Replaces templated "the AI is confident" filler with concrete,
per-image measurements derived from the actual model output.

For every prediction we compute:
   • Lesion size as % of the visible frame (from seg mask, fallback to gradcam)
   • Spatial location (3×3 octant grid)
   • Attention focus tightness (entropy of normalised gradcam over pixels)
   • Lesion-vs-background contrast (luminance ratio)
   • Edge regularity (mask circularity = 4πA/P², 1.0 = perfect circle)
   • Dominant colour family inside the lesion (plain English)
   • Confidence-shape (calibrated phrasing matching the actual confidence)

Output: a list of plain-English sentences + the raw measurement dict.

This module is pure-Python + numpy + cv2 — no model dependency,
so it's cheap to call on every prediction.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────
def _lesion_mask(gradcam: Optional[np.ndarray],
                 seg_mask: Optional[np.ndarray],
                 shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Pick the best available lesion mask. Prefers the seg decoder; falls
    back to the GradCAM top-25% threshold if seg isn't there."""
    H, W = shape
    if seg_mask is not None and seg_mask.size > 0:
        m = cv2.resize(seg_mask.astype(np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR)
        return (m > 0.5).astype(np.uint8)
    if gradcam is not None and gradcam.size > 0:
        g = cv2.resize(gradcam.astype(np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR)
        thr = float(np.quantile(g, 0.75))
        return (g >= thr).astype(np.uint8)
    return None


def _lesion_size_pct(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0: return 0.0
    return 100.0 * float(mask.sum()) / float(mask.size)


def _location_octant(mask: np.ndarray) -> str:
    """Return one of:  upper-left  upper-centre  upper-right
                       centre-left centre        centre-right
                       lower-left  lower-centre  lower-right"""
    if mask is None or mask.sum() < 4: return "across the frame"
    ys, xs = np.where(mask > 0)
    cy = ys.mean() / mask.shape[0]
    cx = xs.mean() / mask.shape[1]
    v = "upper"   if cy < 0.33 else ("lower"   if cy > 0.66 else "centre")
    h = "left"    if cx < 0.33 else ("right"   if cx > 0.66 else "centre")
    if v == "centre" and h == "centre": return "centre"
    if v == "centre": return f"middle-{h}"
    if h == "centre": return f"{v}-middle"
    return f"{v}-{h}"


def _attention_tightness(gradcam: Optional[np.ndarray]) -> float:
    """Normalised entropy of the GradCAM. 1.0 = uniform (diffuse), 0.0 = a single pixel.

    Returned as a 0-1 score where HIGHER = TIGHTER focus (what we want).
    """
    if gradcam is None or gradcam.size == 0: return 0.0
    g = gradcam.flatten().astype(np.float64)
    g = g - g.min()
    s = g.sum()
    if s < 1e-9: return 0.0
    p = g / s
    p = p[p > 0]
    entropy = -float((p * np.log(p)).sum())
    max_entropy = float(np.log(gradcam.size))
    if max_entropy < 1e-9: return 0.0
    normalised = entropy / max_entropy             # 0=peaked, 1=uniform
    return float(1.0 - normalised)                 # invert so 1 = tightest


def _contrast(image_rgb: np.ndarray, mask: np.ndarray) -> float:
    """|mean(luminance_in_mask) - mean(luminance_out)| / 255 → 0-1."""
    if mask is None or mask.sum() < 4: return 0.0
    if image_rgb is None or image_rgb.size == 0: return 0.0
    # Resize mask to image
    if mask.shape != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inside  = gray[mask > 0]
    outside = gray[mask == 0]
    if inside.size == 0 or outside.size == 0: return 0.0
    return float(abs(inside.mean() - outside.mean()) / 255.0)


def _circularity(mask: np.ndarray) -> float:
    """4πA / P²  →  1.0 = perfect circle, <0.5 = jagged/irregular."""
    if mask is None or mask.sum() < 16: return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if perim < 1e-6: return 0.0
    return float(min(1.0, 4 * np.pi * area / (perim * perim)))


def _dominant_color(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[str, Tuple[int,int,int]]:
    """Return ("plain-English colour name", (r,g,b) mean inside mask)."""
    if mask is None or mask.sum() < 4 or image_rgb is None: return ("uncertain hue", (128,128,128))
    if mask.shape != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    inside = image_rgb[mask > 0]
    if inside.size == 0: return ("uncertain hue", (128,128,128))
    r, g, b = inside[:,0].mean(), inside[:,1].mean(), inside[:,2].mean()
    rgb = (int(r), int(g), int(b))
    # Hue-based plain-English description
    hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0,0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s < 30:
        label = "pale grey" if v < 130 else "off-white"
    elif h < 10 or h > 165:
        label = "deep red" if v < 120 else ("salmon pink" if v > 170 else "red-pink")
    elif h < 20:
        label = "rust / dark orange"
    elif h < 30:
        label = "yellow-orange"
    elif h < 90:
        label = "yellow-green"
    else:
        label = "purple-grey"
    return (label, rgb)


def _confidence_phrase(p: float) -> str:
    if p >= 0.92: return "very confident"
    if p >= 0.80: return "confident"
    if p >= 0.70: return "moderately confident"
    if p >= 0.55: return "tentative"
    return "not very confident"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def smart_rationale(*,
    image_rgb:         np.ndarray,
    pathology_class:   str,
    confidence:        float,
    gradcam:           Optional[np.ndarray] = None,
    seg_mask:          Optional[np.ndarray] = None,
    uncertainty:       float = 0.0,
    extras:            Optional[Dict] = None,
) -> Dict:
    """Compute per-image rationale.

    Returns:
        {
          "bullets":   list[str]       human-readable observations
          "metrics":   dict            raw numeric measurements
          "summary":   str             one-line summary
        }
    """
    H, W = image_rgb.shape[:2]
    mask = _lesion_mask(gradcam, seg_mask, (H, W))
    size_pct  = _lesion_size_pct(mask) if mask is not None else 0.0
    location  = _location_octant(mask) if mask is not None else "unknown"
    tight     = _attention_tightness(gradcam)
    contrast  = _contrast(image_rgb, mask) if mask is not None else 0.0
    circ      = _circularity(mask) if mask is not None else 0.0
    colour, rgb = _dominant_color(image_rgb, mask) if mask is not None else ("?", (0,0,0))

    bullets: List[str] = []

    # 1. Confidence phrasing
    cp = _confidence_phrase(confidence)
    bullets.append(
        f"The AI is **{cp}** in its reading ({confidence*100:.0f}%), "
        f"with an internal uncertainty of {uncertainty:.2f}."
    )

    # 2. Lesion size + location (only if we have a mask)
    if mask is not None and size_pct > 0.1:
        if size_pct < 1.0:
            size_label = f"a small region (~{size_pct:.1f}% of the visible field)"
        elif size_pct < 5.0:
            size_label = f"a modest region (~{size_pct:.1f}% of the visible field)"
        elif size_pct < 15.0:
            size_label = f"a sizeable region (~{size_pct:.0f}% of the visible field)"
        else:
            size_label = f"a large area (~{size_pct:.0f}% of the visible field)"
        bullets.append(
            f"It is focused on **{size_label}**, located in the "
            f"**{location}** of the image."
        )

    # 3. Attention focus quality
    if gradcam is not None:
        if tight > 0.85:
            bullets.append(
                f"Its attention is **tightly focused** "
                f"(focus score {tight*100:.0f}%) — a single concentrated region, "
                f"which is what we want to see."
            )
        elif tight > 0.65:
            bullets.append(
                f"Its attention is **moderately focused** "
                f"(focus score {tight*100:.0f}%)."
            )
        else:
            bullets.append(
                f"Its attention is **spread out** "
                f"(focus score {tight*100:.0f}%) — be cautious, "
                f"the model may be looking at background detail."
            )

    # 4. Lesion-vs-background contrast
    if contrast > 0.18:
        bullets.append(
            f"The highlighted region stands out **clearly** from the surrounding "
            f"mucosa (brightness contrast {contrast*100:.0f}%)."
        )
    elif contrast > 0.08:
        bullets.append(
            f"The highlighted region has **modest** contrast against the "
            f"surrounding mucosa ({contrast*100:.0f}%)."
        )
    elif mask is not None and mask.sum() > 16:
        bullets.append(
            f"The highlighted region has **low** contrast against the "
            f"background ({contrast*100:.0f}%) — small or flat lesions can look like this."
        )

    # 5. Shape — only relevant for polyp-like classes
    if pathology_class == "polyps" and circ > 0:
        if circ > 0.75:
            bullets.append(
                f"The shape is **round and smooth** (circularity {circ:.2f}), "
                f"consistent with a sessile or pedunculated polyp."
            )
        elif circ > 0.45:
            bullets.append(
                f"The shape is **roughly oval** (circularity {circ:.2f})."
            )
        else:
            bullets.append(
                f"The shape is **irregular** (circularity {circ:.2f}) — "
                f"flat / serrated lesions or scope-edge artefacts can look like this."
            )

    # 6. Dominant colour
    if mask is not None and mask.sum() > 16:
        bullets.append(
            f"The dominant colour inside the region is **{colour}** "
            f"(RGB ≈ {rgb})."
        )

    # One-line summary
    summary = (
        f"AI says **{pathology_class}** ({cp}, {confidence*100:.0f}% confidence), "
        f"focused on the {location} of the image, "
        f"covering ~{size_pct:.0f}% of the frame."
    )

    metrics = {
        "lesion_size_pct":      size_pct,
        "location":             location,
        "attention_tightness":  tight,
        "contrast_score":       contrast,
        "circularity":          circ,
        "dominant_colour":      colour,
        "dominant_rgb":         list(rgb),
        "confidence_phrase":    cp,
        "n_bullets":            len(bullets),
    }
    return {"bullets": bullets, "metrics": metrics, "summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic colonoscopy frame with a "polyp" in the upper-left
    H = W = 224
    img = np.full((H, W, 3), [160, 95, 90], dtype=np.uint8)        # pink mucosa
    yy, xx = np.mgrid[0:H, 0:W]
    disk = ((yy - 60)**2 + (xx - 70)**2) < 30**2
    img[disk] = [200, 90, 80]                                       # darker polyp
    gradcam = disk.astype(np.float32) + 0.05*np.random.rand(H, W)
    seg = disk.astype(np.float32) * 0.9
    r = smart_rationale(image_rgb=img, pathology_class="polyps",
                        confidence=0.87, gradcam=gradcam, seg_mask=seg,
                        uncertainty=0.14)
    print("─ summary ─")
    print(" ", r["summary"])
    print("\n─ bullets ─")
    for b in r["bullets"]: print(f"  • {b}")
    print("\n─ metrics ─")
    for k, v in r["metrics"].items(): print(f"  {k:24s} = {v}")
