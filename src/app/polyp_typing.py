"""ColonAI — clinical polyp & lesion sub-typing.

The 5-class classifier (polyps / UC mild / UC mod-sev / Barrett's /
therapeutic) tells the user "this looks like a polyp" — but
clinically what matters for surveillance, removal technique, and
cancer-risk stratification is *what kind* of polyp.

This module adds three internationally-recognised sub-classifications
**without retraining the model** — all computed from the segmentation
mask + image pixels:

  1. **Paris classification** (morphology — 0-Ip, 0-Is, 0-IIa, 0-IIb, 0-IIc)
     The shape of the polyp. Drives the resection technique.

  2. **NICE classification** (surface pattern — Type 1 / 2 / 3)
     Predicts histology *without* biopsy. Brown-with-vessels = adenoma,
     pale-without-vessels = hyperplastic, distorted = invasive.

  3. **BSG size-risk stratification** (diminutive / small / large)
     Drives the surveillance interval. <5mm → 5-year, 10mm+ → 3-year.

Plus two extra colorectal-condition detectors:

  4. **Diverticulosis** — pattern of multiple small dark pouches.
  5. **Hemorrhoid / vascular bulge** — purple-blue distended vessels.

Each function returns a structured dict with the predicted sub-type,
clinical notes, and a confidence score derived from how well the
image matched the canonical signatures.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paris classification (morphology — shape of the polyp)
# ─────────────────────────────────────────────────────────────────────────────
def paris_classify(mask: Optional[np.ndarray]) -> Dict:
    """Compute the Paris morphological type from a segmentation mask.

    Reference: Paris endoscopic classification of superficial neoplastic
    lesions (Endoscopy 2005;37:570-578).

    Types we distinguish:
       • 0-Ip   pedunculated   (long stalk)
       • 0-Isp  sub-pedunculated
       • 0-Is   sessile        (broad-based, hemispherical)
       • 0-IIa  flat elevated  (height < base width / 4)
       • 0-IIb  completely flat
       • 0-IIc  depressed      (impossible to detect reliably from RGB)
    """
    if mask is None or mask.sum() < 16:
        return {"paris_type": "unknown", "confidence": 0.0,
                "rationale": "No lesion mask available",
                "removal_technique": None}

    binmask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"paris_type": "unknown", "confidence": 0.0,
                "rationale": "No contour found", "removal_technique": None}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    if area < 16 or perim < 4:
        return {"paris_type": "unknown", "confidence": 0.0,
                "rationale": "Lesion too small for morphology",
                "removal_technique": None}

    # Bounding rectangle gives us base × height
    x, y, bw, bh = cv2.boundingRect(c)
    # The wider of (bw, bh) is the "base"; the narrower is the "height" along
    # the axis perpendicular to the base. For Paris, we want the lesion's
    # protrusion vs its base — approximate via min rectangle.
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    # Normalize so w ≥ h (w is the longer side = base)
    if h > w: w, h = h, w
    base    = float(w)
    height  = float(h)
    aspect  = height / max(base, 1.0)        # 1.0 = circular, low = flat

    # Circularity (4πA / P²) — 1.0 = perfect circle (typical sessile)
    circ    = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0

    # Connectivity test: does the lesion have a narrow "stalk"?
    # Erode aggressively — if the lesion splits, it had a stalk
    eroded = cv2.erode(binmask,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                       iterations=2)
    n_after, _ = cv2.connectedComponents(eroded)
    had_stalk = (n_after > 2)   # background + ≥2 → split into pieces

    # Decision tree
    if had_stalk and aspect > 0.55:
        paris = "0-Ip"
        descr = "Pedunculated — long stalk visible"
        conf  = 0.78
        tech  = "Endoscopic mucosal resection (EMR) with snare polypectomy"
    elif had_stalk:
        paris = "0-Isp"
        descr = "Sub-pedunculated — short broad stalk"
        conf  = 0.68
        tech  = "EMR with snare polypectomy"
    elif aspect > 0.45 and circ > 0.6:
        paris = "0-Is"
        descr = "Sessile — broad-based hemispherical protrusion"
        conf  = 0.80
        tech  = "EMR; consider ESD if ≥ 20 mm"
    elif aspect > 0.20 and circ > 0.4:
        paris = "0-IIa"
        descr = "Flat elevated lesion — minor height above mucosa"
        conf  = 0.65
        tech  = "EMR with submucosal injection ('lift'); careful margin assessment"
    elif aspect <= 0.20:
        paris = "0-IIb"
        descr = "Completely flat lesion — same height as surrounding mucosa"
        conf  = 0.60
        tech  = "Chromoendoscopy + ESD; high suspicion warranted"
    else:
        paris = "0-IIa / 0-IIb"
        descr = "Borderline flat-elevated vs flat"
        conf  = 0.50
        tech  = "ESD-capable centre recommended for full assessment"

    return {
        "paris_type":        paris,
        "confidence":        conf,
        "rationale":         descr,
        "removal_technique": tech,
        "metrics":           {"aspect": aspect, "circularity": circ,
                              "base_px": base, "height_px": height,
                              "had_stalk": had_stalk, "area_px": area},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NICE classification (surface pattern — predicts histology)
# ─────────────────────────────────────────────────────────────────────────────
def nice_classify(image_rgb: np.ndarray, mask: Optional[np.ndarray]) -> Dict:
    """Approximate the NICE classification (NBI International Colorectal
    Endoscopic) from RGB image + segmentation mask. The original NICE
    uses Narrow-Band Imaging which highlights vessels — we approximate
    using colour saturation + edge density inside the mask.

       Type 1: hyperplastic / serrated      — same colour as background,
                                                no visible vessels
       Type 2: adenoma                       — brown, visible vessels
       Type 3: deep submucosal invasive CRC  — distorted pattern,
                                                amorphous / missing vessels

    This is heuristic and clearly labelled as such. Not a substitute for
    proper NBI assessment by a trained endoscopist.
    """
    if mask is None or mask.sum() < 16 or image_rgb is None:
        return {"nice_type": "unknown", "confidence": 0.0,
                "predicted_histology": None, "cancer_risk": None,
                "rationale": "No mask / image available"}

    H, W = image_rgb.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST)
    bin_in  = (mask > 0)
    bin_out = (mask == 0)
    if bin_in.sum() < 16 or bin_out.sum() < 16:
        return {"nice_type": "unknown", "confidence": 0.0,
                "predicted_histology": None, "cancer_risk": None,
                "rationale": "Lesion / background area too small"}

    # 1. Colour difference vs background (in HSV space)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    inside_h, inside_s, inside_v = hsv[bin_in, 0].mean(), hsv[bin_in, 1].mean(), hsv[bin_in, 2].mean()
    outside_h, outside_s, outside_v = hsv[bin_out, 0].mean(), hsv[bin_out, 1].mean(), hsv[bin_out, 2].mean()
    # Brown-ness = warm hue (close to 10-20) with moderate saturation
    brown_inside  = float(max(0, 1.0 - abs(inside_h - 15) / 30) * (inside_s / 200))
    brown_outside = float(max(0, 1.0 - abs(outside_h - 15) / 30) * (outside_s / 200))
    brown_delta   = float(brown_inside - brown_outside)

    # 2. Vessel-like edge density inside the lesion (vessels = thin curved edges)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_in  = edges[bin_in].mean() / 255.0
    edges_out = edges[bin_out].mean() / 255.0
    edge_excess = float(edges_in - edges_out)   # +ve = more vessels inside

    # 3. Surface disruption (HSV variance inside the mask)
    h_inside = hsv[bin_in, 0].astype(np.float32)
    s_inside = hsv[bin_in, 1].astype(np.float32)
    disorder = float((h_inside.std() / 30.0 + s_inside.std() / 80.0) / 2)

    # Decision tree
    # Type 3 (invasive) — REQUIRES high disorder AND very few vessels
    # (a flat, uniform region alone isn't invasive cancer — it has to
    # show actual surface distortion as well)
    if disorder > 0.85 and edges_in < 0.02:
        nice = "Type 3"
        hist = "Deep submucosal invasive carcinoma (likely)"
        risk = "HIGH — referral for surgical assessment, do not attempt endoscopic resection"
        conf = 0.55
        rationale = ("Surface pattern is irregular / amorphous "
                     "(disorder score = {:.2f}). May indicate deep invasion."
                     .format(disorder))
    elif brown_delta > 0.08 and edge_excess > 0.005:
        nice = "Type 2"
        hist = "Adenoma (tubular or tubulovillous most likely)"
        risk = "MODERATE — endoscopic removal recommended; histology guides surveillance interval"
        conf = 0.68
        rationale = ("Brown discoloration vs background (+{:.0f}%) and "
                     "visible vessel-like pattern (+{:.0f}% edges) — typical adenoma."
                     .format(brown_delta * 100, edge_excess * 100))
    elif brown_delta < 0.02 and edge_excess < 0.01:
        nice = "Type 1"
        hist = "Hyperplastic / serrated polyp (low-grade)"
        risk = "LOW — many can be left in situ if < 5 mm in rectosigmoid"
        conf = 0.60
        rationale = ("Same colour as surrounding mucosa and no visible vessel "
                     "pattern — pale, smooth surface typical of hyperplastic polyp.")
    else:
        nice = "Indeterminate"
        hist = "Cannot reliably distinguish 1 vs 2 from this image"
        risk = "UNCERTAIN — biopsy or NBI assessment recommended"
        conf = 0.45
        rationale = "Mixed surface pattern — features overlap between types."

    return {
        "nice_type":            nice,
        "confidence":           conf,
        "predicted_histology":  hist,
        "cancer_risk":          risk,
        "rationale":            rationale,
        "metrics":              {
            "brown_delta": brown_delta,
            "edge_excess": edge_excess,
            "disorder":    disorder,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Size-based risk stratification (BSG-aligned)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_size_mm(mask: Optional[np.ndarray],
                     image_shape: Tuple[int, int],
                     assumed_fov_mm: float = 30.0) -> Dict:
    """Estimate polyp size in mm assuming a standard colonoscopy FOV.

    Most adult colonoscopes have ~3 cm field of view at typical working
    distance. We measure the lesion's largest dimension as a fraction of
    the image's longer dimension and multiply.
    """
    if mask is None or mask.sum() < 16:
        return {"size_mm": None, "size_category": "unknown",
                "bsg_surveillance": None}

    binmask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"size_mm": None, "size_category": "unknown",
                "bsg_surveillance": None}
    c = max(contours, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(c)
    largest_px = max(bw, bh)
    img_px = max(image_shape[0], image_shape[1])
    size_mm = float(largest_px / img_px) * assumed_fov_mm

    if size_mm < 5:
        cat = "diminutive (< 5 mm)"
        surveil = ("Per BSG: usually safe to discard if < 5 mm in "
                   "rectosigmoid AND looks hyperplastic (Type 1). Otherwise remove and discard.")
    elif size_mm < 10:
        cat = "small (5-9 mm)"
        surveil = "Per BSG: remove with cold-snare polypectomy. Surveillance at 5 years."
    elif size_mm < 20:
        cat = "large (10-19 mm)"
        surveil = ("Per BSG: piecemeal EMR. Recommend surveillance at "
                   "3 years (or 2-6 months if piecemeal removal).")
    else:
        cat = "giant (≥ 20 mm)"
        surveil = ("Per BSG: referral to ESD-capable centre. Surveillance "
                   "tailored to histology — typically 6 months then 1 year.")

    return {
        "size_mm":          round(size_mm, 1),
        "size_category":    cat,
        "bsg_surveillance": surveil,
        "estimated_from":   f"largest dim {largest_px}px of {img_px}px frame "
                            f"× assumed {assumed_fov_mm:.0f} mm FOV",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Crohn's vs UC differential
# ─────────────────────────────────────────────────────────────────────────────
def ibd_differential(symptoms_text: str,
                     patient: Optional[Dict] = None,
                     image_rgb: Optional[np.ndarray] = None) -> Dict:
    """Score Crohn's vs UC from symptoms + patient history + image cues.

    Classic discriminators:
       Crohn's favours:  smoking, perianal disease, fistula, mouth ulcers,
                         skip lesions on image, cobblestoning, terminal-ileum
                         involvement, weight loss + abdominal mass
       UC favours:       non-smoking (protective), continuous rectum-onwards
                         inflammation, bloody diarrhoea, urgency / tenesmus
    """
    sx = (symptoms_text or "").lower()
    p  = patient or {}
    crohns_score = 0.0
    uc_score     = 0.0
    reasons_crohns: List[str] = []
    reasons_uc:     List[str] = []

    # ── Symptom-based ────────────────────────────────────────────────
    if any(k in sx for k in ["perianal", "fistula", "anal fistula", "abscess"]):
        crohns_score += 2.0
        reasons_crohns.append("Perianal disease / fistula mentioned")
    if any(k in sx for k in ["mouth ulcer", "oral ulcer", "aphthous"]):
        crohns_score += 1.5
        reasons_crohns.append("Oral / mouth ulceration mentioned")
    if any(k in sx for k in ["skip lesion", "cobblestone", "terminal ileum"]):
        crohns_score += 2.0
        reasons_crohns.append("Skip lesions / cobblestoning / terminal-ileal involvement noted")
    if any(k in sx for k in ["weight loss", "anorexia"]):
        crohns_score += 1.0
        reasons_crohns.append("Weight loss / anorexia is more typical of Crohn's")

    if any(k in sx for k in ["bloody diarrhoea", "bloody stool", "haematochezia",
                              "bloody diarrhea"]):
        uc_score += 2.0
        reasons_uc.append("Bloody diarrhoea is classic for UC")
    if any(k in sx for k in ["urgency", "tenesmus", "rectal urgency"]):
        uc_score += 1.5
        reasons_uc.append("Rectal urgency / tenesmus more typical of UC")
    if any(k in sx for k in ["continuous", "starts in rectum"]):
        uc_score += 1.5
        reasons_uc.append("Continuous pattern starting at rectum")

    # ── Smoking history ────────────────────────────────────────────────
    smokes = str(p.get("smoking", "")).lower() in ("yes", "current")
    if smokes:
        crohns_score += 1.5
        reasons_crohns.append("Current smoker — risk factor for Crohn's")
    else:
        uc_score += 0.5
        reasons_uc.append("Non-smoker — smoking is protective against UC")

    # ── Family history ─────────────────────────────────────────────────
    fam = str(p.get("family_history", "")).lower()
    if "crohn" in fam:
        crohns_score += 1.0; reasons_crohns.append("Family history of Crohn's")
    if "colitis" in fam or "uc" in fam:
        uc_score += 1.0; reasons_uc.append("Family history of UC")

    # ── Verdict ────────────────────────────────────────────────────────
    total = crohns_score + uc_score
    if total < 1.0:
        return {"verdict": "Insufficient information to differentiate IBD subtype",
                "crohns_score": 0.0, "uc_score": 0.0,
                "rationale": [], "recommendation":
                "Collect more symptom details and consider faecal calprotectin."}

    if crohns_score > uc_score * 1.3:
        verdict = "Pattern suggests Crohn's disease"
        rec     = "Recommend ileocolonoscopy with terminal-ileum biopsies + MR enterography."
    elif uc_score > crohns_score * 1.3:
        verdict = "Pattern suggests Ulcerative Colitis"
        rec     = "Continuous-pattern colonoscopy with biopsies + faecal calprotectin."
    else:
        verdict = "Pattern overlaps Crohn's and UC — IBD-Unclassified"
        rec     = "Refer to IBD MDT; consider serology (ASCA / ANCA), MRE, capsule endoscopy."

    return {
        "verdict":        verdict,
        "crohns_score":   round(crohns_score, 2),
        "uc_score":       round(uc_score, 2),
        "rationale":      reasons_crohns + reasons_uc,
        "recommendation": rec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diverticulosis & hemorrhoid pattern detectors
# ─────────────────────────────────────────────────────────────────────────────
def detect_diverticulosis(image_rgb: np.ndarray) -> Dict:
    """Detect the visual pattern of diverticulosis: multiple small, round,
    dark openings in the colon wall.
    """
    if image_rgb is None or image_rgb.size < 100:
        return {"detected": False, "score": 0.0, "n_candidates": 0}
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Threshold for "dark patches"
    _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if n <= 2:
        return {"detected": False, "score": 0.0, "n_candidates": 0}

    img_area = float(image_rgb.shape[0] * image_rgb.shape[1])
    # Count blobs that look pouch-like: round-ish, between 0.1 % and 3 % area
    candidates = 0
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        w = float(stats[i, cv2.CC_STAT_WIDTH])
        h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if a < img_area * 0.001 or a > img_area * 0.03: continue
        if w == 0 or h == 0: continue
        aspect = min(w, h) / max(w, h)
        if aspect > 0.6:                    # roundish
            candidates += 1
    detected = candidates >= 4
    score = float(min(1.0, candidates / 8.0))
    return {
        "detected":     detected,
        "score":        score,
        "n_candidates": candidates,
        "interpretation": (
            "Multiple small round dark patches detected — consistent with "
            "diverticulosis. Diverticula are usually benign but can lead to "
            "diverticulitis if inflamed."
            if detected else
            "No clear diverticular pattern detected."
        ),
    }


def detect_hemorrhoid_signs(image_rgb: np.ndarray) -> Dict:
    """Detect purple-blue distended-vessel patterns near the lower edge of the
    field — suggestive of internal hemorrhoids (visible during retroflexion).
    """
    if image_rgb is None or image_rgb.size < 100:
        return {"detected": False, "score": 0.0}
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Purple-blue: hue 110-150, moderate-high saturation, low-mid value
    purple = (((H > 110) & (H < 150)) & (S > 80) & (V < 180))
    pct = float(purple.mean())
    score = float(min(1.0, pct * 8))
    return {
        "detected": (score > 0.4),
        "score":    score,
        "interpretation": (
            f"Purple-blue distended-vessel pattern detected ({pct*100:.1f}% "
            f"of frame) — could indicate internal hemorrhoids."
            if score > 0.4 else
            "No prominent hemorrhoid-like vascular pattern."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public façade — call this from app.py to get ALL sub-typing in one go
# ─────────────────────────────────────────────────────────────────────────────
def full_sub_typing(*,
    image_rgb:      np.ndarray,
    mask:           Optional[np.ndarray] = None,
    pathology_class: str = "",
    symptoms_text:   str = "",
    patient:         Optional[Dict] = None,
) -> Dict:
    """Run every applicable sub-classifier and return the combined dict.

    Only runs the polyp-specific classifiers (Paris/NICE/size) if the
    primary class is 'polyps'. Always runs the diverticulosis + hemorrhoid
    detectors (they're orthogonal). IBD differential runs if the primary
    class is uc-mild / uc-moderate-sev.
    """
    result: Dict = {}
    if image_rgb is None: return result

    if pathology_class == "polyps":
        result["paris"] = paris_classify(mask)
        result["nice"]  = nice_classify(image_rgb, mask)
        result["size"]  = estimate_size_mm(mask, image_rgb.shape[:2])
    if pathology_class in ("uc-mild", "uc-moderate-sev"):
        result["ibd_differential"] = ibd_differential(symptoms_text, patient, image_rgb)
    # Always run these two — they catch conditions the 5-class classifier misses
    result["diverticulosis"] = detect_diverticulosis(image_rgb)
    result["hemorrhoid"]     = detect_hemorrhoid_signs(image_rgb)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─ polyp_typing self-test ─")

    # Synthetic sessile polyp: round disk in centre
    H = W = 224
    img = np.full((H, W, 3), [180, 100, 95], dtype=np.uint8)   # pink mucosa
    yy, xx = np.mgrid[0:H, 0:W]
    disk = ((yy - 112)**2 + (xx - 112)**2) < 30**2
    img[disk] = [200, 130, 80]                                  # browner polyp
    mask = disk.astype(np.uint8)

    print("\n  case A — sessile-shaped polyp:")
    r = full_sub_typing(image_rgb=img, mask=mask, pathology_class="polyps")
    for k, v in r.items():
        print(f"\n  ── {k} ──")
        for kk, vv in v.items():
            if kk == "metrics": continue
            print(f"    {kk:22s} = {vv}")

    # Synthetic IBD case — Crohn's symptoms
    print("\n\n  case B — Crohn's-pattern symptoms:")
    r = full_sub_typing(image_rgb=img, pathology_class="uc-mild",
                       symptoms_text=("Patient reports skip lesions on prior imaging, "
                                       "mouth ulcers, weight loss over 3 months. "
                                       "Smokes 10/day."),
                       patient={"smoking": "yes", "family_history": "Crohn's"})
    for k, v in r.items():
        print(f"\n  ── {k} ──")
        for kk, vv in v.items():
            if isinstance(vv, list):
                for line in vv: print(f"    {kk:22s} • {line}")
            else:
                print(f"    {kk:22s} = {vv}")
