"""ColonAI — Counterfactual Explainer.

Answer the question every clinician asks:
    "What would the model have said if X had been different?"

Two flavours of counterfactual, both work on the fused inference function:

1. **Modality-silencing counterfactuals**
   - Silence each modality in turn.
   - Report Δ predicted class, Δ confidence on the original class,
     and (if changed) the new prediction.
   - This is the most clinically useful answer: "if I gave you NO clinical
     text, you would have said …".

2. **Tabular perturbation counterfactuals**
   - For the 12 TCGA features, sweep each feature ± 1 SD and report
     which features push the prediction the most. Only safe perturbations
     (e.g. age ± 10y, BMI ± 5) — never invent biology.

3. **Image perturbation counterfactuals** (lightweight, no GAN)
   - Brightness ± 30%, hue rotation, mild blur, mirrored image.
   - Reports the model's stability under common acquisition variations.
   - This catches "the model only said cancer because the image was dark".

All three return the SAME schema so the UI can render them uniformly:

    {
        scenario:    str         "Silence image" / "Brighten 30%" / ...
        new_class:   str         what the model says under the scenario
        new_conf:    float
        delta_conf:  float       new - original
        flipped:     bool        prediction changed?
        risk_change: str         "▲ less confident" / "▼ more confident" / "= no change"
    }

Pure post-hoc — none of this changes the underlying model.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import math


# ──────────────────────────────────────────────────────────────────────
#  Modality-silencing counterfactuals
# ──────────────────────────────────────────────────────────────────────
def silence_modalities(
    *,
    predict_fn: Callable[..., Dict[str, float]],
    image: Optional[Any] = None,
    text: Optional[str] = None,
    tabular: Optional[Any] = None,
    silenced_image: Optional[Any] = None,
    silenced_tabular: Optional[Any] = None,
    original_class: str,
    original_confidence: float,
) -> List[Dict[str, Any]]:
    """Generate one counterfactual per modality (silencing it).

    Returns a list of scenario dicts, one per modality that was present.
    """
    scenarios: List[Dict[str, Any]] = []

    def _run(label: str, **kwargs):
        try:
            p = predict_fn(**kwargs)
            new_cls = max(p.items(), key=lambda kv: kv[1])[0] if p else original_class
            new_conf = float(p.get(new_cls, 0.0))
            delta = new_conf - float(original_confidence) if new_cls == original_class \
                    else -float(original_confidence)
            scenarios.append(_format_scenario(label, original_class, new_cls,
                                              new_conf, delta))
        except Exception as exc:
            scenarios.append({
                "scenario": label,
                "new_class": "(error)",
                "new_conf": 0.0,
                "delta_conf": 0.0,
                "flipped": False,
                "risk_change": "= unavailable",
                "error": str(exc),
            })

    if image is not None:
        _run("Silence image (replace with mid-grey)",
             image=silenced_image, text=text, tabular=tabular)
    if text is not None and str(text).strip():
        _run("Silence clinical text (use empty string)",
             image=image, text="", tabular=tabular)
    if tabular is not None:
        _run("Silence tabular features (use median patient)",
             image=image, text=text, tabular=silenced_tabular)
    return scenarios


# ──────────────────────────────────────────────────────────────────────
#  Tabular perturbation counterfactuals
# ──────────────────────────────────────────────────────────────────────
def perturb_tabular(
    *,
    predict_fn: Callable[..., Dict[str, float]],
    image: Optional[Any] = None,
    text: Optional[str] = None,
    tabular: Dict[str, float],
    original_class: str,
    original_confidence: float,
    perturbations: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Sweep each tabular feature ± a sensible amount.

    Parameters
    ----------
    tabular        Original feature dict (e.g. {"age": 62, "bmi": 28, …})
    perturbations  Per-feature (low, high) deltas. Default: clinically
                   sensible perturbations defined inline.
    """
    if perturbations is None:
        perturbations = {
            "age":              (-10, +10),
            "bmi":              (-5, +5),
            "pack_years":       (-10, +10),
            "cigs_per_day":     (-10, +10),
            "alcohol_history":  (-1, +1),
            "family_hx_cancer": (-1, +1),
            "gender_male":      (-1, +1),
            "site_rectum":      (-1, +1),
        }

    scenarios: List[Dict[str, Any]] = []
    for feat, (lo, hi) in perturbations.items():
        if feat not in tabular: continue
        orig = float(tabular[feat])
        for delta, tag in [(lo, "↓"), (hi, "↑")]:
            new_val = orig + delta
            # Clip binary features to {0,1}
            if feat in ("gender_male", "site_rectum", "alcohol_history",
                        "family_hx_cancer"):
                new_val = float(max(0.0, min(1.0, round(new_val))))
            if abs(new_val - orig) < 1e-6: continue
            new_tab = dict(tabular); new_tab[feat] = new_val
            try:
                p = predict_fn(image=image, text=text, tabular=new_tab)
                new_cls = max(p.items(), key=lambda kv: kv[1])[0] if p else original_class
                new_conf = float(p.get(new_cls, 0.0))
                delta_conf = new_conf - float(original_confidence) if new_cls == original_class \
                             else -float(original_confidence)
                label = f"{feat} {tag} ({orig:g} → {new_val:g})"
                scenarios.append(_format_scenario(label, original_class, new_cls,
                                                   new_conf, delta_conf))
            except Exception as exc:
                continue

    # Sort by |delta| descending so most-influential perturbations rise to top
    scenarios.sort(key=lambda s: -abs(s["delta_conf"]))
    return scenarios[:6]                # top 6


# ──────────────────────────────────────────────────────────────────────
#  Image acquisition counterfactuals (lightweight, no GAN)
# ──────────────────────────────────────────────────────────────────────
def perturb_image(
    *,
    predict_fn: Callable[..., Dict[str, float]],
    image,
    text: Optional[str] = None,
    tabular: Optional[Any] = None,
    original_class: str,
    original_confidence: float,
) -> List[Dict[str, Any]]:
    """Image stability under common acquisition variations.

    Tests: brightness ±30%, hue rotation 15°, mild blur, mirror.
    `image` may be a PIL Image, a numpy array, or a torch tensor —
    we handle the first two; if it's a tensor we try to convert it.
    """
    scenarios: List[Dict[str, Any]] = []
    img = _coerce_to_pil(image)
    if img is None:
        return []

    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError:
        return []

    cases: List[Tuple[str, Any]] = []
    # Brightness ± 30%
    try:
        cases.append(("Brightness +30%", ImageEnhance.Brightness(img).enhance(1.30)))
        cases.append(("Brightness −30%", ImageEnhance.Brightness(img).enhance(0.70)))
    except Exception: pass
    # Mirror
    try:
        cases.append(("Mirrored (L↔R)", ImageOps.mirror(img)))
    except Exception: pass
    # Blur
    try:
        cases.append(("Mild Gaussian blur (σ=2)",
                      img.filter(ImageFilter.GaussianBlur(radius=2))))
    except Exception: pass
    # Hue rotate (HSV)
    try:
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        h = h.point(lambda p: (p + 25) % 256)
        from PIL import Image as _PI
        cases.append(("Hue rotated +25°",
                      _PI.merge("HSV", (h, s, v)).convert("RGB")))
    except Exception: pass

    for label, perturbed in cases:
        try:
            p = predict_fn(image=perturbed, text=text, tabular=tabular)
            if not p: continue
            new_cls = max(p.items(), key=lambda kv: kv[1])[0]
            new_conf = float(p.get(new_cls, 0.0))
            delta_conf = new_conf - float(original_confidence) if new_cls == original_class \
                         else -float(original_confidence)
            scenarios.append(_format_scenario(label, original_class, new_cls,
                                               new_conf, delta_conf))
        except Exception:
            continue

    return scenarios


# ──────────────────────────────────────────────────────────────────────
#  Stability summary
# ──────────────────────────────────────────────────────────────────────
def stability_score(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of counterfactual scenarios into a stability index.

    Stability = fraction of scenarios where the prediction did NOT flip.
    Lower stability ⇒ model is fragile to perturbations.
    """
    if not scenarios:
        return {"stability": 1.0, "flips": 0, "total": 0, "verdict": "no perturbations tested"}
    n = len(scenarios)
    flips = sum(1 for s in scenarios if s.get("flipped", False))
    stability = (n - flips) / n
    if stability >= 0.95:
        verdict = "very stable"
    elif stability >= 0.80:
        verdict = "stable"
    elif stability >= 0.60:
        verdict = "moderately stable"
    elif stability >= 0.40:
        verdict = "fragile — interpret with caution"
    else:
        verdict = "highly unstable — model may not be reliable on this case"
    return {
        "stability":   stability,
        "flips":       flips,
        "total":       n,
        "verdict":     verdict,
        "max_delta":   max((abs(s["delta_conf"]) for s in scenarios), default=0.0),
    }


# ──────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────
def _format_scenario(label: str, orig_cls: str, new_cls: str,
                     new_conf: float, delta_conf: float) -> Dict[str, Any]:
    flipped = (new_cls != orig_cls)
    if flipped:
        arrow = "⇄ FLIPPED"
    elif delta_conf > 0.05:
        arrow = "▲ more confident"
    elif delta_conf < -0.05:
        arrow = "▼ less confident"
    else:
        arrow = "= no meaningful change"
    return {
        "scenario":     label,
        "new_class":    new_cls,
        "new_conf":     float(new_conf),
        "delta_conf":   float(delta_conf),
        "flipped":      bool(flipped),
        "risk_change":  arrow,
    }


def _coerce_to_pil(image) -> Optional[Any]:
    """Best-effort conversion of an image-like input to a PIL Image."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if hasattr(image, "size") and hasattr(image, "convert"):    # PIL.Image
        return image.convert("RGB")
    try:
        import numpy as np
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != "uint8":
                arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-9)).astype("uint8")
            return Image.fromarray(arr)
    except Exception:
        pass
    try:
        import torch
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu().float()
            if t.ndim == 4: t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-9)).astype("uint8")
            return Image.fromarray(arr)
    except Exception:
        pass
    return None
