"""ColonAI — Modality Attribution.

Quantify HOW MUCH each input modality (image / text / tabular) contributed
to the model's final prediction. This is the per-modality analogue of
GradCAM / Integrated-Gradients: instead of asking *which pixels* mattered,
it asks *which modality* mattered.

Two estimators, both work on the fused multi-modal model:

1. **Silencing attribution** (model-free, always works)
   - Run the model with each modality silenced in turn (image → grey;
     text → blank; tabular → median of training set).
   - Δ confidence on the predicted class ≈ that modality's contribution.
   - Normalise to sum to 100%.

2. **Gradient-based attribution** (requires Captum or autograd)
   - Compute ∂log p(ŷ) / ∂x for each modality's input tensor.
   - L2-norm of gradient × input ≈ saliency per modality.
   - Normalise to sum to 100%.

The UI shows a stacked bar:   Image 62% ▮ Text 11% ▮ Tabular 27%
…so the user can see at a glance what the model was leaning on.

This module is **defensive**: if any modality is missing, or the model
doesn't expose the expected forward-pass kwargs, we fall back to a
silencing-only computation, and if THAT fails we return an honest
"attribution unavailable" report rather than crash.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import math


# ──────────────────────────────────────────────────────────────────────
#  Silencing attribution
# ──────────────────────────────────────────────────────────────────────
def silencing_attribution(
    *,
    predict_fn: Callable[..., Dict[str, float]],
    image: Optional[Any] = None,
    text: Optional[str] = None,
    tabular: Optional[Any] = None,
    silenced_image: Optional[Any] = None,
    silenced_text: str = "",
    silenced_tabular: Optional[Any] = None,
    predicted_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Attribution by ablation: remove each modality, observe Δ prob.

    Parameters
    ----------
    predict_fn  Callable returning a {class_name: prob} dict.
                Must accept kwargs (image=..., text=..., tabular=...).
    image, text, tabular
                Inputs to the model. Pass None for modalities not used.
    silenced_*  What to substitute when silencing. Defaults:
                image → mid-grey tensor of same shape (caller-provided)
                text → empty string
                tabular → zeros (caller-provided)
    predicted_class
                Class name to track. If None, uses argmax of full prediction.

    Returns
    -------
    dict with:
        contributions   {modality: pct} summing to 100
        deltas          {modality: float}    Δp when silenced
        baseline_prob   float                 prob with no silencing
        method          'silencing'
    """
    # 1. Baseline
    try:
        baseline = predict_fn(image=image, text=text, tabular=tabular)
    except Exception as exc:
        return _failure(f"baseline predict failed: {exc}")
    if not baseline:
        return _failure("baseline predict returned no probabilities")

    if predicted_class is None:
        predicted_class = max(baseline.items(), key=lambda kv: kv[1])[0]
    baseline_prob = float(baseline.get(predicted_class, 0.0))

    deltas: Dict[str, float] = {}

    # 2. Silence each modality in turn
    if image is not None:
        try:
            p = predict_fn(image=silenced_image, text=text, tabular=tabular)
            deltas["image"] = max(0.0, baseline_prob - float(p.get(predicted_class, 0.0)))
        except Exception:
            pass

    if text is not None and text.strip():
        try:
            p = predict_fn(image=image, text=silenced_text, tabular=tabular)
            deltas["text"] = max(0.0, baseline_prob - float(p.get(predicted_class, 0.0)))
        except Exception:
            pass

    if tabular is not None:
        try:
            p = predict_fn(image=image, text=text, tabular=silenced_tabular)
            deltas["tabular"] = max(0.0, baseline_prob - float(p.get(predicted_class, 0.0)))
        except Exception:
            pass

    if not deltas:
        return _failure("no modality could be silenced")

    # 3. Normalise to 100%
    total = sum(deltas.values())
    if total < 1e-6:
        # Model is so over-confident that ablating any modality barely moves it
        # → assume even split.
        n = len(deltas)
        contributions = {k: 100.0 / n for k in deltas}
    else:
        contributions = {k: float(v / total * 100.0) for k, v in deltas.items()}

    return {
        "contributions":  contributions,
        "deltas":         deltas,
        "baseline_prob":  baseline_prob,
        "predicted_class": predicted_class,
        "method":         "silencing",
        "interpretable":  True,
    }


# ──────────────────────────────────────────────────────────────────────
#  Gradient-based attribution
# ──────────────────────────────────────────────────────────────────────
def gradient_attribution(
    *,
    model: Any,
    image_tensor: Optional[Any] = None,
    text_ids: Optional[Any] = None,
    text_attention_mask: Optional[Any] = None,
    tabular_tensor: Optional[Any] = None,
    target_class: int,
    head: str = "pathology",
) -> Dict[str, Any]:
    """Per-modality saliency via gradient × input on the unified transformer.

    Assumes the model is the UnifiedMultiModalTransformer:
       forward(images, text_input_ids, text_attention_mask, tabular) → dict[head] = logits

    Returns the L2-norm of (∂logit / ∂x · x) per modality, normalised
    to percentages.
    """
    try:
        import torch
    except ImportError:
        return _failure("torch not available")

    model.eval()
    use_grad: List[Any] = []
    img_g = txt_g = tab_g = None

    def _requires_grad(t):
        if t is None: return None
        t = t.detach().clone()
        if t.is_floating_point():
            t.requires_grad_(True)
            use_grad.append(t)
        return t

    img = _requires_grad(image_tensor)
    tab = _requires_grad(tabular_tensor)
    # text ids are integer indices, not differentiable directly; we skip them
    # for gradient attribution and treat text contribution via silencing
    # in the fused estimator below.

    try:
        with torch.enable_grad():
            out = model(
                images=img,
                text_input_ids=text_ids,
                text_attention_mask=text_attention_mask,
                tabular=tab,
            )
            logits = out[head] if isinstance(out, dict) else out
            target = logits[0, target_class]
            target.backward()
    except Exception as exc:
        return _failure(f"backward failed: {exc}")

    if img is not None and img.grad is not None:
        img_g = float((img.grad * img).pow(2).mean().sqrt().item())
    if tab is not None and tab.grad is not None:
        tab_g = float((tab.grad * tab).pow(2).mean().sqrt().item())

    deltas: Dict[str, float] = {}
    if img_g is not None: deltas["image"] = img_g
    if tab_g is not None: deltas["tabular"] = tab_g

    if not deltas:
        return _failure("no differentiable modalities found")

    total = sum(deltas.values())
    contributions = {k: float(v / total * 100.0) for k, v in deltas.items()} if total > 0 else \
                    {k: 100.0 / len(deltas) for k in deltas}

    return {
        "contributions":  contributions,
        "deltas":         deltas,
        "method":         "gradient",
        "interpretable":  True,
        "note":           "text excluded (token IDs not differentiable); use silencing for text",
    }


# ──────────────────────────────────────────────────────────────────────
#  Fused estimator (silencing + heuristic confidence weighting)
# ──────────────────────────────────────────────────────────────────────
def fused_attribution(
    *,
    image_confidence: Optional[float] = None,
    text_confidence: Optional[float] = None,
    tabular_confidence: Optional[float] = None,
    silencing_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """When silencing is unavailable (or unreliable), fall back to a
    confidence-weighted heuristic: a modality whose own classifier was
    confident probably contributed more to the fused prediction.

    Either way, normalise to 100%.
    """
    # Prefer silencing if it gave sane numbers
    if silencing_result and silencing_result.get("interpretable"):
        contribs = silencing_result.get("contributions", {})
        if all(0.0 <= v <= 100.0 for v in contribs.values()) and len(contribs) >= 2:
            return {**silencing_result, "method": silencing_result.get("method") + "+fused"}

    deltas: Dict[str, float] = {}
    if image_confidence is not None:    deltas["image"] = float(image_confidence)
    if text_confidence is not None:     deltas["text"] = float(text_confidence)
    if tabular_confidence is not None:  deltas["tabular"] = float(tabular_confidence)

    if not deltas:
        return _failure("no confidence values to fuse")

    total = sum(deltas.values())
    contributions = {k: float(v / total * 100.0) for k, v in deltas.items()} if total > 0 else \
                    {k: 100.0 / len(deltas) for k in deltas}
    return {
        "contributions":  contributions,
        "deltas":         deltas,
        "method":         "confidence-weighted",
        "interpretable":  True,
        "note":           "approximation: assumes higher unimodal confidence → bigger fused contribution",
    }


# ──────────────────────────────────────────────────────────────────────
#  Rendering helpers (for the UI)
# ──────────────────────────────────────────────────────────────────────
_BAR_BLOCKS = "▁▂▃▄▅▆▇█"

def render_text_bar(contributions: Dict[str, float], width: int = 32) -> str:
    """Render a Unicode horizontal bar for the contributions dict."""
    out_chars: List[str] = []
    items = sorted(contributions.items(), key=lambda kv: -kv[1])
    for name, pct in items:
        cells = max(1, int(round((pct / 100.0) * width)))
        out_chars.append(f"{name:>8} | {'█' * cells} {pct:5.1f}%")
    return "\n".join(out_chars)


def attribution_summary(result: Dict[str, Any]) -> str:
    """One-sentence summary of the attribution for the rationale card."""
    if not result.get("interpretable"):
        return "Per-modality attribution unavailable for this case."
    contribs = result.get("contributions", {})
    if not contribs:
        return "Per-modality attribution unavailable for this case."
    # Find the dominant modality
    top = max(contribs.items(), key=lambda kv: kv[1])
    method = result.get("method", "?")
    label = {
        "image": "the endoscopic image",
        "text": "the clinical text",
        "tabular": "the TCGA tabular features",
    }.get(top[0], top[0])
    return (f"{label} drove ~{top[1]:.0f}% of the decision "
            f"(estimated by {method}; remaining contributions: " +
            ", ".join(f"{k} {v:.0f}%" for k, v in contribs.items() if k != top[0]) + ").")


# ──────────────────────────────────────────────────────────────────────
def _failure(reason: str) -> Dict[str, Any]:
    return {
        "contributions": {},
        "deltas": {},
        "method": "none",
        "interpretable": False,
        "reason": reason,
    }
