"""ColonAI — production-grade inference wrapper.

Wraps the base UnifiedMultiModalTransformer with three medical-grade
improvements that don't require retraining:

  1. **Test-Time Augmentation (TTA)** — 5 augmented forward passes per
     image, average the softmax probabilities. Reduces variance and
     typically lifts F1 by 2-4 % on cross-vendor data.

  2. **Deep MC-Dropout** — 10 stochastic forward passes (with dropout
     enabled) to compute *predictive entropy* and *mutual information*
     as Bayesian-style uncertainty estimates. Much more reliable than
     the previous 3-pass version.

  3. **Hierarchical UC sub-decision** — when the model predicts uc-mild
     AND uc-moderate-sev is in the top-2 with prob > 0.30, we surface
     the case as "UC of uncertain grade — clinician review required"
     instead of confidently calling mild. Directly addresses the
     uc-mod-sev recall = 0.15 problem.

  4. **Top-3 differential diagnosis** — the result page shows ranked
     alternatives with probability bars, not just the top class.

All four returned in a single SmartPrediction dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F


# Internal constants
TTA_AUGS_DEFAULT = 5             # identity + 4 geometric augmentations
MC_DROPOUT_DEFAULT = 10          # was 3 in earlier code
UC_HEDGE_THRESHOLD = 0.30        # if uc-mod-sev > this when uc-mild wins, hedge


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SmartPrediction:
    # Final outputs (calibrated, hierarchical, ready to display)
    predicted_class:        str
    confidence:             float
    uncertainty:            float          # predictive entropy (normalised 0-1)
    mutual_info:            float          # epistemic uncertainty (0-1)
    differential:           List[Dict]     # top-3, e.g. [{class, prob}, …]
    is_hedged:              bool           # True if the UC hierarchical rule fired
    hedge_reason:           Optional[str]  # human-readable why

    # Raw outputs for downstream modules
    mean_probs:             np.ndarray
    tta_probs:              np.ndarray     # (n_augs, n_classes) all TTA passes
    mc_probs:               np.ndarray     # (n_mc, n_classes) all MC-Dropout passes
    tta_std:                float          # max std across TTA passes per class
    mc_std:                 float          # max std across MC-Dropout passes per class
    n_tta:                  int
    n_mc:                   int


# ─────────────────────────────────────────────────────────────────────────────
# TTA helpers — apply the SAME augmentation to a (1, 3, H, W) tensor
# ─────────────────────────────────────────────────────────────────────────────
def _identity(x):           return x
def _hflip(x):              return torch.flip(x, dims=[3])
def _vflip(x):              return torch.flip(x, dims=[2])
def _rot90(x):              return torch.rot90(x, k=1, dims=[2, 3])
def _rot270(x):             return torch.rot90(x, k=3, dims=[2, 3])

TTA_TRANSFORMS = [_identity, _hflip, _vflip, _rot90, _rot270]


# ─────────────────────────────────────────────────────────────────────────────
# Core smart-inference entry point
# ─────────────────────────────────────────────────────────────────────────────
def smart_predict(
    model,
    image_tensor:    torch.Tensor,
    input_ids:       torch.Tensor,
    attention_mask:  torch.Tensor,
    tabular:         torch.Tensor,
    *,
    class_names:     List[str],
    temperature:     float = 1.0,
    n_tta:           int = TTA_AUGS_DEFAULT,
    n_mc:            int = MC_DROPOUT_DEFAULT,
    uc_hedge_thresh: float = UC_HEDGE_THRESHOLD,
) -> SmartPrediction:
    """Run TTA + MC-Dropout + hierarchical rule and return SmartPrediction."""
    device = image_tensor.device
    model.eval()                                # TTA pass uses eval mode

    # ── 1. TTA ensemble: deterministic mean over augmentations ──────────
    tta_probs_list: List[np.ndarray] = []
    augs = TTA_TRANSFORMS[: n_tta]
    with torch.no_grad():
        for aug in augs:
            x = aug(image_tensor)
            out = model(image=x, input_ids=input_ids,
                        attention_mask=attention_mask, tabular=tabular)
            # Temperature-scale the logits (T was fit for calibration)
            logits = out["pathology"] / max(temperature, 1e-3)
            p = F.softmax(logits, dim=-1)[0].cpu().numpy()
            tta_probs_list.append(p)
    tta_probs = np.stack(tta_probs_list, axis=0)        # (n_tta, n_classes)
    mean_probs_tta = tta_probs.mean(axis=0)

    # ── 2. MC-Dropout: stochastic sampling for uncertainty ─────────────
    # Switch to train mode so Dropout activates, but don't update BN.
    _enable_dropout_only(model)
    mc_probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_mc):
            out = model(image=image_tensor, input_ids=input_ids,
                        attention_mask=attention_mask, tabular=tabular)
            logits = out["pathology"] / max(temperature, 1e-3)
            p = F.softmax(logits, dim=-1)[0].cpu().numpy()
            mc_probs_list.append(p)
    model.eval()
    mc_probs = np.stack(mc_probs_list, axis=0)          # (n_mc, n_classes)
    mean_probs_mc = mc_probs.mean(axis=0)

    # Final probabilities: average TTA mean + MC mean (both equally weighted)
    final_probs = 0.5 * mean_probs_tta + 0.5 * mean_probs_mc

    # ── 3. Predictive entropy + mutual information (epistemic) ─────────
    # H(mean) = predictive entropy   — total uncertainty
    # H(mean) − E[H(p)] = mutual info — epistemic / model uncertainty
    eps = 1e-9
    pred_entropy   = float(-(final_probs * np.log(final_probs + eps)).sum())
    cond_entropies = -(mc_probs * np.log(mc_probs + eps)).sum(axis=1)
    cond_entropy   = float(cond_entropies.mean())
    mutual_info    = max(0.0, pred_entropy - cond_entropy)
    # Normalise to 0-1 by dividing by log(n_classes)
    norm = float(np.log(max(2, final_probs.shape[0])))
    pred_entropy_n = float(pred_entropy / norm) if norm > 0 else 0.0
    mutual_info_n  = float(mutual_info / norm) if norm > 0 else 0.0

    # ── 4. Variance summaries ───────────────────────────────────────────
    tta_std = float(tta_probs.std(axis=0).max())
    mc_std  = float(mc_probs.std(axis=0).max())

    # ── 5. Hierarchical UC rule ─────────────────────────────────────────
    pred_idx = int(final_probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(final_probs[pred_idx])

    is_hedged = False
    hedge_reason: Optional[str] = None
    if pred_class == "uc-mild" and "uc-moderate-sev" in class_names:
        modsev_idx = class_names.index("uc-moderate-sev")
        modsev_prob = float(final_probs[modsev_idx])
        if modsev_prob > uc_hedge_thresh:
            is_hedged = True
            hedge_reason = (
                f"AI is leaning UC-mild ({confidence*100:.0f}%) but "
                f"UC-moderate-severe is also plausible "
                f"({modsev_prob*100:.0f}%). Because under-calling severe UC "
                f"can delay treatment, we surface this as 'UC of uncertain "
                f"grade — recommend in-person review and sigmoidoscopy'.")
            # Keep `predicted_class` as uc-mild but flag is_hedged=True so
            # the UI shows the alert. We deliberately do NOT swap the class
            # — that would be a fabricated prediction. The hedge is honest.

    # ── 6. Differential diagnosis (top-3, drop < 5 %) ──────────────────
    order = np.argsort(-final_probs)[:3]
    differential = [
        {"class": class_names[i], "prob": float(final_probs[i])}
        for i in order if final_probs[i] >= 0.05
    ]

    return SmartPrediction(
        predicted_class = pred_class,
        confidence      = confidence,
        uncertainty     = pred_entropy_n,
        mutual_info     = mutual_info_n,
        differential    = differential,
        is_hedged       = is_hedged,
        hedge_reason    = hedge_reason,
        mean_probs      = final_probs,
        tta_probs       = tta_probs,
        mc_probs        = mc_probs,
        tta_std         = tta_std,
        mc_std          = mc_std,
        n_tta           = n_tta,
        n_mc            = n_mc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: enable Dropout layers without putting the WHOLE model in train mode
# (we don't want BatchNorm to update running stats during inference)
# ─────────────────────────────────────────────────────────────────────────────
def _enable_dropout_only(model: torch.nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                          torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (synthetic — no real model needed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─ smart_inference self-test ─")

    # Tiny fake model that returns deterministic logits + has a Dropout
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.4)
        def forward(self, image, input_ids, attention_mask, tabular):
            # Pretend uc-mild wins, uc-mod-sev close second
            # [polyps, uc-mild, uc-mod-sev, barretts, therapeutic]
            base = torch.tensor([[0.10, 0.45, 0.35, 0.05, 0.05]])
            # Add dropout-induced noise
            noisy = self.drop(base)
            return {"pathology": noisy}

    m = FakeModel()
    img = torch.randn(1, 3, 224, 224)
    ids = torch.zeros(1, 64, dtype=torch.long)
    msk = torch.ones(1, 64, dtype=torch.long)
    tab = torch.zeros(1, 12)
    class_names = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]

    sp = smart_predict(m, img, ids, msk, tab, class_names=class_names,
                       temperature=0.5, n_tta=5, n_mc=8)
    print(f"  predicted_class   = {sp.predicted_class}")
    print(f"  confidence        = {sp.confidence:.3f}")
    print(f"  uncertainty       = {sp.uncertainty:.3f}")
    print(f"  mutual_info       = {sp.mutual_info:.3f}")
    print(f"  is_hedged         = {sp.is_hedged}")
    print(f"  hedge_reason      = {sp.hedge_reason}")
    print(f"  differential      = {sp.differential}")
    print(f"  tta_std / mc_std  = {sp.tta_std:.3f} / {sp.mc_std:.3f}")
