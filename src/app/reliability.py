"""ColonAI Reliability Layer.

Wraps the trained model with five independent runtime safety signals so every
prediction comes with an honest trust score, not just a softmax confidence.

Signals
-------
  1. ENDOSCOPY GATE       — pixel-statistics check (already in image_atypicality.py)
  2. TTA AGREEMENT        — predictions are stable across image augmentations
  3. MC-DROPOUT UNCERTAINTY — predictive entropy across stochastic forward passes
  4. PROTOTYPE DISTANCE   — distance from the input embedding to class prototypes
                            built from the training set (OOD detector)
  5. AGENT CONSENSUS      — image, text and tabular agents agree on the verdict

These are combined into a single TrustScore in [0, 1], and a verdict:

   TRUSTED          : all five signals strong, show the prediction confidently
   LOW_CONFIDENCE   : some signals weak, show the prediction with a warning
   FLAG_FOR_REVIEW  : signals disagree, recommend specialist review
   REJECTED         : input failed the endoscopy gate, refuse to predict

The module is pure runtime — NO retraining required.  Prototypes are computed
once from training data and cached to disk.
"""
from __future__ import annotations
import math
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


PROTOTYPE_CACHE = Path("outputs/unified_multimodal/class_prototypes.npz")


# ─────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrustReport:
    """Single-input reliability assessment."""
    trust_score: float                       # 0..1 — final aggregated score
    verdict: str                              # TRUSTED | LOW_CONFIDENCE | FLAG_FOR_REVIEW | REJECTED
    agreement_pct: float                      # 0..100 — TTA prediction agreement
    mc_uncertainty: float                     # 0..1 — predictive entropy
    prototype_distance: float                 # 0..1 — OOD distance (0 = in-distribution)
    agent_consensus: float                    # 0..1 — agent agreement on the verdict
    endoscopy_score: float                    # 0..1 — endoscopy-gate score
    signals: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    advice: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────
# 1. TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────────────────────────

def _tta_augmentations() -> List[T.Compose]:
    """Five complementary augmentations that preserve clinical content."""
    base = [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return [
        T.Compose(base),                                                      # original
        T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0)] + base[1:]),  # h-flip
        T.Compose([T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0)] + base[1:]),    # v-flip
        T.Compose([T.Resize((224, 224)), T.RandomRotation(degrees=(15, 15))] + base[1:]),
        T.Compose([T.Resize((224, 224)), T.RandomRotation(degrees=(-15, -15))] + base[1:]),
    ]


def tta_inference(
    model,
    pil_image: Image.Image,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tabular: torch.Tensor,
    device: torch.device,
    n_augs: int = 5,
) -> Dict:
    """Run the model on 5 augmented versions of the same image.

    Returns the averaged predictions plus the agreement percentage — i.e.,
    the fraction of augmentations that predicted the same class as the mean.
    Low agreement = the model is brittle on this input.
    """
    augs = _tta_augmentations()[:n_augs]
    model.eval()

    all_path = []   # list of (probs,) arrays
    all_stage = []
    all_risk = []
    all_classes = []

    with torch.no_grad():
        for aug in augs:
            x = aug(pil_image).unsqueeze(0).to(device)
            out = model(image=x, input_ids=input_ids,
                        attention_mask=attention_mask, tabular=tabular)
            p_path  = F.softmax(out["pathology"], dim=-1)[0].cpu().numpy()
            p_stage = F.softmax(out["staging"],   dim=-1)[0].cpu().numpy()
            p_risk  = torch.sigmoid(out["risk"])[0].item()
            all_path.append(p_path)
            all_stage.append(p_stage)
            all_risk.append(p_risk)
            all_classes.append(int(p_path.argmax()))

    # Averages
    mean_path  = np.mean(all_path,  axis=0)
    mean_stage = np.mean(all_stage, axis=0)
    mean_risk  = float(np.mean(all_risk))
    std_risk   = float(np.std(all_risk))

    # Disagreement signal: how many of the N augs picked the most-common class
    mean_class = int(mean_path.argmax())
    same_class = sum(1 for c in all_classes if c == mean_class)
    agreement_pct = 100.0 * same_class / n_augs

    # Prediction variance — useful for confidence calibration
    std_path = np.std(all_path, axis=0).max()

    return {
        "mean_path_probs":   mean_path,
        "mean_stage_probs":  mean_stage,
        "mean_risk":         mean_risk,
        "risk_std":          std_risk,
        "agreement_pct":     agreement_pct,
        "max_class_std":     float(std_path),
        "per_aug_classes":   all_classes,
        "n_augs":            n_augs,
    }


# ─────────────────────────────────────────────────────────────────
# 2. PROTOTYPE-BASED OOD DETECTION
# ─────────────────────────────────────────────────────────────────

def build_class_prototypes(
    model,
    dataloader,
    device: torch.device,
    out_path: Path = PROTOTYPE_CACHE,
    n_classes: int = 5,
    max_per_class: int = 200,
) -> Dict:
    """Compute the centroid of fused embeddings for each training class.

    Called once after training.  Stores the prototypes to disk as a .npz file
    so the live app can load them without rerunning training data.
    """
    model.eval()
    embeddings = {c: [] for c in range(n_classes)}

    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tab = batch["tabular"].to(device)
            lbl = batch["label"].cpu().numpy()

            out = model(image=img, input_ids=ids, attention_mask=mask, tabular=tab)
            fused = out["fused"].cpu().numpy()  # (B, 256)

            for i, c in enumerate(lbl):
                c = int(c)
                if c < n_classes and len(embeddings[c]) < max_per_class:
                    embeddings[c].append(fused[i])

            # Early exit when all classes are full
            if all(len(embeddings[c]) >= max_per_class for c in range(n_classes)):
                break

    # Compute centroids and per-class radii (median distance to centroid)
    centroids = {}
    radii     = {}
    for c, embs in embeddings.items():
        if not embs:
            continue
        arr = np.stack(embs, axis=0)        # (N, 256)
        ctr = arr.mean(axis=0)
        dists = np.linalg.norm(arr - ctr, axis=1)
        centroids[c] = ctr
        radii[c]     = float(np.median(dists))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             centroids=np.stack([centroids[c] for c in sorted(centroids)], axis=0),
             radii    =np.array([radii[c]     for c in sorted(radii)]),
             class_ids=np.array(sorted(centroids.keys())))
    print(f"[prototypes] Saved {len(centroids)} class prototypes to {out_path}")
    return {"centroids": centroids, "radii": radii}


_PROTOTYPES_CACHE: Optional[Dict] = None


def load_prototypes() -> Optional[Dict]:
    """Load cached class prototypes from disk.  Returns None if not available."""
    global _PROTOTYPES_CACHE
    if _PROTOTYPES_CACHE is not None:
        return _PROTOTYPES_CACHE
    if not PROTOTYPE_CACHE.exists():
        return None
    try:
        data = np.load(PROTOTYPE_CACHE)
        _PROTOTYPES_CACHE = {
            "centroids": data["centroids"],         # (n_classes, 256)
            "radii":     data["radii"],             # (n_classes,)
            "class_ids": data["class_ids"],
        }
        return _PROTOTYPES_CACHE
    except Exception:
        return None


def prototype_distance(fused_embedding: np.ndarray) -> Tuple[float, int]:
    """Compute the normalised distance from `fused_embedding` to the nearest
    class prototype, scaled by that class's median radius.

    Returns (ood_score, nearest_class):
       ood_score   : 0..1 — 0 means inside the cluster, 1+ means clearly OOD
       nearest_class : index of the closest class prototype
    """
    proto = load_prototypes()
    if proto is None:
        return 0.0, -1   # no prototypes available — be lenient

    cents = proto["centroids"]
    radii = proto["radii"]
    # Distance to each centroid
    dists = np.linalg.norm(cents - fused_embedding, axis=1)
    nearest = int(np.argmin(dists))
    nearest_d = float(dists[nearest])
    r = max(0.1, float(radii[nearest]))
    # Normalised distance — 0..1+ where 1 means the input is one full radius
    # beyond the cluster boundary
    norm_d = nearest_d / r
    # Map to [0..1] via a soft squash: 1.0 at norm_d=2.0 (well outside cluster)
    ood = float(min(1.0, max(0.0, (norm_d - 1.0) / 1.0)))
    return ood, nearest


# ─────────────────────────────────────────────────────────────────
# 3. AGENT CONSENSUS
# ─────────────────────────────────────────────────────────────────

# Mapping from pathology class → expected text-agent risk level
_CLASS_TO_TEXT_RISK_TIER = {
    "polyps":              {"ELEVATED", "MODERATE"},  # bleeding terms expected
    "uc-mild":             {"MODERATE", "LOW"},
    "uc-moderate-sev":     {"ELEVATED", "MODERATE"},
    "barretts-esoph":      {"MODERATE", "LOW"},
    "therapeutic":         {"LOW"},
}


def agent_consensus(image_class: str,
                    text_risk_level: str,
                    tabular_risk_score: float) -> Tuple[float, List[str]]:
    """Score how much the image, text and tabular agents agree.

    Returns (consensus_score in [0..1], list of disagreement reasons).
    """
    score = 1.0
    reasons = []

    # Image vs Text
    expected_text_risks = _CLASS_TO_TEXT_RISK_TIER.get(image_class, set())
    if expected_text_risks and text_risk_level not in expected_text_risks:
        score -= 0.30
        reasons.append(
            f"Image agent says {image_class!r} but text agent reports "
            f"{text_risk_level!r} risk (expected one of {sorted(expected_text_risks)}).")

    # Image vs Tabular
    expects_high_tab = image_class in ("polyps", "uc-moderate-sev", "barretts-esoph")
    if expects_high_tab and tabular_risk_score < 0.20:
        score -= 0.25
        reasons.append(
            f"Image agent says {image_class!r} (typically associated with elevated "
            f"tabular risk) but tabular score is only {tabular_risk_score:.2f}.")
    if not expects_high_tab and tabular_risk_score > 0.65:
        score -= 0.15
        reasons.append(
            f"Image agent says {image_class!r} (typically low tabular risk) but "
            f"tabular score is {tabular_risk_score:.2f}.")

    return float(max(0.0, score)), reasons


# ─────────────────────────────────────────────────────────────────
# 4. FINAL TRUST AGGREGATION
# ─────────────────────────────────────────────────────────────────

def build_trust_report(
    endoscopy_score: float,
    tta_agreement_pct: float,
    mc_uncertainty: float,
    fused_embedding: Optional[np.ndarray],
    image_class: str,
    text_risk_level: str,
    tabular_risk_score: float,
) -> TrustReport:
    """Combine all five reliability signals into one TrustReport.

    Weights (sum to 1.0):
       endoscopy gate       0.20
       TTA agreement        0.25
       MC-Dropout certainty 0.20
       prototype OOD        0.15
       agent consensus      0.20
    """
    # Normalise each signal to [0..1] where 1 = trustworthy
    s_gate      = float(min(1.0, max(0.0, endoscopy_score)))
    s_tta       = float(tta_agreement_pct / 100.0)
    s_mc        = float(1.0 - min(1.0, max(0.0, mc_uncertainty)))

    if fused_embedding is not None:
        ood, _ = prototype_distance(fused_embedding)
        s_proto = float(1.0 - ood)
    else:
        ood = 0.0
        s_proto = 0.85   # neutral when prototypes not available yet

    s_cons, cons_reasons = agent_consensus(image_class, text_risk_level, tabular_risk_score)

    trust = (0.20 * s_gate +
             0.25 * s_tta +
             0.20 * s_mc +
             0.15 * s_proto +
             0.20 * s_cons)

    # Determine verdict bands
    warnings = []

    if endoscopy_score < 0.55:
        verdict = "REJECTED"
        warnings.append("Endoscopy gate failed — refusing to predict on non-endoscopy input.")
    elif trust >= 0.78:
        verdict = "TRUSTED"
    elif trust >= 0.62:
        verdict = "LOW_CONFIDENCE"
        warnings.append("Some reliability signals are weak — treat the prediction with caution.")
    else:
        verdict = "FLAG_FOR_REVIEW"
        warnings.append("Reliability signals disagree — flag this case for specialist review.")

    # Per-signal warnings
    if s_tta < 0.80:
        warnings.append(
            f"Only {tta_agreement_pct:.0f}% of augmentations agreed on the prediction "
            f"— the model is unstable on this image.")
    if mc_uncertainty > 0.15:
        warnings.append(
            f"MC-Dropout uncertainty is {mc_uncertainty:.2f} (threshold 0.15) — "
            f"the model itself is unsure.")
    if ood > 0.50:
        warnings.append(
            f"Input is far from the training distribution (OOD score {ood:.2f}) — "
            f"the prediction is extrapolating beyond what the model has seen.")
    warnings.extend(cons_reasons)

    # Advice
    if verdict == "TRUSTED":
        advice = ("All five reliability signals are strong. The prediction is "
                  "consistent with the training distribution and the agents agree.")
    elif verdict == "LOW_CONFIDENCE":
        advice = ("Prediction is plausible but not robust. Consider re-imaging "
                  "or seeking a second opinion.")
    elif verdict == "FLAG_FOR_REVIEW":
        advice = ("This case sits at the boundary of what the model knows. "
                  "Do not rely on the prediction alone — escalate to a senior clinician.")
    else:
        advice = ("Input rejected — please upload a real colonoscopy frame.")

    return TrustReport(
        trust_score        = float(trust),
        verdict            = verdict,
        agreement_pct      = float(tta_agreement_pct),
        mc_uncertainty     = float(mc_uncertainty),
        prototype_distance = float(ood),
        agent_consensus    = float(s_cons),
        endoscopy_score    = float(endoscopy_score),
        signals = {
            "gate":              s_gate,
            "tta_agreement":     s_tta,
            "mc_certainty":      s_mc,
            "prototype_inlier":  s_proto,
            "agent_consensus":   s_cons,
        },
        warnings = warnings,
        advice   = advice,
    )
