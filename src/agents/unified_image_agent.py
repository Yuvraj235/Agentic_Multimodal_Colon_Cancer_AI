"""
Unified Image Agent
───────────────────
Vision perception + Grad-CAM++ XAI for the UnifiedMultiModalTransformer.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

PATHOLOGY_CLASSES = [
    "polyps",
    "uc-mild",
    "uc-moderate-sev",
    "barretts-esoph",
    "therapeutic",
]


@dataclass
class ImageEvidence:
    predicted_class: str
    class_idx: int
    confidence: float
    probabilities: Dict[str, float]
    gradcam_heatmap: Optional[np.ndarray]
    gradcam_overlay: Optional[np.ndarray]
    roi_coverage: float
    roi_quadrant: str
    max_activation: float
    risk_flags: List[str]


class GradCAMPlusPlus:
    """Grad-CAM++ hooked onto a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def generate(self, image: torch.Tensor, class_idx: int,
                 input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 tabular: torch.Tensor) -> np.ndarray:
        self.model.eval()
        image = image.detach().requires_grad_(True)
        out = self.model(image, input_ids, attention_mask, tabular)
        score = out["pathology"][0, class_idx]
        self.model.zero_grad()
        score.backward()

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            return np.zeros((7, 7), dtype=np.float32)

        grads_sq = grads ** 2
        denom = 2 * grads_sq + acts * grads ** 3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom) * 1e-10)
        alpha = grads_sq / denom
        weights = (alpha * F.relu(score.exp() * grads)).mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)


class UnifiedImageAgent:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        target = model.get_image_target_layer()
        self.gradcam = GradCAMPlusPlus(model, target)

    @torch.no_grad()
    def _probs(self, image, input_ids, attention_mask, tabular):
        out = self.model(image, input_ids, attention_mask, tabular)
        return F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]

    def perceive(self, image: torch.Tensor, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor, tabular: torch.Tensor,
                 raw_np: Optional[np.ndarray] = None) -> ImageEvidence:
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        tabular = tabular.to(self.device)

        path_probs = self._probs(image, input_ids, attention_mask, tabular)
        class_idx = int(path_probs.argmax())
        conf = float(path_probs[class_idx])

        cam = self.gradcam.generate(image, class_idx, input_ids, attention_mask, tabular)

        overlay = None
        if raw_np is not None:
            h, w = raw_np.shape[:2]
            cam_r = cv2.resize(cam, (w, h))
            hmap = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
            overlay = (0.4 * hmap + 0.6 * raw_np).astype(np.uint8)

        high = cam > 0.5
        coverage = float(high.mean())
        H, W = cam.shape
        q = {"upper-left": cam[:H // 2, :W // 2].mean(),
             "upper-right": cam[:H // 2, W // 2:].mean(),
             "lower-left": cam[H // 2:, :W // 2].mean(),
             "lower-right": cam[H // 2:, W // 2:].mean()}
        quadrant = max(q, key=q.get)
        max_act = float(cam.max())

        flags = []
        if class_idx in (0, 1, 2, 3):  # any pathological finding
            flags.append("PATHOLOGICAL_FINDING")
        if class_idx in (1, 2):   # UC mild or moderate-severe
            flags.append("INFLAMMATORY_BOWEL_DISEASE")
        if class_idx == 2:         # UC moderate-severe
            flags.append("HIGH_MALIGNANCY_RISK")
        if coverage > 0.30:
            flags.append("EXTENDED_TISSUE_INVOLVEMENT")
        if max_act > 0.85:
            flags.append("HIGH_ACTIVATION_LESION")
        if conf < 0.55:
            flags.append("UNCERTAIN_PREDICTION")

        return ImageEvidence(
            predicted_class=PATHOLOGY_CLASSES[class_idx],
            class_idx=class_idx,
            confidence=conf,
            probabilities={c: float(p) for c, p in zip(PATHOLOGY_CLASSES, path_probs)},
            gradcam_heatmap=cam,
            gradcam_overlay=overlay,
            roi_coverage=coverage,
            roi_quadrant=quadrant,
            max_activation=max_act,
            risk_flags=flags,
        )
