"""
Fusion Reasoning Agent
───────────────────────
Combines outputs of Image, Text, and Tabular agents using the
cross-attention Fusion Transformer to produce the final diagnosis,
cancer stage, and modality importance weights.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.agents.unified_image_agent import ImageEvidence
from src.agents.text_agent import TextEvidence
from src.agents.tabular_risk_agent import TabularEvidence

STAGE_LABELS = ["No Cancer", "Stage I", "Stage II", "Stage III/IV"]
PATHOLOGY_CLASSES = [
    "polyps",
    "uc-mild",
    "uc-moderate-sev",
    "barretts-esoph",
    "therapeutic",
]


@dataclass
class FusionDiagnosis:
    # Primary outputs
    pathology_class: str
    pathology_confidence: float
    pathology_probs: Dict[str, float]

    cancer_stage: str
    stage_confidence: float
    stage_probs: Dict[str, float]

    cancer_risk_score: float
    cancer_risk_label: str            # "Benign" | "Malignant"

    # Modality weights
    image_weight: float
    text_weight: float
    tabular_weight: float

    # Aggregated flags
    all_risk_flags: List[str]
    overall_confidence: float

    # Raw fused embedding for XAI
    fused_embedding: Optional[np.ndarray]


class FusionReasoningAgent:
    """
    Takes encoded outputs from three modality agents and runs the
    fusion transformer to generate the final diagnosis.
    """

    def __init__(self, model, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def fuse(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular: torch.Tensor,
        img_ev: Optional[ImageEvidence] = None,
        txt_ev: Optional[TextEvidence] = None,
        tab_ev: Optional[TabularEvidence] = None,
    ) -> FusionDiagnosis:
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        tabular = tabular.to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(image, input_ids, attention_mask, tabular)

        # Pathology
        path_probs = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
        path_idx = int(path_probs.argmax())
        path_conf = float(path_probs[path_idx])

        # Staging
        stage_probs = F.softmax(out["staging"], dim=-1).cpu().numpy()[0]
        stage_idx = int(stage_probs.argmax())
        stage_conf = float(stage_probs[stage_idx])

        # Binary risk
        risk_probs = F.softmax(out["risk"], dim=-1).cpu().numpy()[0]
        risk_score = float(risk_probs[1])
        risk_label = "Malignant" if risk_score >= 0.5 else "Benign"

        # Modality weights
        mod_w = out["mod_weights"][0].cpu().numpy()

        # Aggregate risk flags
        all_flags = []
        if img_ev:
            all_flags.extend(img_ev.risk_flags)
        if txt_ev:
            all_flags.extend(txt_ev.risk_flags)
        if tab_ev:
            all_flags.extend(tab_ev.risk_flags)

        # Overall confidence: weighted average of head confidences
        overall_conf = float(
            0.5 * path_conf + 0.3 * stage_conf + 0.2 * (1 - abs(risk_score - 0.5) * 2))

        fused_emb = out["fused"][0].cpu().numpy()

        return FusionDiagnosis(
            pathology_class=PATHOLOGY_CLASSES[path_idx],
            pathology_confidence=path_conf,
            pathology_probs={c: float(p) for c, p in zip(PATHOLOGY_CLASSES, path_probs)},
            cancer_stage=STAGE_LABELS[stage_idx],
            stage_confidence=stage_conf,
            stage_probs={s: float(p) for s, p in zip(STAGE_LABELS, stage_probs)},
            cancer_risk_score=risk_score,
            cancer_risk_label=risk_label,
            image_weight=float(mod_w[0]),
            text_weight=float(mod_w[1]),
            tabular_weight=float(mod_w[2]),
            all_risk_flags=list(set(all_flags)),
            overall_confidence=overall_conf,
            fused_embedding=fused_emb,
        )
