# src/agents/pathology_agent.py

import torch
import torch.nn.functional as F
from src.contracts.modality_report import ModalityReport


class PathologyReasoningAgent:
    """
    Vision-based pathology reasoning agent.
    Consumes visual features and returns a ModalityReport.
    """

    def __init__(self, head, label_map, device="cpu"):
        self.head = head.to(device)
        self.label_map = label_map
        self.idx_to_class = {v: k for k, v in label_map.items()}
        self.device = device

        self.head.eval()

    @torch.no_grad()
    def reason(self, perception_output):
        """
        perception_output: dict with key 'features'
        """

        if "features" not in perception_output:
            raise KeyError("Perception output must contain 'features'")

        features = perception_output["features"].to(self.device)

        logits = self.head(features)
        probs = F.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        confidence = confidence.item()

        probabilities = {
            self.idx_to_class[i]: probs[0, i].item()
            for i in range(probs.shape[1])
        }

        predicted_label = self.idx_to_class[pred_idx]

        return ModalityReport(
            modality="vision",
            evidence={
                "predicted_class": predicted_label,
                "class_index": pred_idx,
                "probabilities": probabilities,
                "features": features
            },
            confidence=confidence,
            explanation={},     # filled later by ExplanationAgent
            risk_flags=[]
        )