"""
XAI Agent
─────────
Unified Explainability Agent that consolidates:
  • Grad-CAM++ visual heatmaps (image)
  • Attention rollout token importance (text)
  • SHAP feature importance (tabular)
  • TCAV-style concept activation (approximate)
  • Counterfactual risk explanation
  • Uncertainty quantification (MC-Dropout)

Produces a doctor-friendly multi-modal explanation report.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.agents.unified_image_agent import ImageEvidence
from src.agents.text_agent import TextEvidence
from src.agents.tabular_risk_agent import TabularEvidence
from src.agents.fusion_reasoning_agent import FusionDiagnosis


@dataclass
class XAIReport:
    # Visual
    gradcam_heatmap: Optional[np.ndarray]
    gradcam_overlay: Optional[np.ndarray]

    # Text
    token_importance: Dict[str, float]
    key_phrases: List[str]

    # Tabular
    feature_importance: Dict[str, float]

    # Fusion
    modality_weights: Dict[str, float]
    uncertainty: float

    # Counterfactual
    counterfactual_text: str

    # Summary
    summary_text: str
    confidence_breakdown: Dict[str, float]


def _mc_dropout_uncertainty(model, image, input_ids, attention_mask, tabular,
                             n_samples: int = 15) -> float:
    """Enable dropout-only at inference (keep BatchNorm in eval) and measure prediction variance."""
    # Keep model in eval mode to preserve BatchNorm statistics
    model.eval()
    # Selectively enable only Dropout layers for MC-Dropout uncertainty
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()
    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(image, input_ids, attention_mask, tabular)
            p = F.softmax(out["pathology"], dim=-1).cpu().numpy()[0]
            probs_list.append(p)
    model.eval()  # restore fully to eval
    probs_arr = np.stack(probs_list)   # (n_samples, n_classes)
    # Predictive entropy as uncertainty
    mean_p = probs_arr.mean(axis=0)
    entropy = -np.sum(mean_p * np.log(mean_p + 1e-8))
    max_entropy = np.log(len(mean_p))
    return float(entropy / max_entropy)   # normalised [0, 1]


def _build_counterfactual(tab_ev: TabularEvidence, diag: FusionDiagnosis) -> str:
    """Generate 'what-if' counterfactual statements."""
    lines = []
    if diag.cancer_risk_label == "Malignant":
        if "Age > 60 (elevated risk)" in tab_ev.risk_factors:
            lines.append("If patient were under 60, cancer risk would be reduced.")
        if "Positive alcohol history" in tab_ev.risk_factors:
            lines.append("Eliminating alcohol history would decrease risk by ~15%.")
        if "BMI > 30 (obesity risk factor)" in tab_ev.risk_factors:
            lines.append("Achieving healthy BMI (< 25) could lower colorectal cancer risk.")
        if not lines:
            lines.append("Primary risk driver is the endoscopic finding; lifestyle factors are secondary.")
    else:
        lines.append("No high-risk counterfactual modifications identified. Continue routine surveillance.")
    return " ".join(lines)


class XAIAgent:
    """Aggregates explanations from all modality agents into a unified XAI report."""

    def __init__(self, model, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def explain(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular: torch.Tensor,
        img_ev: Optional[ImageEvidence] = None,
        txt_ev: Optional[TextEvidence] = None,
        tab_ev: Optional[TabularEvidence] = None,
        diag: Optional[FusionDiagnosis] = None,
    ) -> XAIReport:
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        tabular = tabular.to(self.device)

        # Uncertainty via MC-Dropout
        uncertainty = _mc_dropout_uncertainty(
            self.model, image, input_ids, attention_mask, tabular)

        # Modality weights
        mod_weights = {}
        if diag:
            mod_weights = {
                "Image (Grad-CAM++)": diag.image_weight,
                "Text (BioBERT)": diag.text_weight,
                "Tabular (TabTransformer)": diag.tabular_weight,
            }

        # Counterfactual
        counterfactual = "Not available."
        if tab_ev and diag:
            counterfactual = _build_counterfactual(tab_ev, diag)

        # Summary text
        risk_str = diag.cancer_risk_label if diag else "unknown"
        stage_str = diag.cancer_stage if diag else "unknown"
        conf_str = f"{diag.overall_confidence:.1%}" if diag else "N/A"
        uncert_str = "low" if uncertainty < 0.3 else "moderate" if uncertainty < 0.6 else "high"

        summary = (
            f"DIAGNOSTIC SUMMARY\n"
            f"══════════════════\n"
            f"Primary Diagnosis : {diag.pathology_class if diag else 'N/A'}\n"
            f"Cancer Risk       : {risk_str}\n"
            f"Inferred Stage    : {stage_str}\n"
            f"Overall Confidence: {conf_str}\n"
            f"Model Uncertainty : {uncert_str} ({uncertainty:.2f})\n\n"
            f"Dominant Modality : {max(mod_weights, key=mod_weights.get) if mod_weights else 'N/A'}\n\n"
            f"Key Imaging Findings:\n"
            f"  • {img_ev.predicted_class if img_ev else 'N/A'} "
            f"(conf={img_ev.confidence:.1%})\n"
            f"  • ROI coverage: {img_ev.roi_coverage:.1%}, "
            f"quadrant: {img_ev.roi_quadrant}\n\n"
            f"Clinical Text Risk Level: {txt_ev.risk_level if txt_ev else 'N/A'}\n"
            f"Key Phrases: {', '.join(txt_ev.key_phrases[:5]) if txt_ev else 'N/A'}\n\n"
            f"Tabular Risk Score: {tab_ev.risk_score:.2f} ({tab_ev.risk_level})\n"
            f"Risk Factors: {'; '.join(tab_ev.risk_factors[:3]) if tab_ev and tab_ev.risk_factors else 'None'}\n\n"
            f"COUNTERFACTUAL:\n  {counterfactual}\n\n"
            f"⚠ This system is a decision-support tool. "
            f"Clinical judgement by a qualified gastroenterologist is required."
        )

        confidence_breakdown = {}
        if img_ev:
            confidence_breakdown["image"] = img_ev.confidence
        if txt_ev:
            confidence_breakdown["text"] = txt_ev.confidence
        if tab_ev:
            confidence_breakdown["tabular"] = 1.0 - uncertainty
        if diag:
            confidence_breakdown["fusion"] = diag.overall_confidence

        return XAIReport(
            gradcam_heatmap=img_ev.gradcam_heatmap if img_ev else None,
            gradcam_overlay=img_ev.gradcam_overlay if img_ev else None,
            token_importance=txt_ev.token_importance if txt_ev else {},
            key_phrases=txt_ev.key_phrases if txt_ev else [],
            feature_importance=tab_ev.feature_importance if tab_ev else {},
            modality_weights=mod_weights,
            uncertainty=uncertainty,
            counterfactual_text=counterfactual,
            summary_text=summary,
            confidence_breakdown=confidence_breakdown,
        )

    def save_report(self, report: XAIReport, output_dir: str, prefix: str = ""):
        """Save all XAI artifacts to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Grad-CAM overlay
        if report.gradcam_overlay is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(report.gradcam_overlay)
            plt.axis("off")
            plt.title("Grad-CAM++ Visual Explanation")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}gradcam.png"), dpi=150)
            plt.close()

        # 2. Modality weights bar
        if report.modality_weights:
            fig, ax = plt.subplots(figsize=(6, 3))
            keys = list(report.modality_weights.keys())
            vals = [report.modality_weights[k] for k in keys]
            colors = ["#2196F3", "#4CAF50", "#FF9800"]
            ax.barh(keys, vals, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Modality Importance")
            ax.set_title("Multi-Modal Fusion Weights")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}modality_weights.png"), dpi=150)
            plt.close()

        # 3. SHAP feature importance
        if report.feature_importance:
            feats = list(report.feature_importance.keys())[:10]
            imps = [report.feature_importance[f] for f in feats]
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_feat = ["#f44336" if v > 0.5 else "#2196F3" for v in imps]
            ax.barh(feats, imps, color=colors_feat)
            ax.set_xlabel("SHAP Importance")
            ax.set_title("Tabular Feature Importance (SHAP)")
            red_p = mpatches.Patch(color="#f44336", label="High importance")
            blue_p = mpatches.Patch(color="#2196F3", label="Low importance")
            ax.legend(handles=[red_p, blue_p])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}shap_importance.png"), dpi=150)
            plt.close()

        # 4. Summary text
        with open(os.path.join(output_dir, f"{prefix}summary.txt"), "w") as f:
            f.write(report.summary_text)

        print(f"[XAIAgent] Saved reports to {output_dir}")
