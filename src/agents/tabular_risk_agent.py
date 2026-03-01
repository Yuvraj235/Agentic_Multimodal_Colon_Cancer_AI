"""
Tabular Risk Agent
──────────────────
Processes structured patient data (TCGA tabular features) through the
TabTransformer encoder and produces SHAP-based feature importance scores
and a risk score with clinical interpretation.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.data.multimodal_dataset import TABULAR_FEATURES

RISK_THRESHOLDS = {
    "age_at_index": 60,
    "bmi": 30,
    "pack_years_smoked": 20,
    "tumor_stage_encoded": 2,
}

FEATURE_CLINICAL_NAMES = {
    "age_at_index": "Patient Age",
    "bmi": "Body Mass Index",
    "year_of_diagnosis": "Year of Diagnosis",
    "days_to_last_follow_up": "Follow-up Days",
    "cigarettes_per_day": "Cigarettes/Day",
    "pack_years_smoked": "Pack-Years Smoked",
    "alcohol_history": "Alcohol History",
    "gender": "Gender (Male)",
    "race_encoded": "Race",
    "tumor_stage_encoded": "Tumor Stage",
    "morphology_encoded": "Tumor Morphology",
    "site_of_resection_encoded": "Resection Site",
}


@dataclass
class TabularEvidence:
    tabular_vector: np.ndarray          # raw feature values
    feature_importance: Dict[str, float]  # SHAP-style importance
    risk_score: float                   # [0, 1] overall risk
    risk_level: str                     # "low" | "moderate" | "high" | "critical"
    risk_factors: List[str]             # clinical risk factor strings
    risk_flags: List[str]
    stage_prediction: int               # 0-3
    stage_confidence: float


def _kernel_shap_importance(tab_encoder, x: torch.Tensor,
                             n_samples: int = 64) -> np.ndarray:
    """
    Lightweight SHAP-style importance via perturbation.
    For each feature, ablate it (set to 0) and measure output change.
    """
    tab_encoder.eval()
    with torch.no_grad():
        baseline = tab_encoder(x)         # (1, 1, d_model)
        baseline_norm = baseline.norm()

    n_feat = x.shape[1]
    importances = np.zeros(n_feat, dtype=np.float32)

    for i in range(n_feat):
        x_ablated = x.clone()
        x_ablated[0, i] = 0.0
        with torch.no_grad():
            perturbed = tab_encoder(x_ablated)
            delta = (baseline - perturbed).norm().item()
        importances[i] = delta / (baseline_norm.item() + 1e-8)

    if importances.max() > 0:
        importances /= importances.max()
    return importances


class TabularRiskAgent:
    """Processes tabular patient data and generates risk assessment + SHAP."""

    def __init__(self, model, device: torch.device):
        self.model = model.to(device)
        self.tab_encoder = model.tabular_encoder
        self.device = device

    def assess(self, tabular: torch.Tensor,
               raw_values: Optional[np.ndarray] = None) -> TabularEvidence:
        tabular = tabular.to(self.device)

        # SHAP-style importances
        importances = _kernel_shap_importance(self.tab_encoder, tabular)

        # Stage prediction from full model (tabular-only forward with dummy image/text)
        self.model.eval()
        B = tabular.shape[0]
        dummy_img = torch.zeros(B, 3, 224, 224, device=self.device)
        dummy_ids = torch.ones(B, 128, dtype=torch.long, device=self.device)
        dummy_mask = torch.ones(B, 128, dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(dummy_img, dummy_ids, dummy_mask, tabular)
        stage_probs = F.softmax(out["staging"], dim=-1).cpu().numpy()[0]
        stage_pred = int(stage_probs.argmax())
        stage_conf = float(stage_probs[stage_pred])

        risk_probs = F.softmax(out["risk"], dim=-1).cpu().numpy()[0]
        risk_score = float(risk_probs[1])   # probability of high-risk class

        # Clinical risk level
        if risk_score >= 0.75:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.25:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Feature importance dict
        feat_imp = {FEATURE_CLINICAL_NAMES.get(f, f): float(v)
                    for f, v in zip(TABULAR_FEATURES, importances)}

        # Rule-based risk factors
        vec = tabular[0].cpu().numpy()
        risk_factors = []
        feat_vals = {f: float(v) for f, v in zip(TABULAR_FEATURES, vec)}

        if feat_vals.get("age_at_index", 0) > RISK_THRESHOLDS["age_at_index"]:
            risk_factors.append(f"Age > {RISK_THRESHOLDS['age_at_index']} (elevated risk)")
        if feat_vals.get("bmi", 0) > RISK_THRESHOLDS["bmi"]:
            risk_factors.append(f"BMI > {RISK_THRESHOLDS['bmi']} (obesity risk factor)")
        if feat_vals.get("pack_years_smoked", 0) > RISK_THRESHOLDS["pack_years_smoked"]:
            risk_factors.append(f"Pack-years smoked > {RISK_THRESHOLDS['pack_years_smoked']}")
        if feat_vals.get("alcohol_history", 0) > 0.5:
            risk_factors.append("Positive alcohol history")
        if feat_vals.get("tumor_stage_encoded", 0) >= 2:
            risk_factors.append("Tumour stage III/IV indicated")

        risk_flags = []
        if risk_level in ("high", "critical"):
            risk_flags.append("HIGH_CLINICAL_RISK")
        if stage_pred >= 2:
            risk_flags.append("ADVANCED_STAGE_INDICATED")
        if len(risk_factors) >= 3:
            risk_flags.append("MULTIPLE_RISK_FACTORS")

        return TabularEvidence(
            tabular_vector=vec,
            feature_importance=feat_imp,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            risk_flags=risk_flags,
            stage_prediction=stage_pred,
            stage_confidence=stage_conf,
        )
