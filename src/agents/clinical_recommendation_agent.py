"""
Clinical Recommendation Agent
──────────────────────────────
Generates gastroenterologist-style clinical recommendations based on:
  • Fusion diagnosis (stage, risk, pathology)
  • Tabular evidence (risk factors)
  • Image evidence (ROI coverage, flags)
  • XAI uncertainty

Follows NICE / BSG colon cancer staging guidelines.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from src.agents.fusion_reasoning_agent import FusionDiagnosis
from src.agents.tabular_risk_agent import TabularEvidence
from src.agents.unified_image_agent import ImageEvidence
from src.agents.xai_agent import XAIReport


SURVEILLANCE_INTERVALS = {
    "No Cancer": "Routine 5-year colonoscopy screening (low-risk protocol).",
    "Stage I": "3-year surveillance colonoscopy. Surgical consultation for resection.",
    "Stage II": "Annual colonoscopy surveillance. Oncology referral. CT chest/abdomen/pelvis staging.",
    "Stage III/IV": "Urgent oncology referral within 2 weeks. Multi-disciplinary team (MDT) review. "
                    "Consider neoadjuvant chemotherapy (FOLFOX/FOLFIRI). Staging CT + PET scan.",
}

POLYP_PROTOCOLS = {
    "1-2 low-risk adenomas": "5-year surveillance.",
    "3-4 adenomas or 1 large (≥10mm)": "3-year surveillance.",
    ">5 adenomas or serrated adenoma": "1-year surveillance. Genetic counselling if Lynch syndrome suspected.",
}

PATHOLOGY_ACTIONS = {
    "polyps": (
        "Colonic polyp identified. Polypectomy if not already performed. "
        "Histopathology for adenoma subtype. Surveillance colonoscopy per guidelines."
    ),
    "uc-mild": (
        "Mild ulcerative colitis confirmed. Optimise 5-ASA therapy. "
        "Surveillance colonoscopy in 2 years."
    ),
    "uc-moderate-sev": (
        "Moderate-to-severe ulcerative colitis. Escalate to immunomodulators/biologics. "
        "Urgent gastroenterology referral. Annual surveillance colonoscopy."
    ),
    "barretts-esoph": (
        "Barrett's esophagus or esophagitis identified. Proton pump inhibitor therapy. "
        "Endoscopic surveillance per Barrett's protocol (2–3 year intervals)."
    ),
    "therapeutic": "Post-therapeutic site confirmed. Tattoo resection margin for surveillance. 3-month follow-up endoscopy.",
}


@dataclass
class ClinicalRecommendation:
    urgency: str          # "Routine" | "Elective" | "Urgent" | "Emergency"
    primary_action: str
    surveillance: str
    referrals: List[str]
    investigations: List[str]
    lifestyle_advice: List[str]
    full_report: str
    disclaimer: str


class ClinicalRecommendationAgent:
    """Generates clinical recommendations simulating a gastroenterologist."""

    DISCLAIMER = (
        "\n\n─────────────────────────────────────────\n"
        "⚠ DISCLAIMER: This AI-generated report is a clinical decision-support tool only. "
        "It does NOT replace the professional judgement of a qualified gastroenterologist "
        "or oncologist. All recommendations must be verified by a licensed clinician "
        "before any clinical action is taken. This system has not been validated for "
        "standalone diagnostic use.\n"
        "─────────────────────────────────────────"
    )

    def generate(
        self,
        diag: FusionDiagnosis,
        tab_ev: Optional[TabularEvidence] = None,
        img_ev: Optional[ImageEvidence] = None,
        xai: Optional[XAIReport] = None,
    ) -> ClinicalRecommendation:

        stage = diag.cancer_stage
        risk_label = diag.cancer_risk_label
        path_class = diag.pathology_class
        uncertainty = xai.uncertainty if xai else 0.5

        # ── Urgency ──────────────────────────────────────────────
        if stage == "Stage III/IV" or risk_label == "Malignant":
            urgency = "Urgent"
        elif stage == "Stage II":
            urgency = "Elective"
        elif "PATHOLOGICAL_FINDING" in diag.all_risk_flags:
            urgency = "Elective"
        else:
            urgency = "Routine"

        # ── Primary Action ────────────────────────────────────────
        primary_action = PATHOLOGY_ACTIONS.get(path_class, "Clinical review recommended.")
        if uncertainty > 0.65:
            primary_action += (
                " Note: Model uncertainty is HIGH. "
                "Expert review strongly recommended before any intervention."
            )

        # ── Surveillance ──────────────────────────────────────────
        surveillance = SURVEILLANCE_INTERVALS.get(stage, "Gastroenterology review required.")

        # ── Referrals ─────────────────────────────────────────────
        referrals = []
        if stage in ("Stage II", "Stage III/IV") or risk_label == "Malignant":
            referrals.append("Colorectal Surgery — tumour resection assessment")
            referrals.append("Oncology — chemotherapy/radiotherapy planning")
        if path_class in ("polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph"):
            referrals.append("Gastroenterology — specialist endoscopy review")
        if tab_ev and "MULTIPLE_RISK_FACTORS" in tab_ev.risk_flags:
            referrals.append("Genetic Counselling — Lynch syndrome / familial CRC risk")
        if not referrals:
            referrals.append("General Practitioner — routine follow-up")

        # ── Investigations ─────────────────────────────────────────
        investigations = []
        if stage != "No Cancer":
            investigations += [
                "CT Chest/Abdomen/Pelvis (staging)",
                "CEA tumour marker (baseline)",
                "FBC, LFTs, U&E",
            ]
        if stage == "Stage III/IV":
            investigations += ["PET-CT scan", "MRI pelvis (rectal cancer)"]
        if path_class in ("polyps", "uc-moderate-sev", "barretts-esoph"):
            investigations.append("Tissue biopsy for histopathology")
        if uncertainty > 0.5:
            investigations.append("Repeat colonoscopy for verification")

        # ── Lifestyle Advice ───────────────────────────────────────
        lifestyle = ["Maintain healthy weight (BMI 18.5–24.9)",
                     "Increase dietary fibre (≥30g/day)",
                     "Limit red/processed meat (<500g/week)",
                     "Cease smoking if applicable",
                     "Limit alcohol to <14 units/week"]
        if tab_ev and "Positive alcohol history" in tab_ev.risk_factors:
            lifestyle.insert(0, "⚠ PRIORITY: Reduce alcohol intake significantly")
        if tab_ev and any("obesity" in r for r in tab_ev.risk_factors):
            lifestyle.insert(0, "⚠ PRIORITY: Weight reduction programme referral")

        # ── Full Report ────────────────────────────────────────────
        mod_str = (
            f"Image: {diag.image_weight:.0%}, "
            f"Text: {diag.text_weight:.0%}, "
            f"Tabular: {diag.tabular_weight:.0%}"
        ) if diag.image_weight else "N/A"

        full_report = (
            f"CLINICAL RECOMMENDATION REPORT\n"
            f"══════════════════════════════\n"
            f"URGENCY          : {urgency}\n"
            f"PATHOLOGY        : {path_class}\n"
            f"CANCER RISK      : {risk_label} (score: {diag.cancer_risk_score:.2f})\n"
            f"INFERRED STAGE   : {stage} (conf: {diag.stage_confidence:.1%})\n"
            f"MODEL UNCERTAINTY: {uncertainty:.2f}\n"
            f"MODALITY WEIGHTS : {mod_str}\n\n"
            f"PRIMARY ACTION:\n  {primary_action}\n\n"
            f"SURVEILLANCE:\n  {surveillance}\n\n"
            f"REFERRALS:\n" + "\n".join(f"  • {r}" for r in referrals) + "\n\n"
            f"INVESTIGATIONS:\n" + "\n".join(f"  • {i}" for i in investigations) + "\n\n"
            f"LIFESTYLE ADVICE:\n" + "\n".join(f"  • {l}" for l in lifestyle)
            + self.DISCLAIMER
        )

        return ClinicalRecommendation(
            urgency=urgency,
            primary_action=primary_action,
            surveillance=surveillance,
            referrals=referrals,
            investigations=investigations,
            lifestyle_advice=lifestyle,
            full_report=full_report,
            disclaimer=self.DISCLAIMER,
        )
