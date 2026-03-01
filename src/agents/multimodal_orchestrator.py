"""
Multimodal Orchestrator Agent
─────────────────────────────
Central controller that coordinates all 6 specialist agents:
  1. UnifiedImageAgent       → visual evidence + Grad-CAM++
  2. TextAgent               → clinical text + BioBERT attention
  3. TabularRiskAgent        → patient data + SHAP scores
  4. FusionReasoningAgent    → cross-modal fusion diagnosis
  5. XAIAgent                → unified explainability report
  6. ClinicalRecommendationAgent → doctor-style clinical plan

Follows the Agentic AI pattern: each agent is autonomous,
communicates via typed dataclasses, orchestrator coordinates.
"""

from __future__ import annotations
import os
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch

from src.agents.unified_image_agent import UnifiedImageAgent, ImageEvidence
from src.agents.text_agent import TextAgent, TextEvidence
from src.agents.tabular_risk_agent import TabularRiskAgent, TabularEvidence
from src.agents.fusion_reasoning_agent import FusionReasoningAgent, FusionDiagnosis
from src.agents.xai_agent import XAIAgent, XAIReport
from src.agents.clinical_recommendation_agent import (
    ClinicalRecommendationAgent, ClinicalRecommendation)


@dataclass
class MultiModalDiagnosticOutput:
    # Agent outputs
    image_evidence: ImageEvidence
    text_evidence: TextEvidence
    tabular_evidence: TabularEvidence
    fusion_diagnosis: FusionDiagnosis
    xai_report: XAIReport
    clinical_recommendation: ClinicalRecommendation

    # Meta
    inference_time_ms: float
    case_id: Optional[str] = None


class MultiModalOrchestrator:
    """
    Full agentic pipeline orchestrator.
    Usage:
        orchestrator = MultiModalOrchestrator(model, tokenizer, device)
        output = orchestrator.run(image, input_ids, attention_mask, tabular, text)
    """

    def __init__(self, model, tokenizer, device: torch.device,
                 output_dir: str = "outputs/multimodal"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialise all agents
        self.image_agent = UnifiedImageAgent(model, device)
        self.text_agent = TextAgent(model, tokenizer, device)
        self.tabular_agent = TabularRiskAgent(model, device)
        self.fusion_agent = FusionReasoningAgent(model, device)
        self.xai_agent = XAIAgent(model, device)
        self.recommendation_agent = ClinicalRecommendationAgent()

        print("[Orchestrator] All 6 agents initialised.")

    def run(
        self,
        image: torch.Tensor,             # (1, 3, H, W)
        input_ids: torch.Tensor,          # (1, seq_len)
        attention_mask: torch.Tensor,     # (1, seq_len)
        tabular: torch.Tensor,            # (1, n_features)
        text: str = "",
        raw_image_np: Optional[np.ndarray] = None,
        case_id: Optional[str] = None,
        save: bool = False,
    ) -> MultiModalDiagnosticOutput:
        t0 = time.perf_counter()

        # ── Step 1: Image Agent ──────────────────────────────
        print("[Orchestrator] Step 1/6 — Image Agent")
        img_ev: ImageEvidence = self.image_agent.perceive(
            image, input_ids, attention_mask, tabular, raw_image_np)

        # ── Step 2: Text Agent ───────────────────────────────
        print("[Orchestrator] Step 2/6 — Text Agent")
        txt_ev: TextEvidence = self.text_agent.analyse(
            text, input_ids, attention_mask)

        # ── Step 3: Tabular Risk Agent ───────────────────────
        print("[Orchestrator] Step 3/6 — Tabular Risk Agent")
        tab_ev: TabularEvidence = self.tabular_agent.assess(tabular)

        # ── Step 4: Fusion Reasoning Agent ───────────────────
        print("[Orchestrator] Step 4/6 — Fusion Reasoning Agent")
        diag: FusionDiagnosis = self.fusion_agent.fuse(
            image, input_ids, attention_mask, tabular,
            img_ev=img_ev, txt_ev=txt_ev, tab_ev=tab_ev)

        # ── Step 5: XAI Agent ────────────────────────────────
        print("[Orchestrator] Step 5/6 — XAI Agent")
        xai: XAIReport = self.xai_agent.explain(
            image, input_ids, attention_mask, tabular,
            img_ev=img_ev, txt_ev=txt_ev, tab_ev=tab_ev, diag=diag)

        # ── Step 6: Clinical Recommendation Agent ────────────
        print("[Orchestrator] Step 6/6 — Clinical Recommendation Agent")
        rec: ClinicalRecommendation = self.recommendation_agent.generate(
            diag, tab_ev=tab_ev, img_ev=img_ev, xai=xai)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        output = MultiModalDiagnosticOutput(
            image_evidence=img_ev,
            text_evidence=txt_ev,
            tabular_evidence=tab_ev,
            fusion_diagnosis=diag,
            xai_report=xai,
            clinical_recommendation=rec,
            inference_time_ms=elapsed_ms,
            case_id=case_id,
        )

        if save:
            self._save(output, case_id)

        self._print_summary(output)
        return output

    def _save(self, output: MultiModalDiagnosticOutput,
              case_id: Optional[str] = None):
        prefix = f"{case_id}_" if case_id else ""
        case_dir = os.path.join(self.output_dir, case_id or "case")
        os.makedirs(case_dir, exist_ok=True)

        # Save XAI artifacts
        self.xai_agent.save_report(output.xai_report, case_dir, prefix)

        # Save clinical recommendation
        rec_path = os.path.join(case_dir, f"{prefix}clinical_report.txt")
        with open(rec_path, "w") as f:
            f.write(output.clinical_recommendation.full_report)

        # Save JSON summary
        summary = {
            "case_id": case_id,
            "pathology_class": output.fusion_diagnosis.pathology_class,
            "cancer_risk": output.fusion_diagnosis.cancer_risk_label,
            "cancer_stage": output.fusion_diagnosis.cancer_stage,
            "cancer_risk_score": output.fusion_diagnosis.cancer_risk_score,
            "stage_confidence": output.fusion_diagnosis.stage_confidence,
            "overall_confidence": output.fusion_diagnosis.overall_confidence,
            "uncertainty": output.xai_report.uncertainty,
            "risk_flags": output.fusion_diagnosis.all_risk_flags,
            "urgency": output.clinical_recommendation.urgency,
            "inference_ms": output.inference_time_ms,
            "modality_weights": output.xai_report.modality_weights,
        }
        json_path = os.path.join(case_dir, f"{prefix}summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Orchestrator] Results saved to {case_dir}")

    def _print_summary(self, output: MultiModalDiagnosticOutput):
        d = output.fusion_diagnosis
        r = output.clinical_recommendation
        print("\n" + "═" * 60)
        print("MULTIMODAL DIAGNOSTIC RESULT")
        print("═" * 60)
        print(f"  Pathology    : {d.pathology_class} ({d.pathology_confidence:.1%})")
        print(f"  Cancer Risk  : {d.cancer_risk_label} ({d.cancer_risk_score:.2f})")
        print(f"  Stage        : {d.cancer_stage} ({d.stage_confidence:.1%})")
        print(f"  Urgency      : {r.urgency}")
        print(f"  Uncertainty  : {output.xai_report.uncertainty:.2f}")
        print(f"  Inference    : {output.inference_time_ms:.1f} ms")
        print(f"  Risk Flags   : {output.fusion_diagnosis.all_risk_flags}")
        print("═" * 60)
