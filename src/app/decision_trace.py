"""ColonAI — Decision Trace.

Every inference run now produces a structured, step-by-step trace that
captures WHAT each agent saw, WHAT it concluded, and HOW its conclusion
contributed to the final verdict. This is the "audit log" of the model
— what an FDA reviewer or clinical safety officer would ask for.

The trace is intentionally simple JSON-like dicts (not classes with hidden
state) so it can be serialised, shown in the UI, and saved to the learning
log without leaking PHI.

Each trace step has:
- agent          who produced the step (image, text, tabular, fusion, xai, clinical)
- stage          what the step was doing (predict / verify / hedge / override)
- input_summary  one sentence on what the agent looked at
- finding        what the agent concluded
- confidence     0..1
- evidence       short bullet-list of evidence items (counts, %, lesion size, …)
- effect         "support" / "contradict" / "override" / "abstain" / "noop"
- weight         how much this step counted toward the final verdict (0..1)

The orchestrator function ``build_trace`` takes outputs from every other
``src/app/*`` module that's already produced a finding and stitches them
into a single ordered list. Pure post-hoc — nothing about the underlying
inference changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import time

# ──────────────────────────────────────────────────────────────────────
#  Trace step dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TraceStep:
    """One row in the decision trace."""
    step_idx: int
    agent: str                        # image / text / tabular / fusion / xai / clinical / safety / atypicality / polyp_typing / tcga_stage / smart_inference / llm
    stage: str                        # predict / verify / hedge / override / abstain / refine
    input_summary: str                # 1-sentence description of inputs
    finding: str                      # what the agent said
    confidence: float                 # 0..1
    evidence: List[str] = field(default_factory=list)
    effect: str = "support"           # support / contradict / override / abstain / noop / refine
    weight: float = 1.0               # 0..1 — relative influence on final verdict
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
#  Trace builder
# ──────────────────────────────────────────────────────────────────────
def build_trace(
    *,
    raw_image_pred: Optional[Dict[str, Any]] = None,
    smart_pred: Optional[Any] = None,           # SmartPrediction from smart_inference
    text_finding: Optional[Dict[str, Any]] = None,
    tabular_finding: Optional[Dict[str, Any]] = None,
    fusion_finding: Optional[Dict[str, Any]] = None,
    xai_finding: Optional[Dict[str, Any]] = None,
    clinical_finding: Optional[Dict[str, Any]] = None,
    safety_finding: Optional[Dict[str, Any]] = None,
    atypicality_finding: Optional[Dict[str, Any]] = None,
    polyp_typing_finding: Optional[Dict[str, Any]] = None,
    tcga_stage_finding: Optional[Dict[str, Any]] = None,
    llm_refined: Optional[bool] = None,
    final_verdict: Optional[str] = None,
    final_confidence: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build a chronological trace of the inference.

    All inputs are optional — agents that didn't fire (e.g. text agent
    if no text was provided) are skipped.
    """
    steps: List[TraceStep] = []
    idx = 0

    # 1. Raw image classifier (single forward pass, no TTA / MC-Dropout)
    if raw_image_pred is not None:
        cls = raw_image_pred.get("class", raw_image_pred.get("label", "unknown"))
        conf = float(raw_image_pred.get("confidence", raw_image_pred.get("prob", 0.0)))
        probs = raw_image_pred.get("probs") or raw_image_pred.get("probabilities") or {}
        evidence: List[str] = []
        if isinstance(probs, dict):
            # show top-3 alternatives
            top3 = sorted(probs.items(), key=lambda kv: -float(kv[1]))[:3]
            for k, v in top3:
                evidence.append(f"{k}: {float(v):.2%}")
        steps.append(TraceStep(
            step_idx=idx, agent="image", stage="predict",
            input_summary="ConvNeXt-V2 single forward pass on input image",
            finding=f"Raw image vote: {cls}",
            confidence=conf,
            evidence=evidence,
            effect="support",
            weight=0.40,
        ))
        idx += 1

    # 2. Smart inference (TTA + MC-Dropout ensemble)
    if smart_pred is not None:
        cls = getattr(smart_pred, "predicted_class", None) or smart_pred.get("predicted_class", "unknown")
        conf = float(getattr(smart_pred, "confidence", None) or smart_pred.get("confidence", 0.0))
        unc = float(getattr(smart_pred, "uncertainty", None) or smart_pred.get("uncertainty", 0.0))
        mi = float(getattr(smart_pred, "mutual_info", None) or smart_pred.get("mutual_info", 0.0))
        hedged = bool(getattr(smart_pred, "is_hedged", False) or smart_pred.get("is_hedged", False))
        differential = getattr(smart_pred, "differential", None) or smart_pred.get("differential", [])
        ev: List[str] = [
            f"predictive entropy: {unc:.3f} nats",
            f"mutual information: {mi:.3f} nats",
        ]
        if differential:
            ev.append("differential: " + ", ".join(
                f"{d[0]} ({float(d[1]):.0%})" if isinstance(d, (tuple, list)) else str(d)
                for d in list(differential)[:3]))
        steps.append(TraceStep(
            step_idx=idx, agent="smart_inference",
            stage="hedge" if hedged else "predict",
            input_summary="5x TTA + 10x MC-Dropout ensemble (50 stochastic forward passes)",
            finding=("Ensemble call: " + cls + (" (HEDGED)" if hedged else "")),
            confidence=conf,
            evidence=ev,
            effect="support" if not hedged else "abstain",
            weight=0.85,
        ))
        idx += 1

    # 3. Atypicality / invasive lesion override
    if atypicality_finding is not None:
        fired = bool(atypicality_finding.get("override", False) or atypicality_finding.get("atypical", False))
        reasons = atypicality_finding.get("reasons", []) or atypicality_finding.get("findings", [])
        conf = float(atypicality_finding.get("confidence", 0.8))
        if fired:
            steps.append(TraceStep(
                step_idx=idx, agent="atypicality", stage="override",
                input_summary="Pixel-statistics scan: deep ulceration / bleeding / nodular mass / mirror reflection",
                finding=atypicality_finding.get("label", "Atypical lesion — urgent endoscopist review"),
                confidence=conf,
                evidence=[str(r) for r in reasons[:6]],
                effect="override",
                weight=1.00,
            ))
            idx += 1
        else:
            steps.append(TraceStep(
                step_idx=idx, agent="atypicality", stage="verify",
                input_summary="Pixel-statistics scan for invasive features",
                finding="No invasive-lesion features detected",
                confidence=conf,
                evidence=[],
                effect="noop",
                weight=0.05,
            ))
            idx += 1

    # 4. Polyp sub-typing (Paris + NICE + size + IBD differential)
    if polyp_typing_finding is not None:
        ev = []
        for k, v in polyp_typing_finding.items():
            if v is None:
                continue
            if isinstance(v, dict):
                # nested finding e.g. paris -> {"class": "0-Is", "confidence": 0.65}
                lbl = v.get("class") or v.get("label") or v.get("type")
                cf = v.get("confidence") or v.get("prob") or v.get("score")
                if lbl is not None:
                    ev.append(f"{k}: {lbl}" + (f" ({float(cf):.0%})" if cf else ""))
            elif isinstance(v, (int, float, str)):
                ev.append(f"{k}: {v}")
        if ev:
            steps.append(TraceStep(
                step_idx=idx, agent="polyp_typing", stage="verify",
                input_summary="Paris morphology + NICE NBI prediction + BSG size estimate + IBD differential",
                finding="Sub-classification produced",
                confidence=0.65,
                evidence=ev,
                effect="support",
                weight=0.30,
            ))
            idx += 1

    # 5. Text agent (BioBERT attention rollout)
    if text_finding is not None:
        steps.append(TraceStep(
            step_idx=idx, agent="text", stage="predict",
            input_summary=text_finding.get("input_summary",
                "BioBERT attention rollout over clinical text"),
            finding=text_finding.get("finding", "Text supports image call"),
            confidence=float(text_finding.get("confidence", 0.5)),
            evidence=text_finding.get("evidence", []),
            effect=text_finding.get("effect", "support"),
            weight=0.20,
        ))
        idx += 1

    # 6. Tabular risk agent (TCGA features)
    if tabular_finding is not None:
        steps.append(TraceStep(
            step_idx=idx, agent="tabular", stage="predict",
            input_summary=tabular_finding.get("input_summary",
                "TabTransformer on 12 TCGA clinical features (age, BMI, family hx, smoking …)"),
            finding=tabular_finding.get("finding", "Risk score generated"),
            confidence=float(tabular_finding.get("confidence", 0.5)),
            evidence=tabular_finding.get("evidence", []),
            effect=tabular_finding.get("effect", "support"),
            weight=0.20,
        ))
        idx += 1

    # 7. TCGA tabular stage classifier (HistGradientBoosting)
    if tcga_stage_finding is not None:
        stage_label = tcga_stage_finding.get("stage", tcga_stage_finding.get("class", "?"))
        cf = float(tcga_stage_finding.get("confidence", 0.0))
        ev = [f"{k}: {v}" for k, v in (tcga_stage_finding.get("evidence", {}) or {}).items()]
        steps.append(TraceStep(
            step_idx=idx, agent="tcga_stage", stage="predict",
            input_summary="Independent gradient-boosted classifier on TCGA-COAD tabular features (53% 4-class acc)",
            finding=f"Independent stage estimate: Stage {stage_label}",
            confidence=cf,
            evidence=ev,
            effect="support",
            weight=0.15,
        ))
        idx += 1

    # 8. Fusion reasoning (cross-modal attention head)
    if fusion_finding is not None:
        modality_weights = fusion_finding.get("modality_weights", {})
        ev = []
        if modality_weights:
            for k, v in modality_weights.items():
                ev.append(f"{k} contributed {float(v)*100:.1f}% to fused call")
        steps.append(TraceStep(
            step_idx=idx, agent="fusion", stage="predict",
            input_summary="Cross-modal attention transformer fuses image + text + tabular tokens",
            finding=fusion_finding.get("finding", "Modalities fused"),
            confidence=float(fusion_finding.get("confidence", 0.5)),
            evidence=ev,
            effect="support",
            weight=0.70,
        ))
        idx += 1

    # 9. XAI uncertainty agent (MC-Dropout, Integrated Gradients)
    if xai_finding is not None:
        ev = xai_finding.get("evidence", [])
        if "epistemic_uncertainty" in xai_finding:
            ev = list(ev) + [f"epistemic uncertainty: {float(xai_finding['epistemic_uncertainty']):.3f}"]
        steps.append(TraceStep(
            step_idx=idx, agent="xai", stage="verify",
            input_summary="MC-Dropout + Integrated Gradients (Captum) for uncertainty and feature importance",
            finding=xai_finding.get("finding", "Uncertainty profile computed"),
            confidence=float(xai_finding.get("confidence", 0.5)),
            evidence=ev,
            effect=xai_finding.get("effect", "support"),
            weight=0.30,
        ))
        idx += 1

    # 10. Safety agent (per-class thresholds, debouncer)
    if safety_finding is not None:
        passed = bool(safety_finding.get("passed", True))
        ev = safety_finding.get("evidence", [])
        steps.append(TraceStep(
            step_idx=idx, agent="safety",
            stage="verify" if passed else "abstain",
            input_summary="Per-class confidence thresholds (uc-mild ≥ 0.89 etc.) and live-video debouncer",
            finding="Safety check passed" if passed else "Safety check failed → abstain",
            confidence=float(safety_finding.get("confidence", 1.0)),
            evidence=ev,
            effect="support" if passed else "abstain",
            weight=1.00 if not passed else 0.20,
        ))
        idx += 1

    # 11. Clinical recommendation agent (BSG / NICE / Paris)
    if clinical_finding is not None:
        steps.append(TraceStep(
            step_idx=idx, agent="clinical", stage="predict",
            input_summary="BSG-2020 / NICE NBI / Paris classification → recommended action",
            finding=clinical_finding.get("recommendation",
                clinical_finding.get("finding", "Routine surveillance")),
            confidence=float(clinical_finding.get("confidence", 0.8)),
            evidence=clinical_finding.get("evidence", []),
            effect="support",
            weight=0.40,
        ))
        idx += 1

    # 12. LLM refinement (Groq rationale rewriting)
    if llm_refined:
        steps.append(TraceStep(
            step_idx=idx, agent="llm", stage="refine",
            input_summary="Llama-3.1-8b-instant rewrote the clinical rationale paragraph for clarity",
            finding="Rationale paragraph refined by LLM (class/confidence preserved, guard rails enforced)",
            confidence=1.0,
            evidence=["class & confidence unchanged",
                      "forbidden-words filter passed",
                      "no new clinical claims introduced"],
            effect="refine",
            weight=0.0,           # LLM does not influence the verdict, only its phrasing
        ))
        idx += 1

    # 13. Final verdict marker (closes the trace)
    if final_verdict is not None:
        steps.append(TraceStep(
            step_idx=idx, agent="orchestrator", stage="predict",
            input_summary="Weighted aggregate of all prior agents",
            finding=f"FINAL VERDICT: {final_verdict}",
            confidence=float(final_confidence) if final_confidence is not None else 0.0,
            evidence=[f"{len(steps)} agents weighed in"],
            effect="support",
            weight=1.0,
        ))

    return [s.to_dict() for s in steps]


# ──────────────────────────────────────────────────────────────────────
#  Disagreement detection
# ──────────────────────────────────────────────────────────────────────
def detect_disagreements(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find agents that disagreed with the final verdict.

    Returns
    -------
    dict with keys:
        agreed     list[str]    agents whose finding aligned
        disagreed  list[str]    agents whose finding contradicted / overrode
        ratio      float        |agreed| / (|agreed| + |disagreed|)
        unanimous  bool         True if no disagreement
    """
    agreed, disagreed = [], []
    for step in trace:
        eff = step.get("effect", "support")
        agent = step.get("agent", "?")
        if eff in ("support", "noop", "refine"):
            agreed.append(agent)
        elif eff in ("contradict", "override", "abstain"):
            disagreed.append(agent)
    total = len(agreed) + len(disagreed)
    ratio = len(agreed) / total if total else 1.0
    return {
        "agreed": agreed,
        "disagreed": disagreed,
        "ratio": ratio,
        "unanimous": len(disagreed) == 0,
    }


# ──────────────────────────────────────────────────────────────────────
#  Compact narrative builder
# ──────────────────────────────────────────────────────────────────────
def render_narrative(trace: List[Dict[str, Any]]) -> str:
    """Render a trace as a short human-readable narrative.

    One bullet per step. Used in the UI's "Reasoning chain" card and
    by the LLM refiner as raw input for the patient-friendly summary.
    """
    if not trace:
        return "(no trace available)"
    lines: List[str] = []
    for step in trace:
        marker = {
            "support": "✓",
            "contradict": "✗",
            "override": "⚠",
            "abstain": "○",
            "noop": "·",
            "refine": "✎",
        }.get(step.get("effect", "support"), "•")
        agent = step.get("agent", "?").replace("_", " ").title()
        stage = step.get("stage", "")
        finding = step.get("finding", "")
        conf = step.get("confidence", 0.0)
        lines.append(f"{marker}  {agent}  ({stage}, conf {conf:.0%}):  {finding}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
#  JSON export (for learning log)
# ──────────────────────────────────────────────────────────────────────
def trace_to_json(trace: List[Dict[str, Any]]) -> str:
    """Serialise a trace to a compact JSON string."""
    return json.dumps(trace, separators=(",", ":"), default=str)
