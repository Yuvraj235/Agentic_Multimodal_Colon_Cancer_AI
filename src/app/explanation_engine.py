"""ColonAI — Unified Explanation Engine.

Stitches the outputs of every other explainability module into a single
coherent report:

    DECISION TRACE                  →  src/app/decision_trace.py
    MODALITY ATTRIBUTION            →  src/app/modality_attribution.py
    COUNTERFACTUAL SCENARIOS        →  src/app/counterfactual.py
    PROTOTYPE RETRIEVAL             →  src/app/prototype_retrieval.py
    SMART RATIONALE (per-image)     →  src/app/smart_rationale.py
    POLYP SUB-TYPING                →  src/app/polyp_typing.py
    ATYPICALITY OVERRIDE            →  src/app/image_atypicality.py
    INVARIANCE / STABILITY          →  src/app/counterfactual.stability_score

Two outputs:

1. ``narrative()``  →  ONE plain-English paragraph (≤120 words) that a
   patient or junior doctor can read. Designed to be safely refined by
   the LLM (src/app/llm_refine.py) without losing clinical content.

2. ``clinician_report()``  →  STRUCTURED dossier with sections
   (Verdict / Evidence / Caveats / Counterfactuals / Similar cases /
   Recommended action). Designed for a senior clinician.

Both outputs are deterministic — same inputs → same words — so they can
be reproduced for audit.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import textwrap


# ──────────────────────────────────────────────────────────────────────
#  Patient-friendly narrative
# ──────────────────────────────────────────────────────────────────────
def narrative(
    *,
    final_class: str,
    final_confidence: float,
    smart_pred: Optional[Any] = None,
    attribution: Optional[Dict[str, Any]] = None,
    disagreement: Optional[Dict[str, Any]] = None,
    stability: Optional[Dict[str, Any]] = None,
    atypicality_fired: bool = False,
    neighbour_concordance: Optional[Dict[str, Any]] = None,
    tcga_stage: Optional[Dict[str, Any]] = None,
    max_words: int = 130,
) -> str:
    """Generate the patient-friendly explanation paragraph."""
    parts: List[str] = []

    # 1. Headline call
    nice_name = _nicen(final_class)
    conf_pct = f"{float(final_confidence) * 100:.0f}%"
    parts.append(f"The system reports **{nice_name}** with confidence {conf_pct}.")

    # 2. If atypicality override fired — surface that loudly
    if atypicality_fired:
        parts.append("⚠ An invasive-lesion pattern (deep ulceration, bleeding, nodular surface or mass reflection) was detected, so the call was upgraded to urgent endoscopist review regardless of the routine classifier.")

    # 3. Smart-inference hedging
    if smart_pred is not None:
        is_hedged = getattr(smart_pred, "is_hedged", False) or (
            isinstance(smart_pred, dict) and smart_pred.get("is_hedged", False))
        if is_hedged:
            parts.append("The 50-pass ensemble (TTA + MC-Dropout) showed enough uncertainty that the system hedged — please treat as a *suggestion*, not a definitive call.")

    # 4. Modality attribution
    if attribution and attribution.get("interpretable"):
        contribs = attribution.get("contributions", {})
        if contribs:
            top = max(contribs.items(), key=lambda kv: kv[1])
            mod_name = {"image": "endoscopic image",
                        "text": "clinical text",
                        "tabular": "tabular features"}.get(top[0], top[0])
            parts.append(f"Most of the decision (~{top[1]:.0f}%) came from the {mod_name}.")

    # 5. Disagreement
    if disagreement and not disagreement.get("unanimous", True):
        dis = disagreement.get("disagreed", [])
        parts.append(f"{len(dis)} agent(s) raised a concern: " +
                     ", ".join(_nicen(a) for a in dis) + " — read the trace below for detail.")

    # 6. Stability
    if stability and stability.get("total", 0) > 0:
        verdict = stability.get("verdict", "")
        parts.append(f"Under perturbations of brightness, blur and modality silencing the prediction was {verdict}.")

    # 7. Neighbour concordance
    if neighbour_concordance and neighbour_concordance.get("k", 0) > 0:
        conc = neighbour_concordance.get("concordance", 0) * 100
        if conc >= 80:
            parts.append(f"{conc:.0f}% of the most-similar training cases were also {nice_name}, which supports the call.")
        elif conc >= 50:
            parts.append(f"{conc:.0f}% of similar training cases were {nice_name} — moderate independent support.")
        else:
            parts.append(f"Only {conc:.0f}% of similar training cases were {nice_name} — review carefully.")

    # 8. TCGA stage (always shown as SECONDARY estimate)
    if tcga_stage and tcga_stage.get("stage") not in (None, "?"):
        st_conf = float(tcga_stage.get("confidence", 0.0)) * 100
        parts.append(f"An independent TCGA tabular model estimates Stage {tcga_stage['stage']} ({st_conf:.0f}% — population-level only, not a per-image stage call).")

    # Glue + trim
    text = " ".join(parts)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + " …"
    return text


# ──────────────────────────────────────────────────────────────────────
#  Structured clinician report
# ──────────────────────────────────────────────────────────────────────
def clinician_report(
    *,
    final_class: str,
    final_confidence: float,
    trace: List[Dict[str, Any]],
    attribution: Optional[Dict[str, Any]] = None,
    counterfactuals: Optional[List[Dict[str, Any]]] = None,
    stability: Optional[Dict[str, Any]] = None,
    neighbours: Optional[List[Dict[str, Any]]] = None,
    neighbour_concordance: Optional[Dict[str, Any]] = None,
    polyp_typing: Optional[Dict[str, Any]] = None,
    tcga_stage: Optional[Dict[str, Any]] = None,
    smart_rationale_text: Optional[str] = None,
    clinical_recommendation: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a structured dossier the UI can render section-by-section."""

    sections: List[Dict[str, Any]] = []

    # ── 1. Verdict
    sections.append({
        "title":  "1. Final verdict",
        "body":   f"{_nicen(final_class)} (confidence {final_confidence*100:.1f}%)",
        "kind":   "headline",
    })

    # ── 2. Evidence (smart rationale)
    if smart_rationale_text:
        sections.append({
            "title":  "2. Per-image evidence",
            "body":   smart_rationale_text,
            "kind":   "rationale",
        })

    # ── 3. Modality attribution
    if attribution and attribution.get("interpretable"):
        contribs = attribution.get("contributions", {})
        sections.append({
            "title":  "3. Which input modality drove the call?",
            "body":   _format_attribution(contribs, method=attribution.get("method", "?")),
            "kind":   "attribution",
            "data":   contribs,
        })

    # ── 4. Sub-typing
    if polyp_typing:
        sections.append({
            "title":  "4. Sub-classification (Paris / NICE / BSG)",
            "body":   _format_sub_typing(polyp_typing),
            "kind":   "subtyping",
            "data":   polyp_typing,
        })

    # ── 5. Reasoning chain
    if trace:
        from src.app.decision_trace import render_narrative
        sections.append({
            "title":  "5. Step-by-step reasoning chain",
            "body":   render_narrative(trace),
            "kind":   "trace",
            "data":   trace,
        })

    # ── 6. Counterfactuals
    if counterfactuals:
        sections.append({
            "title":  "6. What would change if…?",
            "body":   _format_counterfactuals(counterfactuals),
            "kind":   "counterfactual",
            "data":   counterfactuals,
        })

    # ── 7. Stability
    if stability and stability.get("total", 0) > 0:
        sections.append({
            "title":  "7. Stability under perturbations",
            "body":   (f"{stability['flips']}/{stability['total']} perturbations flipped the prediction. "
                       f"Stability index = {stability['stability']*100:.0f}%. "
                       f"Verdict: {stability['verdict']}."),
            "kind":   "stability",
            "data":   stability,
        })

    # ── 8. Prototype neighbours
    if neighbours:
        body = "Most-similar training cases (cosine similarity in fused-embedding space):\n"
        for n in neighbours:
            body += f"  • rank {n['rank']}: {_nicen(n['label'])}  (similarity {n['similarity']:.3f})\n"
        if neighbour_concordance:
            agreed = "✓" if neighbour_concordance.get("agrees") else "✗"
            body += (f"\n{agreed} {neighbour_concordance['concordance']*100:.0f}% concordance "
                     f"with model prediction.")
        sections.append({
            "title":  "8. Similar training cases (case-based reasoning)",
            "body":   body,
            "kind":   "neighbours",
            "data":   neighbours,
        })

    # ── 9. TCGA tabular stage (always SECONDARY)
    if tcga_stage and tcga_stage.get("stage") not in (None, "?"):
        sections.append({
            "title":  "9. Independent tabular stage estimate (TCGA)",
            "body":   (f"Stage {tcga_stage['stage']} (confidence {float(tcga_stage.get('confidence', 0))*100:.0f}%). "
                       f"Note: this is a population-level estimate from clinical features (53% 4-class accuracy on cross-validation), "
                       f"NOT a per-image stage call. See docs/STAGING_ROADMAP.md."),
            "kind":   "stage",
            "data":   tcga_stage,
        })

    # ── 10. Recommendation
    if clinical_recommendation:
        sections.append({
            "title":  "10. Recommended next step",
            "body":   clinical_recommendation,
            "kind":   "recommendation",
        })

    # ── 11. Caveats (always)
    sections.append({
        "title":  "11. Caveats",
        "body":   ("• This system is a decision-support tool, not a diagnosis. "
                   "Biopsy remains the gold standard.\n"
                   "• Training data: HyperKvasir + CVC-ClinicDB + TCGA-COAD; "
                   "on a truly held-out (different-vendor) scope, segmentation "
                   "localisation is ~0.45 IoU — lower than on familiar scopes.\n"
                   "• Cancer stage is computed exactly from doctor-entered TNM "
                   "(AJCC 8th ed.); from an image alone it is only a preliminary "
                   "invasion-depth impression, pending biopsy. See docs/STAGING_ROADMAP.md."),
        "kind":   "caveats",
    })

    return {
        "final_class":       final_class,
        "final_confidence":  float(final_confidence),
        "sections":          sections,
    }


# ──────────────────────────────────────────────────────────────────────
#  Internal formatters
# ──────────────────────────────────────────────────────────────────────
_NICE_NAMES = {
    "polyps":          "Polyps",
    "uc-mild":         "Ulcerative colitis (mild)",
    "uc-mod-sev":      "Ulcerative colitis (moderate–severe)",
    "barretts":        "Barrett's oesophagus",
    "therapeutic":     "Post-therapeutic site",
    "image": "Image",
    "text": "Clinical text",
    "tabular": "Tabular features",
    "fusion": "Multi-modal fusion",
    "xai": "Uncertainty/XAI",
    "clinical": "Clinical-guidelines",
    "safety": "Safety policy",
    "atypicality": "Atypicality detector",
    "polyp_typing": "Polyp sub-typing",
    "tcga_stage": "TCGA stage",
    "smart_inference": "Smart-inference ensemble",
    "llm": "LLM refiner",
    "orchestrator": "Orchestrator",
}


def _nicen(s: str) -> str:
    return _NICE_NAMES.get(s, s.replace("_", " ").title())


def _format_attribution(contribs: Dict[str, float], *, method: str) -> str:
    if not contribs:
        return "(per-modality attribution unavailable)"
    rows = sorted(contribs.items(), key=lambda kv: -kv[1])
    out = []
    for k, v in rows:
        bar_cells = max(1, int(round(v / 100.0 * 30)))
        out.append(f"  {_nicen(k):<22} | {'█' * bar_cells} {v:5.1f}%")
    return "Per-modality contribution (estimated by " + method + "):\n" + "\n".join(out)


def _format_counterfactuals(cfs: List[Dict[str, Any]]) -> str:
    if not cfs:
        return "(no counterfactuals computed)"
    out = ["Scenario                                | New class           | Δ conf  | Flipped?"]
    out.append("-" * 90)
    for cf in cfs:
        sc = (cf.get("scenario", "?"))[:40]
        nc = (cf.get("new_class", "?"))[:18]
        dc = cf.get("delta_conf", 0.0) * 100
        fl = "YES" if cf.get("flipped") else "no"
        out.append(f"{sc:<40} | {nc:<18}  | {dc:+5.1f}% | {fl}")
    return "\n".join(out)


def _format_sub_typing(typing: Dict[str, Any]) -> str:
    out: List[str] = []
    label_map = {
        "paris":              "Paris morphology",
        "nice":               "NICE NBI prediction",
        "size_mm":            "Estimated lesion size",
        "size_bucket":        "BSG size bucket",
        "ibd_differential":   "IBD differential",
        "diverticulosis":     "Diverticulosis pattern",
        "hemorrhoid":         "Hemorrhoid features",
    }
    for k, v in typing.items():
        label = label_map.get(k, k)
        if isinstance(v, dict):
            sub = v.get("class") or v.get("type") or v.get("label") or v.get("verdict")
            cf = v.get("confidence") or v.get("score") or v.get("prob")
            if sub is None and "_" in str(v):
                sub = str(v)
            row = f"  • {label}: {sub}"
            if cf is not None:
                row += f" ({float(cf)*100:.0f}%)"
        elif isinstance(v, (int, float)):
            row = f"  • {label}: {v}"
        elif isinstance(v, str):
            row = f"  • {label}: {v}"
        else:
            continue
        out.append(row)
    return "\n".join(out) if out else "(sub-typing unavailable)"


# ──────────────────────────────────────────────────────────────────────
#  Public render helper for plain-text export (markdown body)
# ──────────────────────────────────────────────────────────────────────
def report_to_markdown(report: Dict[str, Any]) -> str:
    """Render the structured report as markdown — for the PDF and the
    'copy explanation to clipboard' button.
    """
    lines = [f"# ColonAI explanation report",
             f"",
             f"**Final verdict:** {_nicen(report['final_class'])}  "
             f"(confidence {report['final_confidence']*100:.1f}%)",
             ""]
    for section in report.get("sections", []):
        lines.append(f"## {section['title']}")
        lines.append("")
        body = section.get("body", "")
        # Indent any monospace blocks to keep markdown happy
        if section.get("kind") in ("attribution", "counterfactual", "trace", "neighbours", "subtyping"):
            lines.append("```")
            lines.append(body)
            lines.append("```")
        else:
            lines.append(body)
        lines.append("")
    return "\n".join(lines)
