"""ColonAI — clinician structured report (the doctor's 5-point summary).

Assembles the five data points a clinician asked for — **Size, Number, Location,
Stage, Treatment** — from the analysis the app already produced, plus any values
the doctor entered. The hard rule (see docs/CAPABILITY_INVENTORY.md + the plan):

  EVERY field is tagged with its `source`, and NOTHING is fabricated:
    measured       — a real measurement from the image/segmentation
    estimated      — a heuristic estimate, explicitly not exact (e.g. size in mm)
    doctor-entered — supplied by the clinician (location, T/N/M)
    computed       — a deterministic rule applied to inputs (AJCC stage from TNM)
    guideline      — a cited guideline next-step (NOT a treatment prescription)
    unavailable    — we genuinely cannot provide it from this case

It also surfaces the existing safety guardrail as `requires_human_review` so the
dashboard can show "Suspicious / Requires human review" instead of a false clear.

Pure + dependency-light: takes the analysis result dict (see page_analysis `out`)
and an optional doctor_inputs dict; returns a JSON-serialisable schema.
"""
from __future__ import annotations
from typing import Dict, Optional, Any
import numpy as np

POLYP_CLASSES = ("polyps", "therapeutic")


def _field(value: Any, source: str, *, confidence: Optional[float] = None,
           detail: Any = None, caveat: Optional[str] = None) -> Dict:
    """One structured field. `source` ∈ measured|estimated|doctor-entered|
    computed|guideline|unavailable."""
    return {"value": value, "source": source, "confidence": confidence,
            "detail": detail, "caveat": caveat}


def _count_polyps(seg_mask) -> Dict:
    """Count distinct polyp regions from the segmentation mask (a real
    measurement). Returns a `_field`. Never invents a number."""
    if seg_mask is None:
        return _field(None, "unavailable",
                      caveat="no segmentation available to count from")
    try:
        m = np.asarray(seg_mask)
        binm = (m > 0.5).astype(np.uint8)
        if binm.sum() == 0:
            return _field(0, "measured", detail="no polyp region segmented")
        min_area = max(20, int(0.0008 * binm.size))
        try:
            import cv2
            n, _, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
            count = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)
        except Exception:
            from scipy import ndimage
            lab, n = ndimage.label(binm)
            count = sum(1 for i in range(1, n + 1) if (lab == i).sum() >= min_area)
        return _field(max(count, 1) if binm.sum() else 0, "measured",
                      detail="distinct regions in the segmentation mask")
    except Exception:
        return _field(None, "unavailable", caveat="could not parse segmentation")


def build_structured_report(analysis: Dict,
                            doctor_inputs: Optional[Dict] = None) -> Dict:
    """Build the 5-point clinician summary from an analysis result dict.

    doctor_inputs (optional): {location, T, N, M, histology_grade}.
    """
    analysis = analysis or {}
    di = doctor_inputs or {}
    pclass = analysis.get("pathology_class") or analysis.get("predicted_class") or "unknown"
    is_polyp = pclass in POLYP_CLASSES
    sub = analysis.get("sub_typing") or {}
    safety = analysis.get("safety_verdict") or {}

    fields: Dict[str, Dict] = {}

    # ── 1) NUMBER — measured from segmentation ──────────────────────────
    if is_polyp:
        fields["number"] = _count_polyps(analysis.get("seg_mask"))
    else:
        fields["number"] = _field(0, "measured",
                                  detail=f"primary finding is '{pclass}', not a polyp")

    # ── 2) SIZE — ESTIMATE only (never claimed exact) ───────────────────
    size = (sub.get("size") or {}) if isinstance(sub, dict) else {}
    if is_polyp and size.get("size_mm") is not None:
        fields["size"] = _field(
            size.get("size_category"), "estimated",
            detail=f"~{size.get('size_mm')} mm",
            caveat="ESTIMATED from the segmentation assuming a typical ~30 mm "
                   "field of view — NOT a calibrated measurement. Confirm endoscopically.")
    else:
        fields["size"] = _field(None, "unavailable",
                                caveat="no lesion mask to estimate size from")

    # ── 3) LOCATION — doctor-entered (not derivable from one image) ─────
    loc = di.get("location")
    fields["location"] = _field(
        loc or None, "doctor-entered",
        caveat=None if loc else "precise anatomical location needs the scope "
               "position — please enter it (the image alone cannot give it).")

    # ── 4) STAGE — computed from doctor's TNM (AJCC), never from the image ─
    t, n, m = di.get("T"), di.get("N"), di.get("M")
    if t and n and m:
        try:
            from src.app.staging import ajcc_colorectal_stage
            res = ajcc_colorectal_stage(t, n, m)   # expects full cats e.g. T3/N1/M0
            grp = res.get("stage_group")
            if grp and grp != "?":
                fields["stage"] = _field(
                    f"Stage {grp}", "computed",
                    detail=f"AJCC 8th ed. ({res.get('rationale', '').strip()})")
            else:
                fields["stage"] = _field(
                    None, "unavailable",
                    caveat="TNM combination not recognised — needs expert review.")
        except Exception as e:
            fields["stage"] = _field(None, "unavailable",
                                     caveat=f"staging error: {type(e).__name__}")
    else:
        fields["stage"] = _field(
            None, "doctor-entered",
            caveat="enter T, N and M (from biopsy + imaging) to compute the exact "
                   "AJCC stage. Stage cannot be read from a colonoscopy image.")

    # ── 5) TREATMENT — guideline next-steps, NOT a prescription ─────────
    rec = analysis.get("recommendation")
    rx_text = None
    if isinstance(rec, dict):
        rx_text = (rec.get("primary_action") or rec.get("urgency")
                   or rec.get("full_report"))
    elif isinstance(rec, str):
        rx_text = rec
    fields["treatment"] = _field(
        rx_text, "guideline" if rx_text else "unavailable",
        caveat="suggested NEXT STEPS aligned to guidelines (e.g. resect/biopsy/"
               "refer/surveillance) — this is decision support, NOT a treatment "
               "prescription. The treating clinician decides therapy.")

    # ── Safety guardrail (sensitivity-first) ────────────────────────────
    action = safety.get("action", "show")
    requires_review = action != "show"
    disclaimer = safety.get("disclaimer") or (
        "AI screening decision-support only — confirm every finding with a "
        "qualified clinician before any clinical decision.")

    return {
        "schema_version": "1.0",
        "primary_finding": pclass,
        "fields": fields,
        "requires_human_review": bool(requires_review),
        "review_reason": safety.get("reason", "") if requires_review else "",
        "safety_action": action,
        "disclaimer": disclaimer,
    }


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for src.app imports
    # synthetic analysis: a polyp with a seg mask, no doctor TNM
    seg = np.zeros((224, 224), dtype=np.float32); seg[60:140, 70:150] = 0.9
    demo = {
        "pathology_class": "polyps", "confidence": 0.91, "uncertainty": 0.08,
        "seg_mask": seg,
        "sub_typing": {"size": {"size_mm": 12.4, "size_category": "large (10-19 mm)"}},
        "recommendation": {"primary_action": "Endoscopic mucosal resection; send histology."},
        "safety_verdict": {"action": "show", "reason": "All safety checks passed.",
                           "disclaimer": "AI decision-support only."},
    }
    r = build_structured_report(demo, doctor_inputs={"location": "sigmoid colon"})
    print(json.dumps(r, indent=2, default=str))
    # assertions: honesty contract
    assert r["fields"]["size"]["source"] == "estimated"
    assert r["fields"]["location"]["value"] == "sigmoid colon"
    assert r["fields"]["stage"]["source"] == "doctor-entered"  # no TNM given
    assert r["fields"]["number"]["source"] == "measured"
    assert r["requires_human_review"] is False
    # with TNM → computed stage (full AJCC categories)
    r2 = build_structured_report(demo, doctor_inputs={"T": "T3", "N": "N1", "M": "M0"})
    assert r2["fields"]["stage"]["source"] == "computed", r2["fields"]["stage"]
    assert "III" in (r2["fields"]["stage"]["value"] or ""), r2["fields"]["stage"]
    # low-confidence abstain → requires review
    demo_abstain = {**demo, "safety_verdict": {"action": "abstain", "reason": "low confidence"}}
    assert build_structured_report(demo_abstain)["requires_human_review"] is True
    print("OK — structured_report honesty contract holds")
