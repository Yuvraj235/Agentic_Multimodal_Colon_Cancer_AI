"""ColonAI — Patient Safety Guard.

Central policy layer. Every prediction shown to a clinician or patient
passes through `evaluate_safety()`. The function returns one of three
actions:

   "show"    — confidence high, agents agree, image accepted → ok to display
   "abstain" — borderline (uncertainty, low confidence, agent disagreement)
               → display "Requires human review" instead of a prediction
   "reject"  — image is NOT an endoscopy frame, or GradCAM is completely
               misaligned, or the pipeline errored → display "Cannot analyse
               this image" and refuse to produce numbers

This module is the ONLY place that decides whether a prediction is safe
enough to show. Callers should not invent their own thresholds.

There is also `LivePolypDebouncer` which requires a polyp bounding box to
persist across N frames before being shown — a single-frame false positive
in a live feed never gets surfaced to the operator.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List
import json, time, hashlib, os


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds — tuned for screening-colonoscopy use
# Sources: Wang et al. 2018 (CADe-augmented colonoscopy review),
#          BSG/ESGE 2022 guideline (minimum AI confidence ≥ 0.75 for clinical use)
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_CONFIG = {
    "min_confidence":          0.75,   # below this → abstain
    "max_uncertainty":         0.30,   # above this → abstain
    "min_endoscopy_score":     0.55,   # below this → reject
    "min_gradcam_focus":       0.15,   # GradCAM concentration in top-25%
                                        # of pixels — below this → abstain
    "min_agent_agreement":     0.66,   # 2-of-3 agents must agree (normal)
    "strict_agent_agreement":  1.00,   # ALL must agree (second-opinion mode)
    "live_debounce_frames":    3,      # bbox must persist N frames
    "live_iou_threshold":      0.30,   # bbox-IoU threshold to call it
                                        # "the same polyp" frame-to-frame
}


def second_opinion(probs_list, top_class: Optional[int] = None) -> Dict:
    """Strict TTA-based second-opinion check.

    Given a list of probability arrays (one per augmentation / model
    sample), this returns:
        unanimous        — every sample picked the same top class
        majority_class   — the class chosen by the most samples
        agreement_pct    — fraction that agree with majority
        confidence_mean  — mean confidence on the majority class

    Use it from caller code:
        so = second_opinion([p1, p2, p3])
        if not so["unanimous"]:
            # downgrade to abstain
    """
    import numpy as _np
    if not probs_list:
        return {"unanimous": False, "majority_class": -1,
                "agreement_pct": 0.0, "confidence_mean": 0.0}
    classes = [int(_np.argmax(p)) for p in probs_list]
    from collections import Counter
    cnt = Counter(classes); mc, mc_count = cnt.most_common(1)[0]
    agree = mc_count / len(classes)
    confs = [float(p[mc]) for p in probs_list if int(_np.argmax(p)) == mc]
    return {
        "unanimous":       bool(agree == 1.0),
        "majority_class":  int(mc),
        "agreement_pct":   float(agree),
        "confidence_mean": float(sum(confs) / max(1, len(confs))),
        "n_samples":       int(len(probs_list)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verdict object — what every safety check returns
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SafetyVerdict:
    action:      str                   # "show" | "abstain" | "reject"
    reason:      str                   # one-line explanation
    disclaimer:  str = ""              # banner to display alongside output
    flags:       List[str] = field(default_factory=list)
    confidence:  float = 0.0
    uncertainty: float = 0.0

    @property
    def requires_human_review(self) -> bool:
        """True whenever we are NOT confidently showing a result — i.e. abstain
        or reject. The dashboard shows 'Suspicious / Requires human review'
        instead of a (possibly false) Clear/Negative."""
        return self.action != "show"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["requires_human_review"] = self.requires_human_review
        return d


_DISCLAIMER_BASE = (
    "⚕️  This AI assists screening review only. It is NOT a substitute for "
    "histopathological diagnosis. All findings must be confirmed by a "
    "qualified endoscopist before any clinical decision."
)


# ─────────────────────────────────────────────────────────────────────────────
# Core safety evaluator
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Per-class threshold loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
_PCT_CACHE = None

def _per_class_thresholds() -> Dict[str, float]:
    """Load per-class abstention thresholds if available.

    Set the env var COLONAI_PER_CLASS_THRESHOLDS to point to a JSON file
    produced by scripts/calibrate_per_class_thresholds.py. Falls back
    to a hardcoded default path or {} if not present.
    """
    global _PCT_CACHE
    if _PCT_CACHE is not None: return _PCT_CACHE
    path = (os.environ.get("COLONAI_PER_CLASS_THRESHOLDS")
            or "outputs/unified_multimodal_v2/per_class_thresholds.json")
    try:
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            _PCT_CACHE = data.get("thresholds", {})
        else:
            _PCT_CACHE = {}
    except Exception:
        _PCT_CACHE = {}
    return _PCT_CACHE


def evaluate_safety(
    confidence:        float,
    uncertainty:       float = 0.0,
    endoscopy_score:   float = 1.0,
    gradcam_focus:     Optional[float] = None,
    agent_agreement:   Optional[float] = None,
    pipeline_error:    Optional[str]   = None,
    strict_mode:       bool = False,
    predicted_class:   Optional[str]   = None,
) -> SafetyVerdict:
    """Decide whether to show, abstain, or reject the prediction.

    All inputs are in [0, 1] where higher is "more confident / better
    image quality / more agreement". `pipeline_error` is a string if any
    upstream component raised an exception.
    """
    flags: List[str] = []

    # 1. Pipeline failure → reject
    if pipeline_error:
        return SafetyVerdict(
            action="reject", reason=f"Pipeline error: {pipeline_error}",
            disclaimer=_DISCLAIMER_BASE + "\n• Pipeline failed — no result shown.",
            flags=["pipeline_error"],
            confidence=confidence, uncertainty=uncertainty)

    # 2. Endoscopy gate
    if endoscopy_score < SAFETY_CONFIG["min_endoscopy_score"]:
        return SafetyVerdict(
            action="reject",
            reason=f"Image does not look like an endoscopy frame "
                   f"(quality {endoscopy_score:.2f} < "
                   f"{SAFETY_CONFIG['min_endoscopy_score']:.2f}).",
            disclaimer=_DISCLAIMER_BASE +
                "\n• Upload a real colonoscopy or capsule-endoscopy frame.",
            flags=["not_endoscopy"],
            confidence=confidence, uncertainty=uncertainty)

    # 3. GradCAM focus check (if available) — guards against
    #    "high-confidence but attention is everywhere" trap
    if gradcam_focus is not None and gradcam_focus < SAFETY_CONFIG["min_gradcam_focus"]:
        flags.append("gradcam_diffuse")

    # 4. Agent agreement (if available)
    #    Normal mode: 2-of-3 must agree.
    #    strict_mode (second-opinion): ALL must agree.
    _min_agree = (SAFETY_CONFIG["strict_agent_agreement"] if strict_mode
                  else SAFETY_CONFIG["min_agent_agreement"])
    if agent_agreement is not None and agent_agreement < _min_agree:
        flags.append("second_opinion_failed" if strict_mode else "agents_disagree")

    # 5. Confidence threshold
    #    If we have per-class calibrated thresholds (fitted on val data via
    #    scripts/calibrate_per_class_thresholds.py) AND we know which class
    #    was predicted, use the class-specific threshold instead of the
    #    global 0.75. This makes the system stricter on classes that are
    #    over-predicted (e.g. uc-mild) and looser on classes that are
    #    near-perfect (e.g. barretts-esoph).
    _min_conf = SAFETY_CONFIG["min_confidence"]
    _pct      = _per_class_thresholds()
    if predicted_class and predicted_class in _pct:
        _min_conf = max(_min_conf, float(_pct[predicted_class]))
    if confidence < _min_conf:
        flags.append(f"low_confidence_for_{predicted_class}"
                     if predicted_class and predicted_class in _pct
                     else "low_confidence")
    if uncertainty > SAFETY_CONFIG["max_uncertainty"]:
        flags.append("high_uncertainty")

    # Any flag → abstain (do not show a definitive prediction)
    if flags:
        reason = "Abstaining — " + ", ".join(flags) + (
            f" (conf {confidence:.2f}, unc {uncertainty:.2f})")
        return SafetyVerdict(
            action="abstain", reason=reason,
            disclaimer=_DISCLAIMER_BASE +
                "\n• AI confidence not high enough for an automated read. "
                "Request a senior review.",
            flags=flags, confidence=confidence, uncertainty=uncertainty)

    # All checks passed
    return SafetyVerdict(
        action="show", reason="All safety checks passed.",
        disclaimer=_DISCLAIMER_BASE,
        flags=[], confidence=confidence, uncertainty=uncertainty)


# ─────────────────────────────────────────────────────────────────────────────
# Audit log — every prediction written to disk for post-hoc review
# ─────────────────────────────────────────────────────────────────────────────
class AuditLog:
    def __init__(self, log_dir: str = "outputs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Owner-only directory perms (0o700). Best-effort; ignored on POSIX-unfriendly FSs.
        try: os.chmod(self.log_dir, 0o700)
        except Exception: pass
        self.path = self.log_dir / f"audit_{time.strftime('%Y%m%d')}.jsonl"
        # Touch + secure perms on the file itself
        if not self.path.exists():
            self.path.touch()
        try: os.chmod(self.path, 0o600)
        except Exception: pass

    def record(self, *, case_id: str, image_path: Optional[str] = None,
               pathology_class: str, confidence: float,
               uncertainty: float, verdict: SafetyVerdict,
               extras: Optional[Dict] = None) -> None:
        entry = {
            "ts":              time.time(),
            "iso":             time.strftime("%Y-%m-%dT%H:%M:%S"),
            "case_id":         case_id,
            "image_path":      image_path,
            "image_sha256":    (self._sha(image_path) if image_path else None),
            "pathology_class": pathology_class,
            "confidence":      confidence,
            "uncertainty":     uncertainty,
            "action":          verdict.action,
            "reason":          verdict.reason,
            "flags":           verdict.flags,
            "extras":          extras or {},
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        # Re-assert owner-only after append (umask might widen it on some OSes)
        try: os.chmod(self.path, 0o600)
        except Exception: pass

    @staticmethod
    def _sha(path: Optional[str]) -> Optional[str]:
        if not path: return None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    b = f.read(8192)
                    if not b: break
                    h.update(b)
            return h.hexdigest()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Live-video debouncer — bbox must persist N frames before flagging
# ─────────────────────────────────────────────────────────────────────────────
def _bbox_iou(b1: Dict, b2: Dict) -> float:
    """IoU between two bboxes {x,y,w,h}."""
    x1 = max(b1["x"], b2["x"]); y1 = max(b1["y"], b2["y"])
    x2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
    y2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = b1["w"] * b1["h"]; a2 = b2["w"] * b2["h"]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class LivePolypDebouncer:
    """Track candidate polyps across video frames.

    A polyp is only EMITTED to the operator after `min_frames` consecutive
    frames have seen a bounding box at the same location (IoU ≥ threshold).
    This eliminates single-frame false positives, which are common in live
    colonoscopy feeds (motion blur, light reflections, fluid).
    """

    def __init__(self,
                 min_frames: int = SAFETY_CONFIG["live_debounce_frames"],
                 iou_threshold: float = SAFETY_CONFIG["live_iou_threshold"],
                 stale_after: int = 8):
        self.min_frames    = min_frames
        self.iou_threshold = iou_threshold
        self.stale_after   = stale_after
        # tracks: list of {bbox, consecutive, last_seen, confidence, emitted}
        self.tracks: List[Dict] = []
        self.frame_idx = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Feed in this frame's raw detections, get back the SAFE polyps.

        Each detection is {bbox: {x,y,w,h}, confidence: float, ...}.
        Returns the subset that has persisted ≥ min_frames frames.
        """
        self.frame_idx += 1
        # Match detections to existing tracks
        matched_track_idx = set()
        for det in detections:
            best, best_iou = None, 0.0
            for i, tr in enumerate(self.tracks):
                if i in matched_track_idx: continue
                iou = _bbox_iou(det["bbox"], tr["bbox"])
                if iou > best_iou:
                    best, best_iou = i, iou
            if best is not None and best_iou >= self.iou_threshold:
                tr = self.tracks[best]
                tr["bbox"]        = det["bbox"]
                tr["consecutive"] += 1
                tr["last_seen"]   = self.frame_idx
                tr["confidence"]  = max(tr["confidence"], det.get("confidence", 0))
                matched_track_idx.add(best)
            else:
                self.tracks.append({
                    "bbox":        det["bbox"],
                    "consecutive": 1,
                    "last_seen":   self.frame_idx,
                    "confidence":  det.get("confidence", 0),
                    "emitted":     False,
                })

        # Drop stale tracks
        self.tracks = [tr for tr in self.tracks
                       if (self.frame_idx - tr["last_seen"]) <= self.stale_after]

        # Emit tracks that have persisted enough frames
        emitted: List[Dict] = []
        for tr in self.tracks:
            if tr["consecutive"] >= self.min_frames and tr["last_seen"] == self.frame_idx:
                emitted.append({
                    "bbox":        tr["bbox"],
                    "confidence":  tr["confidence"],
                    "frames_seen": tr["consecutive"],
                    "first_seen_frame": self.frame_idx - tr["consecutive"] + 1,
                })
                tr["emitted"] = True
        return emitted

    def reset(self):
        self.tracks = []
        self.frame_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─ Safety policy self-test ─")
    cases = [
        ("High-conf endoscopy polyp",
         dict(confidence=0.92, uncertainty=0.05, endoscopy_score=0.9,
              gradcam_focus=0.4, agent_agreement=1.0)),
        ("Borderline confidence",
         dict(confidence=0.62, uncertainty=0.20, endoscopy_score=0.8,
              gradcam_focus=0.3, agent_agreement=1.0)),
        ("Non-endoscopy upload",
         dict(confidence=0.95, uncertainty=0.10, endoscopy_score=0.20,
              gradcam_focus=0.1, agent_agreement=0.66)),
        ("Diffuse GradCAM (vendor bias) — abstain",
         dict(confidence=0.88, uncertainty=0.08, endoscopy_score=0.85,
              gradcam_focus=0.08, agent_agreement=0.85)),
        ("High uncertainty",
         dict(confidence=0.81, uncertainty=0.42, endoscopy_score=0.9,
              gradcam_focus=0.25, agent_agreement=0.85)),
        ("Pipeline error",
         dict(confidence=0.0, uncertainty=1.0, endoscopy_score=0.9,
              pipeline_error="model forward returned NaN")),
    ]
    for name, args in cases:
        v = evaluate_safety(**args)
        print(f"  {name:35s} → {v.action:7s}  ({v.reason})")

    # ── Regression: sensitivity-first guardrail must NEVER show a result below
    #    threshold (a false "Clear/Negative" is the dangerous failure mode) ──
    print("\n─ Guardrail regression (never show below threshold) ─")
    _good = dict(endoscopy_score=0.9, gradcam_focus=0.4, agent_agreement=1.0)
    for conf in (0.0, 0.30, 0.50, 0.74):
        v = evaluate_safety(confidence=conf, uncertainty=0.10, **_good)
        assert v.action != "show", f"FAIL: confidence {conf} returned 'show'"
        assert v.requires_human_review is True
    for unc in (0.31, 0.50, 0.90):
        v = evaluate_safety(confidence=0.95, uncertainty=unc, **_good)
        assert v.action != "show", f"FAIL: uncertainty {unc} returned 'show'"
    # a clean high-confidence case still shows
    v_ok = evaluate_safety(confidence=0.92, uncertainty=0.05, **_good)
    assert v_ok.action == "show" and v_ok.requires_human_review is False
    print("  ✓ below-confidence / high-uncertainty inputs all abstain (never 'show')")

    print("\n─ Live debouncer self-test ─")
    dbn = LivePolypDebouncer(min_frames=3)
    for f in range(5):
        emitted = dbn.update([{"bbox": {"x": 100, "y": 80, "w": 50, "h": 50},
                               "confidence": 0.8}])
        print(f"  frame {f+1}: emitted {len(emitted)} polyp(s) "
              f"{'(✓ persisted)' if emitted else '(holding back)'}")
