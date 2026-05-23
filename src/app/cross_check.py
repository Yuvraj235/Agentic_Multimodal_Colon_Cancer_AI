"""ColonAI — inference-time cross-check between the heads.

The pathology classifier, the GradCAM heatmap, and the segmentation
decoder all carry independent evidence. When they DISAGREE, that's a
red flag — even if any single one looks confident. This module fuses
those signals into a single coherence score and a structured rationale.

Used by the safety policy as the `agent_agreement` input.

Three pairwise checks
─────────────────────
   1. **Class ↔ Segmentation coverage**
        If pathology says "polyps" but segmentation decoder finds < 1 % of
        pixels above threshold → no visible polyp → DISAGREE.
        If pathology says NOT "polyps" but segmentation decoder finds a
        large coherent mask → MAYBE missed polyp → DISAGREE.

   2. **GradCAM peak ↔ Segmentation mask location**
        Mass-centre of the top-25 % GradCAM activation should land
        inside the segmentation mask. Computed as IoU between
        GradCAM-thresholded mask and the seg-decoder mask.

   3. **GradCAM ↔ Integrated-Gradients agreement (if both available)**
        IoU between the two attribution maps. Two attribution methods
        that disagree on what the model is using is a strong signal of
        a fragile prediction.

Output (`CrossCheckReport`)
   * `coherence` ∈ [0, 1]  — geometric mean of the three signals
   * `is_consistent`       — coherence ≥ COHERENCE_FLOOR
   * `signals`             — dict of individual measurements
   * `flags`               — list of human-readable disagreement reasons
   * `rationale`           — list of bullets explaining the prediction
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2


# Thresholds — tuned against the validation/cross-vendor corpus
COHERENCE_FLOOR        = 0.50    # below this → safety policy abstains
SEG_MIN_POLYP_AREA     = 0.005   # 0.5 % of pixels
SEG_MAX_NON_POLYP_AREA = 0.06    # 6 % considered "large coherent mask"
GRADCAM_THRESHOLD_PCT  = 0.75
IG_AGREE_THRESHOLD     = 0.20


@dataclass
class CrossCheckReport:
    coherence:      float
    is_consistent:  bool
    signals:        Dict[str, float] = field(default_factory=dict)
    flags:          List[str]        = field(default_factory=list)
    rationale:      List[str]        = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or (a, b).sum())
    return inter / union if union > 1 else 0.0


def _threshold_top_q(arr: np.ndarray, q: float = GRADCAM_THRESHOLD_PCT) -> np.ndarray:
    """Top-q quantile mask of a heatmap."""
    if arr is None or arr.size == 0:
        return np.zeros((1, 1), dtype=bool)
    thr = float(np.quantile(arr, q))
    return arr >= thr


def _coverage(mask: Optional[np.ndarray]) -> float:
    if mask is None or mask.size == 0: return 0.0
    return float(mask.astype(bool).sum()) / float(mask.size)


def _largest_blob_area_pct(binmask: np.ndarray) -> float:
    if binmask.sum() < 4: return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binmask.astype(np.uint8), 8)
    if n <= 1: return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(areas.max()) / float(binmask.size)


def _quadrant_label(mask: np.ndarray) -> str:
    """Return rough quadrant of the mask's centroid: upper-left, lower-right, …"""
    if mask.sum() < 4: return "uniform"
    ys, xs = np.where(mask)
    cy, cx = ys.mean() / mask.shape[0], xs.mean() / mask.shape[1]
    v = "upper" if cy < 0.5 else "lower"
    h = "left"  if cx < 0.5 else "right"
    return f"{v}-{h}"


def cross_check(
    pathology_class:    str,
    pathology_conf:     float,
    gradcam_map:        Optional[np.ndarray] = None,
    segmentation_mask:  Optional[np.ndarray] = None,
    ig_map:             Optional[np.ndarray] = None,
    image_shape_hw:     Optional[Tuple[int, int]] = None,
) -> CrossCheckReport:
    """Run pairwise consistency checks.

    All maps are 2-D float arrays (heatmap or mask). They will be resized
    to a common 224×224 grid for the comparison.
    """
    signals: Dict[str, float] = {}
    flags:   List[str]        = []
    rationale: List[str]      = []
    is_polyp = pathology_class.lower() == "polyps"

    # ── Normalise everything to 224×224 binary masks ─────────────────────
    def _to_224(arr):
        if arr is None or arr.size == 0: return None
        if arr.shape != (224, 224):
            arr = cv2.resize(arr.astype(np.float32), (224, 224),
                             interpolation=cv2.INTER_LINEAR)
        return arr

    gc224  = _to_224(gradcam_map)
    seg224 = _to_224(segmentation_mask)
    ig224  = _to_224(ig_map)

    # ── 1. Class ↔ Segmentation coverage ─────────────────────────────────
    seg_cov = _coverage(seg224 > 0.5) if seg224 is not None else None
    if seg_cov is not None:
        signals["seg_coverage"] = seg_cov
        if is_polyp and seg_cov < SEG_MIN_POLYP_AREA:
            flags.append("class_says_polyp_but_seg_finds_nothing")
            rationale.append(
                f"Classifier reports a polyp ({pathology_conf*100:.0f}% conf) but "
                f"the segmentation map highlights only {seg_cov*100:.2f}% of the "
                f"image — no visible lesion shape.")
            class_seg_score = 0.0
        elif (not is_polyp) and _largest_blob_area_pct(seg224 > 0.5) > SEG_MAX_NON_POLYP_AREA:
            flags.append("class_says_non_polyp_but_seg_finds_blob")
            rationale.append(
                "A coherent lesion-shaped region was found, but the classifier "
                "did not call it a polyp — manual review recommended.")
            class_seg_score = 0.3
        else:
            class_seg_score = 1.0
        signals["class_seg_consistency"] = class_seg_score
    else:
        class_seg_score = 0.7   # no seg → reduced (not failed) signal

    # ── 2. GradCAM peak ↔ Segmentation IoU ───────────────────────────────
    gc_seg_iou = None
    if gc224 is not None and seg224 is not None:
        gc_bin  = _threshold_top_q(gc224)
        seg_bin = seg224 > 0.5
        if seg_bin.sum() >= 4 and gc_bin.sum() >= 4:
            gc_seg_iou = _binary_iou(gc_bin, seg_bin)
            signals["gradcam_seg_iou"] = float(gc_seg_iou)
            if is_polyp and gc_seg_iou < 0.10:
                flags.append("gradcam_peak_outside_mask")
                rationale.append(
                    f"GradCAM hotspot does not overlap the predicted polyp "
                    f"region (IoU = {gc_seg_iou:.2f}). The model may be using "
                    f"background features.")
    # Score: clip 0..1
    gc_seg_score = (min(1.0, gc_seg_iou * 3.0) if gc_seg_iou is not None
                    else 0.7)

    # ── 3. GradCAM ↔ Integrated-Gradients IoU ────────────────────────────
    gc_ig_iou = None
    if gc224 is not None and ig224 is not None:
        gc_bin = _threshold_top_q(gc224)
        ig_bin = _threshold_top_q(ig224)
        if gc_bin.sum() >= 4 and ig_bin.sum() >= 4:
            gc_ig_iou = _binary_iou(gc_bin, ig_bin)
            signals["gradcam_ig_iou"] = float(gc_ig_iou)
            if gc_ig_iou < IG_AGREE_THRESHOLD:
                flags.append("xai_methods_disagree")
                rationale.append(
                    f"Two independent attribution methods (GradCAM++ and "
                    f"Integrated Gradients) disagree about which pixels drove "
                    f"the prediction (IoU = {gc_ig_iou:.2f}).")
    gc_ig_score = (min(1.0, gc_ig_iou * 4.0) if gc_ig_iou is not None
                   else 0.8)

    # ── Coherence = geometric mean (penalises any low signal) ───────────
    coherence = float((class_seg_score * gc_seg_score * gc_ig_score) ** (1.0 / 3.0))
    signals["coherence_geomean"] = coherence

    # Positive rationale bullets
    if is_polyp and gc_seg_iou is not None and gc_seg_iou >= 0.30:
        rationale.append(
            f"GradCAM++ hotspot overlaps the polyp segmentation region "
            f"(IoU = {gc_seg_iou:.2f}) — the model is looking at the lesion, "
            f"not at scope artefacts.")
    if seg_cov is not None and is_polyp and SEG_MIN_POLYP_AREA <= seg_cov <= 0.5:
        rationale.append(
            f"Segmentation decoder estimates the lesion covers "
            f"{seg_cov*100:.1f}% of the visible field.")
    if seg224 is not None:
        quad = _quadrant_label(seg224 > 0.5)
        if quad != "uniform":
            rationale.append(f"Most attention concentrated in the **{quad}** "
                             f"quadrant of the image.")

    return CrossCheckReport(
        coherence=coherence,
        is_consistent=(coherence >= COHERENCE_FLOOR),
        signals=signals,
        flags=flags,
        rationale=rationale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─ cross_check self-test ─")
    # Synthetic: GradCAM, seg mask, IG all agree on a disk in the centre
    H = W = 224
    yy, xx = np.mgrid[0:H, 0:W]
    blob = ((yy-110)**2 + (xx-110)**2) < 35**2
    gc   = blob.astype(np.float32) + 0.1*np.random.rand(H, W)
    seg  = blob.astype(np.float32) * 0.9
    ig   = blob.astype(np.float32) + 0.05*np.random.rand(H, W)

    print("\n  case A: all three agree on polyp →")
    r = cross_check("polyps", 0.92, gc, seg, ig)
    print(f"     coherence = {r.coherence:.3f}  consistent = {r.is_consistent}")
    for s, v in r.signals.items(): print(f"       {s:25s} = {v:.3f}")
    for line in r.rationale: print(f"       • {line}")

    print("\n  case B: classifier says polyp but seg is empty →")
    r = cross_check("polyps", 0.95, gc, np.zeros((H, W)), ig)
    print(f"     coherence = {r.coherence:.3f}  consistent = {r.is_consistent}")
    for f in r.flags: print(f"       flag: {f}")

    print("\n  case C: gradcam in upper-left, seg in centre →")
    bad_gc = np.zeros((H, W))
    bad_gc[:80, :80] = 1.0
    r = cross_check("polyps", 0.90, bad_gc, seg, ig)
    print(f"     coherence = {r.coherence:.3f}  consistent = {r.is_consistent}")
    for f in r.flags: print(f"       flag: {f}")
    for line in r.rationale: print(f"       • {line}")
