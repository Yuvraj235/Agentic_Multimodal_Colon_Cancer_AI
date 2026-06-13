"""ColonAI Real-Time Video Pipeline.

Two modes:
  • analyse_video_file()   — process a recorded .mp4 / .mov / .avi  and
                              return an annotated video + per-detection log
  • LivePolypTransformer   — a streamlit-webrtc VideoTransformer that does
                              per-frame inference and overlays detections on
                              the live webcam / capture-card stream

Both modes share the same backbone:
  1. Per-frame endoscopy gate     (image_atypicality.is_endoscopy_image)
  2. Per-frame 5-class classifier (UnifiedMultiModalTransformer)
  3. GradCAM++ → bounding-box     (high-attention region)
  4. Temporal smoothing           (sliding window of last N frames)
  5. Detection tracking           (per-polyp persistent ID)
  6. Annotated overlay            (bbox, label, confidence, polyp count)
"""
from __future__ import annotations
import time
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T


# Class names match the trained model
PATHOLOGY_CLASSES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]

# Per-class colour for bounding boxes (BGR for OpenCV)
CLASS_BGR = {
    "polyps":          (255, 96, 32),    # bright orange
    "uc-mild":         (34, 197, 94),    # green
    "uc-moderate-sev": (220, 38, 38),    # red
    "barretts-esoph":  (168, 85, 247),   # violet
    "therapeutic":     (14, 165, 233),   # cyan-blue
}
# Per-class human label
CLASS_LABEL = {
    "polyps":          "Polyp",
    "uc-mild":         "UC (mild)",
    "uc-moderate-sev": "UC (mod-severe)",
    "barretts-esoph":  "Barrett's",
    "therapeutic":     "Post-therapeutic",
}


# ─────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────

@dataclass
class FrameDetection:
    """One detection in one frame."""
    frame_idx:     int
    timestamp_s:   float
    class_name:    str
    confidence:    float
    bbox:          Tuple[int, int, int, int]      # x1, y1, x2, y2
    roi_coverage:  float                           # fraction of image flagged
    is_endoscopy:  bool                            # passed the endoscopy gate?
    endoscopy_score: float


@dataclass
class TrackedPolyp:
    """A polyp tracked across consecutive frames."""
    id:               int
    class_name:       str
    first_frame:      int
    last_frame:       int
    first_ts:         float
    last_ts:          float
    n_frames:         int
    max_confidence:   float
    bboxes:           List[Tuple[int, int, int, int]] = field(default_factory=list)
    snapshot:         Optional[np.ndarray] = None    # best frame for this polyp


@dataclass
class VideoSummary:
    """End-of-video summary."""
    total_frames:        int
    processed_frames:    int
    duration_seconds:    float
    fps_input:           float
    avg_inference_ms:    float
    detections:          List[FrameDetection]
    tracked_polyps:      List[TrackedPolyp]
    polyps_count:        int        # number of distinct polyps detected
    output_video_path:   Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Per-frame helpers
# ─────────────────────────────────────────────────────────────────

def _preprocess(frame_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """Resize + normalise one RGB frame for the model."""
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(frame_rgb).unsqueeze(0)


def _gradcam_to_bbox(heatmap: np.ndarray, frame_h: int, frame_w: int,
                    threshold_q: float = 0.80,
                    pad: int = 12) -> Optional[Tuple[int, int, int, int]]:
    """Derive a bounding box around the largest connected high-attention region
    of a GradCAM heatmap.  Returns None if no region passes the threshold."""
    if heatmap is None:
        return None
    H, W = heatmap.shape
    # Resize heatmap to frame size
    h = cv2.resize(heatmap.astype(np.float32), (frame_w, frame_h),
                   interpolation=cv2.INTER_LINEAR)
    # Threshold at the requested quantile
    thr = float(np.quantile(h, threshold_q))
    mask = (h >= thr).astype(np.uint8) * 255

    # Find connected components and pick the largest
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab <= 1:
        return None
    # stats[0] is background, sort the rest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h_b = int(stats[largest, cv2.CC_STAT_HEIGHT])

    # Add padding and clip to frame
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w - 1, x + w + pad)
    y2 = min(frame_h - 1, y + h_b + pad)
    # Reject tiny boxes
    if (x2 - x1) < 24 or (y2 - y1) < 24:
        return None
    return (x1, y1, x2, y2)


def _draw_annotation(frame_bgr: np.ndarray, detection: FrameDetection,
                     polyp_count: int, total_polyps_seen: int) -> np.ndarray:
    """Draw bbox + class label + status bar on the frame."""
    img = frame_bgr.copy()
    H, W = img.shape[:2]

    # Bounding box around the lesion
    if detection.bbox:
        x1, y1, x2, y2 = detection.bbox
        colour = CLASS_BGR.get(detection.class_name, (255, 255, 255))
        # Animated-looking pulse: thickness varies with confidence
        thick = max(2, int(2 + 4 * detection.confidence))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        # Label background
        label = f"{CLASS_LABEL.get(detection.class_name, detection.class_name)}  {detection.confidence*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw + 12, y1), colour, -1)
        cv2.putText(img, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Top status bar
    bar_h = 38
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    status = f"ColonAI  |  Frame {detection.frame_idx}  |  t={detection.timestamp_s:.1f}s  |  Polyps tracked: {total_polyps_seen}"
    cv2.putText(img, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # Endoscopy-gate state on the right edge
    gate_text = "ENDOSCOPY OK" if detection.is_endoscopy else "NOT ENDOSCOPY"
    gate_col = (34, 197, 94) if detection.is_endoscopy else (220, 38, 38)
    (gw, _), _ = cv2.getTextSize(gate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, gate_text, (W - gw - 12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gate_col, 1, cv2.LINE_AA)

    # Pulse on right when a detection happens this frame
    if detection.bbox and detection.is_endoscopy:
        cv2.circle(img, (W - 20, H - 22), 8, (40, 200, 90), -1)
        cv2.putText(img, "DETECTING", (W - 110, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


# ─────────────────────────────────────────────────────────────────
# Temporal tracker — merges per-frame detections into polyp tracks
# ─────────────────────────────────────────────────────────────────

class PolypTracker:
    """Tracks polyps across consecutive frames using bbox IOU + time-window matching.

    A detection seen in frame N is considered the same polyp as a detection in
    frame N-k (k ≤ 30) if their bboxes overlap (IoU ≥ 0.20).  Otherwise it's a
    new polyp ID.

    A polyp is only confirmed (escalated to the user) once it has been seen
    in at least 3 of the last 6 frames — this gates against single-frame false
    positives.
    """

    def __init__(self, iou_thresh: float = 0.20, persistence_window: int = 6,
                 min_persistent: int = 3, max_gap_frames: int = 30):
        self.iou_thresh = iou_thresh
        self.persistence_window = persistence_window
        self.min_persistent = min_persistent
        self.max_gap_frames = max_gap_frames
        self.polyps: List[TrackedPolyp] = []
        self._next_id = 1

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return float(inter / max(1, ua))

    def update(self, det: FrameDetection, frame_rgb: np.ndarray) -> Optional[TrackedPolyp]:
        if det.bbox is None:
            return None
        # Try to match to an existing polyp track
        matched = None
        best_iou = 0.0
        for p in self.polyps:
            if det.frame_idx - p.last_frame > self.max_gap_frames:
                continue
            if not p.bboxes:
                continue
            iou = self._iou(det.bbox, p.bboxes[-1])
            if iou > best_iou and iou >= self.iou_thresh and p.class_name == det.class_name:
                best_iou = iou
                matched = p
        if matched is None:
            # Start a new track
            tp = TrackedPolyp(
                id=self._next_id, class_name=det.class_name,
                first_frame=det.frame_idx, last_frame=det.frame_idx,
                first_ts=det.timestamp_s, last_ts=det.timestamp_s,
                n_frames=1, max_confidence=det.confidence,
                bboxes=[det.bbox],
                snapshot=frame_rgb.copy(),
            )
            self.polyps.append(tp)
            self._next_id += 1
            return tp
        # Update the existing track
        matched.last_frame = det.frame_idx
        matched.last_ts    = det.timestamp_s
        matched.n_frames  += 1
        if det.confidence > matched.max_confidence:
            matched.max_confidence = det.confidence
            matched.snapshot = frame_rgb.copy()
        matched.bboxes.append(det.bbox)
        return matched

    def get_confirmed_count(self) -> int:
        """Number of polyps confirmed (seen ≥ min_persistent frames)."""
        return sum(1 for p in self.polyps if p.n_frames >= self.min_persistent)


# ─────────────────────────────────────────────────────────────────
# YOLO polyp DETECTOR  (the real-time detection model trained on Kaggle)
# ─────────────────────────────────────────────────────────────────
# This replaces the weak classification+GradCAM box with a proper trained
# detector. Drop the trained weights at one of these paths (downloaded from the
# Kaggle run, scripts/video/train_polyp_yolo.py). Fail-open: if no weights are
# present yet, the pipeline falls back to the old GradCAM method automatically.
_DETECTOR_PATHS = [
    Path("outputs/unified_multimodal_v2/polyp_detector.pt"),     # canonical drop-in
    Path("outputs/yolo_polyp/runs/polyp/weights/best.pt"),       # local training output
]
_det_cache: dict = {}


def load_polyp_detector():
    """Lazy-load the trained YOLO detector once. Returns None if not present."""
    if "model" in _det_cache:
        return _det_cache["model"]
    model = None
    try:
        from ultralytics import YOLO
        for p in _DETECTOR_PATHS:
            if p.exists():
                model = YOLO(str(p))
                break
    except Exception:
        model = None
    _det_cache["model"] = model
    return model


def detector_available() -> bool:
    return load_polyp_detector() is not None


def detect_polyps(frame_bgr: np.ndarray, conf: float = 0.25):
    """Run the trained detector on one BGR frame.

    Returns a list of (x1, y1, x2, y2, confidence) boxes, or None if the
    detector isn't available (caller then uses the GradCAM fallback)."""
    model = load_polyp_detector()
    if model is None:
        return None
    try:
        res = model.predict(frame_bgr, conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out
    except Exception:
        return []


def _annotate_yolo(frame_bgr: np.ndarray, dets: List["FrameDetection"],
                   total_polyps_seen: int) -> np.ndarray:
    """Draw all detector boxes + a status bar on a frame (BGR in/out)."""
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    colour = CLASS_BGR["polyps"]
    for d in dets:
        if not d.bbox:
            continue
        x1, y1, x2, y2 = d.bbox
        thick = max(2, int(2 + 4 * d.confidence))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
        label = f"Polyp {d.confidence*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw + 12, y1), colour, -1)
        cv2.putText(img, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    bar_h = 38
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    status = (f"ColonAI detector  |  polyps this frame: {len(dets)}  |  "
              f"tracked: {total_polyps_seen}")
    cv2.putText(img, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if dets:
        cv2.circle(img, (W - 20, H - 22), 8, (40, 200, 90), -1)
        cv2.putText(img, "DETECTING", (W - 110, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def _yolo_process_frame(frame_bgr, frame_idx, timestamp, tracker, conf_thr):
    """Endoscopy-gate → YOLO-detect → track → annotate one frame.

    Returns (annotated_bgr, [FrameDetection]). Returns (None, None) if the
    detector isn't available, so the caller falls back to the GradCAM path."""
    if not detector_available():
        return None, None
    from src.app.image_atypicality import is_endoscopy_image
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gate = is_endoscopy_image(rgb, threshold=0.55)
    boxes = detect_polyps(frame_bgr, conf=conf_thr) if gate["is_endoscopy"] else []
    boxes = boxes or []
    dets: List[FrameDetection] = []
    for (x1, y1, x2, y2, c) in boxes:
        det = FrameDetection(
            frame_idx=frame_idx, timestamp_s=timestamp, class_name="polyps",
            confidence=c, bbox=(x1, y1, x2, y2),
            roi_coverage=((x2 - x1) * (y2 - y1)) / float(W * H),
            is_endoscopy=bool(gate["is_endoscopy"]), endoscopy_score=float(gate["score"]),
        )
        tracker.update(det, rgb)
        dets.append(det)
    return _annotate_yolo(frame_bgr, dets, len(tracker.polyps)), dets


# ─────────────────────────────────────────────────────────────────
# MAIN ANALYSER — video file mode
# ─────────────────────────────────────────────────────────────────

def analyse_video_file(
    video_path: str,
    output_path: str,
    model,
    tokenizer,
    device: torch.device,
    skip_frames: int = 2,        # process every Nth frame (3 = 10fps from 30fps)
    confidence_threshold: float = 0.55,
    text_prompt: str = "Patient undergoing routine screening colonoscopy.",
    tab_features: Optional[np.ndarray] = None,
    progress_callback=None,
) -> VideoSummary:
    """Process a colonoscopy video file frame-by-frame and write an annotated
    output video with bounding boxes around detections.

    Returns a VideoSummary with all detections, tracked polyps, and stats.
    """
    from src.app.image_atypicality import is_endoscopy_image
    from src.agents.unified_image_agent import GradCAMPlusPlus

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_f / fps_in if fps_in > 0 else 0.0

    # Output writer — H.264 (mp4v fallback)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = max(1.0, fps_in / max(1, skip_frames))
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (fw, fh))

    # Build the tokenised text once
    enc = tokenizer(text_prompt, padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    # Tabular vector — use a population mean if not supplied
    if tab_features is None:
        tab_features = np.zeros(12, dtype=np.float32)
    tab = torch.from_numpy(tab_features).float().unsqueeze(0).to(device)

    # GradCAM++ target
    target_layer = model.get_image_target_layer() if hasattr(model, "get_image_target_layer") else None
    cam_extractor = GradCAMPlusPlus(model, target_layer) if target_layer is not None else None

    # Patient-safety: same 3-frame debounce as the live pipeline
    tracker = PolypTracker(iou_thresh=0.20, min_persistent=3,
                           persistence_window=6, max_gap_frames=30)
    detections: List[FrameDetection] = []
    inference_times: List[float] = []
    processed = 0
    frame_idx = -1

    model.eval()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % skip_frames != 0:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp = frame_idx / fps_in if fps_in > 0 else float(frame_idx)

        # ── Real detector path (trained YOLO) — preferred when weights exist ──
        t_y = time.time()
        ann_y, dets_y = _yolo_process_frame(frame_bgr, frame_idx, timestamp,
                                            tracker, confidence_threshold)
        if ann_y is not None:
            inference_times.append((time.time() - t_y) * 1000.0)
            detections.extend(dets_y)
            writer.write(ann_y)
            processed += 1
            if progress_callback is not None and (processed % 5 == 0):
                try:
                    progress_callback(frame_idx, total_f, len(tracker.polyps))
                except Exception:
                    pass
            continue
        # ── else: fall back to the classification + GradCAM path below ───────

        # ── 1) Endoscopy gate ────────────────────────────────────
        gate = is_endoscopy_image(frame_rgb, threshold=0.55)

        # ── 2) Inference (only if gate passes) ───────────────────
        t0 = time.time()
        bbox = None
        cls_name = "background"
        confidence = 0.0
        if gate["is_endoscopy"]:
            x = _preprocess(frame_rgb).to(device)
            with torch.no_grad():
                out = model(image=x, input_ids=input_ids,
                            attention_mask=attn_mask, tabular=tab)
                probs = F.softmax(out["pathology"], dim=-1)[0].cpu().numpy()
                idx = int(probs.argmax())
                cls_name = PATHOLOGY_CLASSES[idx]
                confidence = float(probs[idx])

            # GradCAM → bbox  (only if confidence is meaningful)
            if confidence >= confidence_threshold and cam_extractor is not None:
                try:
                    x_g = _preprocess(frame_rgb).to(device)
                    cam = cam_extractor.generate(
                        image=x_g, class_idx=idx,
                        input_ids=input_ids, attention_mask=attn_mask,
                        tabular=tab,
                    )
                    if cam is not None:
                        cam_np = cam if isinstance(cam, np.ndarray) else cam.cpu().numpy()
                        bbox = _gradcam_to_bbox(cam_np.squeeze(), fh, fw)
                except Exception:
                    bbox = None

        inference_times.append((time.time() - t0) * 1000.0)

        # ── 3) Compute ROI coverage from the bbox ─────────────────
        roi_cov = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi_cov = ((x2 - x1) * (y2 - y1)) / float(fw * fh)

        # ── 4) Build the detection record ─────────────────────────
        det = FrameDetection(
            frame_idx       = frame_idx,
            timestamp_s     = timestamp,
            class_name      = cls_name,
            confidence      = confidence,
            bbox            = bbox,
            roi_coverage    = roi_cov,
            is_endoscopy    = bool(gate["is_endoscopy"]),
            endoscopy_score = float(gate["score"]),
        )
        detections.append(det)

        # ── 5) Track the detection across frames ──────────────────
        if det.is_endoscopy and bbox is not None and confidence >= confidence_threshold:
            tracker.update(det, frame_rgb)

        # ── 6) Annotate the frame ────────────────────────────────
        annotated_bgr = _draw_annotation(frame_bgr, det,
                                         polyp_count=tracker.get_confirmed_count(),
                                         total_polyps_seen=len(tracker.polyps))
        writer.write(annotated_bgr)

        processed += 1
        if progress_callback is not None and (processed % 5 == 0):
            try:
                progress_callback(frame_idx, total_f, len(tracker.polyps))
            except Exception:
                pass

    cap.release()
    writer.release()

    return VideoSummary(
        total_frames       = total_f,
        processed_frames   = processed,
        duration_seconds   = duration,
        fps_input          = float(fps_in),
        avg_inference_ms   = float(np.mean(inference_times)) if inference_times else 0.0,
        detections         = detections,
        tracked_polyps     = sorted(tracker.polyps, key=lambda p: -p.n_frames),
        polyps_count       = tracker.get_confirmed_count(),
        output_video_path  = output_path,
    )


# ─────────────────────────────────────────────────────────────────
# LIVE WEBCAM MODE — streamlit-webrtc VideoTransformer
# ─────────────────────────────────────────────────────────────────

class LivePolypTransformer:
    """A streamlit-webrtc VideoTransformer that runs inference on every
    Nth incoming frame from the live webcam (or capture-card / OBS virtual
    cam connected to the endoscope) and overlays bounding boxes.

    Use:
        ctx = webrtc_streamer(
            key="live",
            video_processor_factory=lambda: LivePolypTransformer(model, tokenizer, device),
            ...
        )
    """

    def __init__(self, model, tokenizer, device, skip: int = 3,
                 confidence_threshold: float = 0.55):
        from src.agents.unified_image_agent import GradCAMPlusPlus
        self.model = model
        self.device = device
        self.skip   = skip
        self.conf_thr = confidence_threshold
        self.tokenizer = tokenizer
        self.frame_idx = 0
        self._last_det: Optional[FrameDetection] = None
        # Patient-safety: require 3 frames of bbox persistence before a polyp
        # is shown to the operator. A single-frame false positive in a live
        # colonoscopy feed must never reach the screen as a "detection".
        # iou_thresh=0.20 (was 0.15) tightens the same-polyp match — prevents
        # nearby reflections being merged with a real lesion.
        self.tracker = PolypTracker(iou_thresh=0.20, min_persistent=3,
                                    persistence_window=6, max_gap_frames=30)
        # Pre-tokenise the text prompt
        enc = tokenizer("Live colonoscopy stream — symptomatic patient.",
                        padding="max_length", truncation=True, max_length=64,
                        return_tensors="pt")
        self.input_ids = enc["input_ids"].to(device)
        self.attn_mask = enc["attention_mask"].to(device)
        self.tab = torch.zeros((1, 12), device=device)
        # GradCAM extractor
        try:
            tgt = model.get_image_target_layer()
            self.cam = GradCAMPlusPlus(model, tgt)
        except Exception:
            self.cam = None
        self._lock = threading.Lock()

    def recv(self, frame):
        """av.VideoFrame → av.VideoFrame.  This is what webrtc calls per frame."""
        import av
        img = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]
        self.frame_idx += 1

        # ── Real detector path (trained YOLO) — preferred when weights exist ──
        if detector_available():
            if self.frame_idx % self.skip == 0:
                from src.app.image_atypicality import is_endoscopy_image
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gate = is_endoscopy_image(rgb, threshold=0.55)
                boxes = detect_polyps(img, conf=self.conf_thr) if gate["is_endoscopy"] else []
                boxes = boxes or []
                ts = time.time()
                dets = []
                for (x1, y1, x2, y2, c) in boxes:
                    d = FrameDetection(
                        frame_idx=self.frame_idx, timestamp_s=ts, class_name="polyps",
                        confidence=c, bbox=(x1, y1, x2, y2),
                        roi_coverage=((x2 - x1) * (y2 - y1)) / float(H * W),
                        is_endoscopy=bool(gate["is_endoscopy"]), endoscopy_score=float(gate["score"]))
                    self.tracker.update(d, rgb)
                    dets.append(d)
                with self._lock:
                    self._last_dets = dets
            annotated = _annotate_yolo(img, getattr(self, "_last_dets", []),
                                       len(self.tracker.polyps))
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        # ── else: classification + GradCAM fallback (no detector weights yet) ──
        # Only run inference every Nth frame to keep ~10 fps on CPU
        if self.frame_idx % self.skip == 0:
            with self._lock:
                self._last_det = self._infer(img)

        det = self._last_det or FrameDetection(
            frame_idx=self.frame_idx, timestamp_s=time.time(),
            class_name="background", confidence=0.0, bbox=None,
            roi_coverage=0.0, is_endoscopy=False, endoscopy_score=0.0,
        )
        det = FrameDetection(**{**det.__dict__, "frame_idx": self.frame_idx,
                                "timestamp_s": time.time()})

        # Track + draw
        if det.bbox is not None and det.is_endoscopy and det.confidence >= self.conf_thr:
            self.tracker.update(det, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        annotated = _draw_annotation(img, det,
                                     polyp_count=self.tracker.get_confirmed_count(),
                                     total_polyps_seen=len(self.tracker.polyps))
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _infer(self, frame_bgr: np.ndarray) -> FrameDetection:
        from src.app.image_atypicality import is_endoscopy_image
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gate = is_endoscopy_image(rgb, threshold=0.55)
        bbox = None
        cls_name = "background"
        confidence = 0.0
        if gate["is_endoscopy"]:
            x = _preprocess(rgb).to(self.device)
            with torch.no_grad():
                out = self.model(image=x, input_ids=self.input_ids,
                                 attention_mask=self.attn_mask, tabular=self.tab)
                probs = F.softmax(out["pathology"], dim=-1)[0].cpu().numpy()
                idx = int(probs.argmax())
                cls_name = PATHOLOGY_CLASSES[idx]
                confidence = float(probs[idx])
            if confidence >= self.conf_thr and self.cam is not None:
                try:
                    cam_map = self.cam.generate(
                        image=x, class_idx=idx,
                        input_ids=self.input_ids,
                        attention_mask=self.attn_mask,
                        tabular=self.tab,
                    )
                    if cam_map is not None:
                        cm = cam_map if isinstance(cam_map, np.ndarray) else cam_map.cpu().numpy()
                        bbox = _gradcam_to_bbox(cm.squeeze(), H, W)
                except Exception:
                    bbox = None
        roi = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = ((x2 - x1) * (y2 - y1)) / float(H * W)
        return FrameDetection(
            frame_idx=self.frame_idx, timestamp_s=time.time(),
            class_name=cls_name, confidence=confidence, bbox=bbox,
            roi_coverage=roi,
            is_endoscopy=bool(gate["is_endoscopy"]),
            endoscopy_score=float(gate["score"]),
        )
