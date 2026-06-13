"""Real-time polyp detection on the Mac — webcam, capture card, or video file.

The true live demo: Apple Silicon runs YOLO inference fast, so this draws polyp
boxes on a live feed in real time. Point it at a webcam, an endoscope capture
card (OBS virtual cam), or a recorded clip.

Usage:
  python3 scripts/video/live_detect.py                      # default webcam (0)
  python3 scripts/video/live_detect.py --source clip.mp4    # a video file
  python3 scripts/video/live_detect.py --source 1           # second camera / capture card
  python3 scripts/video/live_detect.py --conf 0.3 --weights path/to/best.pt
Press 'q' to quit.

Drop the trained weights at outputs/unified_multimodal_v2/polyp_detector.pt
(downloaded from the Kaggle run) and no --weights flag is needed.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = [
    ROOT / "outputs/unified_multimodal_v2/polyp_detector.pt",
    ROOT / "outputs/yolo_polyp/runs/polyp/weights/best.pt",
]


def _resolve_weights(arg):
    if arg:
        return arg
    for p in DEFAULT_WEIGHTS:
        if p.exists():
            return str(p)
    sys.exit("No detector weights found. Train on Kaggle (docs/VIDEO_PHASE.md), then\n"
             "drop best.pt at outputs/unified_multimodal_v2/polyp_detector.pt, or pass --weights.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index (0,1,...) or video file path")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"missing dep: {e}. Run: pip install ultralytics opencv-python")

    weights = _resolve_weights(args.weights)
    print(f"detector: {weights}")
    model = YOLO(weights)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"could not open source: {args.source}")

    print("Running — press 'q' in the window to quit.")
    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, verbose=False)[0]
        n = 0
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            c = float(b.conf[0]); n += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 96, 255), max(2, int(2 + 4 * c)))
            lbl = f"Polyp {c*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 12, y1), (32, 96, 255), -1)
            cv2.putText(frame, lbl, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        now = time.time(); fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_prev)); t_prev = now
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (15, 23, 42), -1)
        cv2.putText(frame, f"ColonAI live  |  polyps: {n}  |  {fps:4.1f} FPS",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("ColonAI — live polyp detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
