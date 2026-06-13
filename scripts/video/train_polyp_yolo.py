"""Train a real-time polyp DETECTOR (YOLO) — free, on Kaggle (or local MPS).

Runs on the YOLO dataset built by masks_to_yolo.py. Transfer-learns from COCO-
pretrained weights, so it converges fast on our ~4.5k polyp images. Designed for
Kaggle's free T4 (30 GPU-hrs/week): checkpoint/resume survives the 9-12 hr session
cap, and the honest number comes from the ETIS (Pentax) test split that's held out
of training. Exports ONNX + CoreML so the finished model runs real-time on the Mac
for live video.

Kaggle:  !python train_polyp_yolo.py --data /kaggle/input/<your-dataset>/data.yaml
Local :  python3 scripts/video/train_polyp_yolo.py            (uses outputs/yolo_polyp)
Resume:  add --resume   (picks up the last checkpoint after a session restart)

Note: Ultralytics YOLO is AGPL-3.0 (free + open source; fine for research/demo).
MODEL is just a string Ultralytics auto-downloads — swap to any current variant
(yolov8s.pt, yolo11s.pt, …) without other changes.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/yolo_polyp/data.yaml"
PROJECT = ROOT / "outputs/yolo_polyp/runs"


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _fix_yaml_path(data_yaml: Path):
    """Point data.yaml's `path:` at its own folder so it works wherever the
    dataset is mounted (local repo, /kaggle/input/..., etc.)."""
    lines = data_yaml.read_text().splitlines()
    out, seen = [], False
    for ln in lines:
        if ln.strip().startswith("path:"):
            out.append(f"path: {data_yaml.parent}"); seen = True
        else:
            out.append(ln)
    if not seen:
        out.insert(0, f"path: {data_yaml.parent}")
    data_yaml.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights (auto-downloaded)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience")
    ap.add_argument("--name", default="polyp_yolo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install ultralytics")

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        sys.exit(f"data.yaml not found: {data_yaml}\nRun scripts/video/masks_to_yolo.py first.")
    _fix_yaml_path(data_yaml)
    dev = _device()
    print(f"device={dev} · model={args.model} · data={data_yaml}")

    run_dir = PROJECT / args.name
    last = run_dir / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"Resuming from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.model)            # COCO-pretrained → transfer learning
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, device=dev, project=str(PROJECT), name=args.name,
            exist_ok=True, seed=42,
            # light medical-image-appropriate augmentation
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, flipud=0.3,
            mosaic=1.0, close_mosaic=10, degrees=10, translate=0.1, scale=0.4,
        )

    # ── HONEST held-out evaluation on ETIS (Pentax — never trained on) ──────
    best = run_dir / "weights" / "best.pt"
    model = YOLO(str(best)) if best.exists() else model
    print("\n=== Honest held-out test (ETIS / Pentax) ===")
    m = model.val(data=str(data_yaml), split="test", device=dev, name=f"{args.name}_test")
    try:
        print(f"  mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}  "
              f"precision={m.box.mp:.4f}  recall(sensitivity)={m.box.mr:.4f}")
    except Exception:
        print("  (metrics object:", m, ")")

    if not args.no_export:
        print("\nExporting for the Mac live demo …")
        for fmt in ("onnx", "coreml"):
            try:
                p = model.export(format=fmt, imgsz=args.imgsz)
                print(f"  ✓ {fmt}: {p}")
            except Exception as e:
                print(f"  · {fmt} export skipped ({type(e).__name__}: {e})")
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
