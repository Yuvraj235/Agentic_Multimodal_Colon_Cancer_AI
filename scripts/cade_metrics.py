"""ColonAI — CADe (Computer-Aided Detection) metrics.

Standard CAD-EYE / ENDO-AID-comparable metrics:

   * Per-image sensitivity @ IoU≥0.5
   * Mean Average Precision (mAP) at IoU 0.5
   * False Positives Per Image (FPPI) @ sensitivity 0.9
   * Per-polyp detection rate (image-level)

Treats each image as having ≤1 polyp (matches the test datasets).
The predicted bbox is derived from the GradCAM heatmap thresholded
at the 75-th percentile + connected-component bounding box.

Saves outputs/unified_multimodal_v2/cade_metrics.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from src.agents.unified_image_agent import GradCAMPlusPlus
from transformers import AutoTokenizer

CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
BERT = "dmis-lab/biobert-base-cased-v1.2"
OUT  = Path("outputs/unified_multimodal_v2/cade_metrics.json")

DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir"),
]


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def bbox_iou(a, b):
    if a is None or b is None: return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def predicted_bbox(cam, target_size, quantile=0.75):
    """Threshold the CAM, take the largest connected component, return bbox."""
    h, w = target_size
    cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    thr = float(np.quantile(cam, quantile))
    binm = (cam >= thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 1: return None, 0.0
    # largest area among CCs (skip background 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])
    score = float(cam[y:y+bh, x:x+bw].mean())
    return (x, y, x + bw, y + bh), score


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(CKPT, map_location=device)
    model.load_state_dict(s.get("model_state", s), strict=False)
    model.eval()
    cam_ex = GradCAMPlusPlus(model, model.get_image_target_layer())
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids = enc["input_ids"].to(device); msk = enc["attention_mask"].to(device)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=device)
    tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    report = {}
    for ds_name, root in DATASETS:
        root = Path(root); img_dir = root / "images"; msk_dir = root / "masks"
        if not img_dir.exists() or not msk_dir.exists(): continue
        files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".tif",".tiff",".png",".jpg",".jpeg",".bmp")])

        results = []   # per-image: (pred_bbox, gt_bbox, score, polyp_prob)
        print(f"\n[{ds_name}]  {len(files)} images")
        for img_path in tqdm(files, desc=ds_name, leave=False):
            mp = None
            for e in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                c = msk_dir / (img_path.stem + e)
                if c.exists(): mp = c; break
            if mp is None: continue
            try:
                pim = Image.open(img_path).convert("RGB")
                gt  = (np.array(Image.open(mp).convert("L")) > 127).astype(np.uint8)
            except Exception: continue
            gt_bb = bbox_from_mask(gt)
            x = tfm(pim).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = F.softmax(model(x, ids, msk, tab)["pathology"], dim=-1)[0]
                polyp_prob = float(prob[0])
            cam = cam_ex.generate(image=x.detach().requires_grad_(True),
                                  class_idx=0, input_ids=ids,
                                  attention_mask=msk, tabular=tab)
            pred_bb, conf = (predicted_bbox(cam, (pim.height, pim.width))
                             if cam is not None and cam.size >= 4 else (None, 0.0))
            results.append((pred_bb, gt_bb, polyp_prob * conf, polyp_prob))

        # ── Metrics ────────────────────────────────────────────────────
        # 1) Sensitivity @ IoU ≥ 0.5 (per-image detection)
        tp_at_05 = sum(1 for pb, gb, _, pp in results
                       if (gb is not None and pp >= 0.5 and bbox_iou(pb, gb) >= 0.5))
        n_with_gt = sum(1 for _, gb, _, _ in results if gb is not None)
        sens_05 = tp_at_05 / max(1, n_with_gt)
        # 2) FPPI @ polyp_prob ≥ 0.5
        n_pred = sum(1 for _, _, _, pp in results if pp >= 0.5)
        n_fp   = sum(1 for pb, gb, _, pp in results
                     if (pp >= 0.5 and (gb is None or bbox_iou(pb, gb) < 0.5)))
        fppi   = n_fp / max(1, len(results))
        # 3) mAP @ IoU 0.5
        sorted_r = sorted(results, key=lambda r: -r[2])  # descending by score
        tps = fps = 0; prs = []
        for pb, gb, _, pp in sorted_r:
            if gb is not None and bbox_iou(pb, gb) >= 0.5: tps += 1
            else: fps += 1
            prec = tps / max(1, tps + fps); rec = tps / max(1, n_with_gt)
            prs.append((rec, prec))
        prs.sort()
        # 11-point interpolated AP
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            ps = [p for rr, p in prs if rr >= r]
            ap += max(ps) if ps else 0
        ap /= 11

        report[ds_name] = {
            "n_images":         len(results),
            "n_with_mask":      n_with_gt,
            "sensitivity_iou50": sens_05,
            "fppi_at_p50":      fppi,
            "mAP_iou50":        float(ap),
            "polyp_pred_rate":  sum(1 for _,_,_, pp in results if pp >= 0.5)/max(1,len(results)),
        }
        print(f"  ✓ {ds_name}:  sens@IoU50 = {sens_05:.3f}   FPPI = {fppi:.3f}   mAP = {ap:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    # Summary table
    print(f"\n{'='*72}\nCADe METRICS — v2 across cross-vendor datasets\n{'='*72}")
    print(f"{'Dataset':22s} {'sens@IoU50':>10} {'FPPI':>8} {'mAP@50':>8} {'n':>5}")
    print("-"*72)
    for k, v in report.items():
        print(f"{k:22s} {v['sensitivity_iou50']:10.3f} {v['fppi_at_p50']:8.3f} "
              f"{v['mAP_iou50']:8.3f} {v['n_images']:5d}")
    print("="*72)
    print(f"\nReport → {OUT}")


if __name__ == "__main__":
    main()
