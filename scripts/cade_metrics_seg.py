"""ColonAI — CADe metrics derived from the segmentation DECODER (not GradCAM).

Same metrics as cade_metrics.py but the predicted bbox is the bounding
box of the segmentation decoder's thresholded output, which is far
tighter than GradCAM++ heatmaps.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from transformers import AutoTokenizer

# import the SegDecoder class from the training script
sys.path.insert(0, "scripts")
from train_segmentation_head import SegDecoder

CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
SEG  = "outputs/unified_multimodal_v2/seg_head.pth"
OUT  = Path("outputs/unified_multimodal_v2/cade_metrics_seg.json")

DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir"),
]


def bbox_from_mask(m):
    ys, xs = np.where(m > 0)
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


def bbox_from_seg(seg_logits_224, target_h, target_w):
    """Threshold the seg output, take largest CC, return bbox in target resolution."""
    p = torch.sigmoid(seg_logits_224)[0, 0].cpu().numpy()
    p_resized = cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    binm = (p_resized > 0.5).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 1: return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    bw = int(stats[best, cv2.CC_STAT_WIDTH])
    bh = int(stats[best, cv2.CC_STAT_HEIGHT])
    score = float(p_resized[y:y+bh, x:x+bw].mean())
    return (x, y, x + bw, y + bh), score


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(CKPT, map_location=device)
    model.load_state_dict(s.get("model_state", s), strict=False); model.eval()

    decoder = SegDecoder().to(device)
    seg_state = torch.load(SEG, map_location=device)
    decoder.load_state_dict(seg_state["decoder_state"]); decoder.eval()
    print(f"Decoder loaded — val IoU at save: {seg_state.get('val_iou', '?'):.3f}")

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
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
        results = []
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
                feats = model.image_encoder.resnet_backbone(x)
                seg = decoder(feats)
            pred_bb, conf = bbox_from_seg(seg, pim.height, pim.width)
            results.append((pred_bb, gt_bb, polyp_prob * conf, polyp_prob))

        tp = sum(1 for pb, gb, _, pp in results
                 if (gb is not None and pp >= 0.5 and bbox_iou(pb, gb) >= 0.5))
        n_gt = sum(1 for _, gb, _, _ in results if gb is not None)
        sens = tp / max(1, n_gt)
        n_fp = sum(1 for pb, gb, _, pp in results
                   if pp >= 0.5 and (gb is None or bbox_iou(pb, gb) < 0.5))
        fppi = n_fp / max(1, len(results))
        # mAP @ 0.5
        sorted_r = sorted(results, key=lambda r: -r[2])
        tps = fps = 0; prs = []
        for pb, gb, _, pp in sorted_r:
            if gb is not None and bbox_iou(pb, gb) >= 0.5: tps += 1
            else: fps += 1
            prec = tps / max(1, tps + fps); rec = tps / max(1, n_gt)
            prs.append((rec, prec))
        prs.sort()
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            ps = [p for rr, p in prs if rr >= r]
            ap += max(ps) if ps else 0
        ap /= 11

        report[ds_name] = {
            "n_images":          len(results),
            "sensitivity_iou50": sens,
            "fppi_at_p50":       fppi,
            "mAP_iou50":         float(ap),
        }
        print(f"  ✓ {ds_name}:  sens@IoU50 = {sens:.3f}   FPPI = {fppi:.3f}   mAP = {ap:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*72}\nCADe METRICS — segmentation-decoder bboxes (v2)\n{'='*72}")
    print(f"{'Dataset':22s} {'sens@IoU50':>10} {'FPPI':>8} {'mAP@50':>8} {'n':>5}")
    print("-"*72)
    for k, v in report.items():
        print(f"{k:22s} {v['sensitivity_iou50']:10.3f} {v['fppi_at_p50']:8.3f} "
              f"{v['mAP_iou50']:8.3f} {v['n_images']:5d}")
    print("="*72)


if __name__ == "__main__":
    main()
