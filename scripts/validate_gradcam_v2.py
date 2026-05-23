"""Cross-vendor GradCAM IoU/Dice — v2 (new mask-supervised checkpoint).

Same logic as validate_gradcam_cross_vendor.py but points at the deploy-grade
checkpoint and writes the report to a separate file so we can compare
BEFORE vs AFTER.

Run:
    python3 scripts/validate_gradcam_v2.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from src.agents.unified_image_agent import GradCAMPlusPlus
from transformers import AutoTokenizer


CHECKPOINT_V1 = "outputs/unified_multimodal/checkpoints/best_model.pth"
CHECKPOINT_V2 = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
BERT = "dmis-lab/biobert-base-cased-v1.2"
POLYP_CLASS_IDX = 0

DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir"),
]


def preprocess(pil_img):
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img).unsqueeze(0)


def iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 1 else 0.0


def dice(a, b):
    inter = float(np.logical_and(a, b).sum())
    s = float(a.sum() + b.sum())
    return 2 * inter / s if s > 1 else 0.0


def evaluate(checkpoint_path: str, label: str, device) -> dict:
    print(f"\n{'='*60}\n{label}\n  ckpt: {checkpoint_path}\n{'='*60}")
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    cam_ex = GradCAMPlusPlus(model, model.get_image_target_layer())

    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64,
              return_tensors="pt")
    ids = enc["input_ids"].to(device)
    msk = enc["attention_mask"].to(device)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=device)

    report = {}
    for name, root in DATASETS:
        root = Path(root)
        img_dir, msk_dir = root / "images", root / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            print(f"⚠ {name}: dirs missing"); continue
        files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg",".jpeg",".png",".tif",".tiff",".bmp")])
        ious, dices, n_polyp = [], [], 0
        for img_path in tqdm(files, desc=name, leave=False):
            mp = None
            for e in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                c = msk_dir / (img_path.stem + e)
                if c.exists(): mp = c; break
            if mp is None: continue
            try:
                pim = Image.open(img_path).convert("RGB")
                gtm = np.array(Image.open(mp).convert("L"))
            except Exception: continue

            x = preprocess(pim).to(device)
            with torch.no_grad():
                probs = F.softmax(model(x, ids, msk, tab)["pathology"], dim=-1)[0]
                if int(probs.argmax()) == POLYP_CLASS_IDX:
                    n_polyp += 1

            cam = cam_ex.generate(image=x.detach().requires_grad_(True),
                                  class_idx=POLYP_CLASS_IDX,
                                  input_ids=ids, attention_mask=msk, tabular=tab)
            if cam is None or cam.size < 4: continue
            cam_r = cv2.resize(cam.astype(np.float32),
                               (pim.width, pim.height),
                               interpolation=cv2.INTER_LINEAR)
            thr = float(np.quantile(cam_r, 0.75))
            pred_m = cam_r >= thr
            gt_m   = gtm > 127
            ious.append(iou(pred_m, gt_m))
            dices.append(dice(pred_m, gt_m))

        if ious:
            report[name] = {
                "n_images":       len(ious),
                "polyp_class_pred": n_polyp,
                "mean_iou":       float(np.mean(ious)),
                "median_iou":     float(np.median(ious)),
                "mean_dice":      float(np.mean(dices)),
                "median_dice":    float(np.median(dices)),
            }
            print(f"  {name:20s}  IoU={report[name]['mean_iou']:.3f}  "
                  f"Dice={report[name]['mean_dice']:.3f}  "
                  f"polyp-pred={n_polyp}/{len(ious)}")

    cam_ex.handles[0].remove() if hasattr(cam_ex, "handles") else None
    del model; torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    return report


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    v1 = evaluate(CHECKPOINT_V1, "BEFORE (v1)", device)
    v2 = evaluate(CHECKPOINT_V2, "AFTER  (v2 mask-aware)", device)

    out = Path("outputs/unified_multimodal_v2/cross_vendor_gradcam_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"v1_before": v1, "v2_after": v2}, indent=2))

    print("\n" + "="*72)
    print(f"{'Dataset':22s} {'v1 IoU':>9} {'v2 IoU':>9} {'Δ':>8}   "
          f"{'v1 Dice':>9} {'v2 Dice':>9}")
    print("-"*72)
    for k in v1:
        if k in v2:
            d_iou = v2[k]["mean_iou"] - v1[k]["mean_iou"]
            sign = "+" if d_iou >= 0 else ""
            print(f"{k:22s} {v1[k]['mean_iou']:9.3f} {v2[k]['mean_iou']:9.3f} "
                  f"{sign}{d_iou:7.3f}   "
                  f"{v1[k]['mean_dice']:9.3f} {v2[k]['mean_dice']:9.3f}")
    print("="*72)
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
