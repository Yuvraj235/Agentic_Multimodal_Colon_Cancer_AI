"""Cross-vendor GradCAM validation against pixel-level polyp masks.

For each image in ETIS-LaribPolypDB (Pentax), CVC-ColonDB, CVC-300, and
Kvasir-SEG, this script:
  1. Loads the colonoscopy image
  2. Runs it through the trained UnifiedMultiModalTransformer
  3. Extracts the GradCAM++ heatmap for the predicted class
  4. Thresholds at the 75th percentile to produce a binary saliency mask
  5. Compares to the ground-truth polyp mask via Intersection-over-Union (IoU)

Outputs a JSON report with per-dataset IoU statistics — a defensible
number for the paper's external-validation discussion.

Run from project root:
  python3 scripts/validate_gradcam_cross_vendor.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from src.agents.unified_image_agent import GradCAMPlusPlus
from transformers import AutoTokenizer


CHECKPOINT = "outputs/unified_multimodal/checkpoints/best_model.pth"
BERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"
POLYP_CLASS_IDX = 0   # 'polyps' in CLASS_NAMES

DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir"),
]


def preprocess(pil_img):
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img).unsqueeze(0)


def iou(pred_mask, gt_mask):
    """Binary IoU between two HxW boolean arrays."""
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    union = float(np.logical_or (pred_mask, gt_mask).sum())
    return inter / union if union > 1 else 0.0


def dice(pred_mask, gt_mask):
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    s = float(pred_mask.sum() + gt_mask.sum())
    return 2 * inter / s if s > 1 else 0.0


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    # Load model — checkpoint was trained with 5 pathology classes
    print(f"Loading checkpoint {CHECKPOINT} …")
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES,
        n_classes=5,
    ).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded — {len(state)} keys (missing: {len(missing)}, unexpected: {len(unexpected)})")
    model.eval()

    # GradCAM extractor
    tgt = model.get_image_target_layer()
    cam_extractor = GradCAMPlusPlus(model, tgt)

    # Tokenise a generic clinical text
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64,
              return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    tab       = torch.zeros((1, N_TABULAR_FEATURES), device=device)

    # Run validation
    report = {}
    for name, root in DATASETS:
        root = Path(root)
        if not root.exists():
            print(f"⚠ {name}: not found at {root}")
            continue
        img_dir = root / "images"
        msk_dir = root / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            print(f"⚠ {name}: no images/masks folder at {root}")
            continue

        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".tif",".tiff",".bmp")])
        if not img_files:
            print(f"⚠ {name}: no images in {img_dir}")
            continue

        ious, dices, top_attention_polyp = [], [], 0
        n_polyp_pred = 0
        print(f"\n=== {name} ({len(img_files)} images) ===")
        for img_path in tqdm(img_files, desc=name, leave=False):
            # Find matching mask (same stem, possibly different extension)
            mask_path = None
            stem = img_path.stem
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                cand = msk_dir / (stem + ext)
                if cand.exists():
                    mask_path = cand; break
            if not mask_path:
                continue

            # Load image + mask
            try:
                pil_img = Image.open(img_path).convert("RGB")
                msk     = np.array(Image.open(mask_path).convert("L"))
            except Exception:
                continue

            x = preprocess(pil_img).to(device)

            # Forward pass → predicted class
            with torch.no_grad():
                out = model(image=x, input_ids=input_ids,
                            attention_mask=attn_mask, tabular=tab)
                probs = F.softmax(out["pathology"], dim=-1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                if pred_idx == POLYP_CLASS_IDX:
                    n_polyp_pred += 1

            # Compute GradCAM for the polyp class (or predicted class)
            cam = cam_extractor.generate(
                image=x.detach().requires_grad_(True),
                class_idx=POLYP_CLASS_IDX,
                input_ids=input_ids, attention_mask=attn_mask, tabular=tab,
            )
            if cam is None or cam.size < 4:
                continue
            # Resize CAM to image size
            cam_resized = cv2.resize(cam.astype(np.float32),
                                     (pil_img.width, pil_img.height),
                                     interpolation=cv2.INTER_LINEAR)
            # Binarise at the 75th percentile
            thr = float(np.quantile(cam_resized, 0.75))
            pred_mask = (cam_resized >= thr)
            gt_mask   = (msk > 127)  # GT masks are 0/255

            ious.append(iou(pred_mask, gt_mask))
            dices.append(dice(pred_mask, gt_mask))

        if ious:
            stats = {
                "n_images":          len(ious),
                "polyp_class_pred":  n_polyp_pred,
                "mean_iou":          float(np.mean(ious)),
                "median_iou":        float(np.median(ious)),
                "mean_dice":         float(np.mean(dices)),
                "median_dice":       float(np.median(dices)),
            }
            print(f"  ✓ {name}: mean IoU = {stats['mean_iou']:.3f}, "
                  f"mean Dice = {stats['mean_dice']:.3f}, "
                  f"polyp-class predicted on {n_polyp_pred}/{len(ious)} images")
            report[name] = stats

    # Save report
    out_path = Path("outputs/unified_multimodal/cross_vendor_gradcam_iou.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n→ Saved report to {out_path}")

    # Overall summary
    print("\n" + "=" * 60)
    print("CROSS-VENDOR GRADCAM VALIDATION SUMMARY")
    print("=" * 60)
    for name, s in report.items():
        print(f"  {name:25s}  IoU={s['mean_iou']:.3f}  Dice={s['mean_dice']:.3f}  "
              f"({s['polyp_class_pred']}/{s['n_images']} predicted polyp)")
    if report:
        all_ious  = [s["mean_iou"]  for s in report.values()]
        all_dices = [s["mean_dice"] for s in report.values()]
        print(f"\n  Overall mean IoU  : {np.mean(all_ious):.3f}")
        print(f"  Overall mean Dice : {np.mean(all_dices):.3f}")


if __name__ == "__main__":
    main()
