"""ColonAI — side-by-side GradCAM overlays  v1 vs v2.

For each dataset, picks 2-3 representative images and renders a 4-panel
comparison: input | ground-truth mask | v1 GradCAM | v2 GradCAM.

Saves to outputs/unified_multimodal_v2/figures/overlays/<dataset>/<n>.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from src.agents.unified_image_agent import GradCAMPlusPlus
from transformers import AutoTokenizer

CKPT_V1 = "outputs/unified_multimodal/checkpoints/best_model.pth"
CKPT_V2 = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
OUT_DIR = Path("outputs/unified_multimodal_v2/figures/overlays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG", 3),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB", 3),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB", 2),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300", 2),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir", 2),
]

mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load_model(ckpt, device):
    m = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(ckpt, map_location=device)
    m.load_state_dict(s.get("model_state", s), strict=False); m.eval()
    return m


def preprocess(pil):
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])(pil).unsqueeze(0)


def cam_for(model, cam_ex, x, ids, msk, tab):
    cam = cam_ex.generate(image=x.detach().requires_grad_(True), class_idx=0,
                          input_ids=ids, attention_mask=msk, tabular=tab)
    return cam.astype(np.float32) if (cam is not None and cam.size >= 4) else None


def overlay(img_rgb, cam, alpha=0.45):
    h, w = img_rgb.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min() + 1e-6)
    cmap = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, cmap, alpha, 0)


def iou(pred, gt):
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union > 1 else 0.0


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")
    print("Loading v1 …"); m1 = load_model(CKPT_V1, device); ex1 = GradCAMPlusPlus(m1, m1.get_image_target_layer())
    print("Loading v2 …"); m2 = load_model(CKPT_V2, device); ex2 = GradCAMPlusPlus(m2, m2.get_image_target_layer())

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids = enc["input_ids"].to(device); msk = enc["attention_mask"].to(device)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=device)

    for ds_name, root, n_samples in DATASETS:
        img_dir = Path(root) / "images"; msk_dir = Path(root) / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            print(f"⚠ {ds_name}: dirs missing"); continue
        files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".tif",".tiff",".png",".jpg",".jpeg",".bmp")])
        if not files: continue
        out_ds = OUT_DIR / ds_name; out_ds.mkdir(parents=True, exist_ok=True)
        # Spread picks across corpus
        idxs = np.linspace(0, len(files)-1, n_samples, dtype=int)
        print(f"\n{ds_name}: {n_samples} samples")
        for k, idx in enumerate(idxs):
            img_path = files[int(idx)]
            mpath = None
            for e in (".png",".jpg",".jpeg",".tif",".tiff",".bmp"):
                c = msk_dir / (img_path.stem + e)
                if c.exists(): mpath = c; break
            if mpath is None: continue
            try:
                pim = Image.open(img_path).convert("RGB")
                gt  = np.array(Image.open(mpath).convert("L")) > 127
            except Exception: continue
            x = preprocess(pim).to(device)
            cam1 = cam_for(m1, ex1, x, ids, msk, tab)
            cam2 = cam_for(m2, ex2, x, ids, msk, tab)
            if cam1 is None or cam2 is None: continue
            img_rgb = np.array(pim)
            ov1 = overlay(img_rgb, cam1)
            ov2 = overlay(img_rgb, cam2)
            # IoU at 75th-percentile threshold
            def iou_at(cam):
                cr = cv2.resize(cam, (pim.width, pim.height),
                                interpolation=cv2.INTER_LINEAR)
                return iou(cr >= float(np.quantile(cr, 0.75)), gt)
            i1 = iou_at(cam1); i2 = iou_at(cam2)

            fig, axes = plt.subplots(1, 4, figsize=(13, 4))
            axes[0].imshow(img_rgb); axes[0].set_title("Input")
            axes[1].imshow(img_rgb); axes[1].imshow(gt, alpha=0.45, cmap="Greens")
            axes[1].set_title("Ground truth mask")
            axes[2].imshow(ov1); axes[2].set_title(f"v1 GradCAM\nIoU = {i1:.3f}")
            axes[3].imshow(ov2); axes[3].set_title(f"v2 (mask-aware) GradCAM\nIoU = {i2:.3f}",
                                                    color=("#16a34a" if i2 > i1 else "#dc2626"))
            for a in axes: a.set_xticks([]); a.set_yticks([])
            plt.suptitle(f"{ds_name}  ·  {img_path.name}", fontsize=10, y=0.99)
            plt.tight_layout()
            out_png = out_ds / f"overlay_{k+1:02d}_{img_path.stem}.png"
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {out_png.name}  (v1 IoU {i1:.3f}  →  v2 IoU {i2:.3f}  Δ {i2-i1:+.3f})")
    print(f"\nAll overlays saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
