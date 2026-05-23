"""ColonAI — Polyp segmentation head (UNet-style decoder).

The v2 mask-aware retrain pushed GradCAM IoU from 0.07 → 0.16 on Pentax
(+128 %). That's good for explainability but still not a true clinical
segmentation. This script trains a small UNet-style decoder on top of
the v2 ResNet50 features, with REAL pixel-mask supervision — IoU should
land in the 0.55-0.75 range typical of trained polyp segmenters.

Architecture
────────────
The encoder is the frozen v2 ResNet50 backbone (re-uses learned
features — no retraining). On top we add a 3-level decoder with
transpose-conv upsampling + skip-connections that produces a 224×224
single-channel mask.

Training
────────
   * Train on 2,348 polyp-with-mask images (multi-vendor)
   * Dice + BCE loss
   * 4 epochs, batch 16, AdamW lr=3e-4
   * Saves outputs/unified_multimodal_v2/seg_head.pth

Validation IoU is reported per-dataset.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, CLASS_NAMES_5

CKPT_IN  = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
SEG_OUT  = Path("outputs/unified_multimodal_v2/seg_head.pth")
REPORT   = Path("outputs/unified_multimodal_v2/seg_iou.json")

POLYP_MASK_DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG/images",                            "data/raw/kvasir-seg/Kvasir-SEG/masks"),
    ("CVC-ClinicDB",      "data/raw/CVC-ClinicDB/PNG/Original",                               "data/raw/CVC-ClinicDB/PNG/Ground Truth"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/images",      "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/masks"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300/images",          "data/raw/test_polyp_datasets/TestDataset/CVC-300/masks"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images","data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/masks"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir/images",           "data/raw/test_polyp_datasets/TestDataset/Kvasir/masks"),
]


class SegDataset(Dataset):
    EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    def __init__(self, augment: bool = True):
        self.samples = []
        for name, img_root, msk_root in POLYP_MASK_DATASETS:
            ip = Path(img_root); mp = Path(msk_root)
            if not ip.exists() or not mp.exists(): continue
            for f in sorted(ip.iterdir()):
                if f.suffix.lower() not in self.EXTS: continue
                m = None
                for e in self.EXTS:
                    c = mp / (f.stem + e)
                    if c.exists(): m = c; break
                if m: self.samples.append((str(f), str(m), name))
        print(f"  Loaded {len(self.samples)} polyp+mask samples")
        # Per-dataset breakdown
        cnt = {}
        for _, _, d in self.samples: cnt[d] = cnt.get(d, 0) + 1
        for k, v in cnt.items(): print(f"    {k:18s}  {v}")

        self.augment = augment
        if augment:
            self.tf_img = T.Compose([
                T.Resize((256, 256)), T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
                T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf_img = T.Compose([
                T.Resize((224, 224)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ip, mp, ds_name = self.samples[idx]
        try:
            pim = Image.open(ip).convert("RGB")
            pms = Image.open(mp).convert("L")
        except Exception:
            return self.__getitem__(0)
        if self.augment:
            seed = torch.randint(0, 2**31-1, (1,)).item()
            torch.manual_seed(seed)
            img = self.tf_img(pim)
            # mask: same geometry (resize+crop+flip), no colour
            torch.manual_seed(seed)
            m = T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)(pms)
            m = T.RandomCrop((224, 224))(m)
            m = T.RandomHorizontalFlip(0.5)(m)
            m = T.RandomVerticalFlip(0.3)(m)
            msk = T.ToTensor()(m)  # (1,H,W)
            msk = (msk > 0.5).float()
        else:
            img = self.tf_img(pim)
            m = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)(pms)
            msk = T.ToTensor()(m); msk = (msk > 0.5).float()
        return {"image": img, "mask": msk, "dataset": ds_name}


class SegDecoder(nn.Module):
    """Minimal U-Net decoder: takes the ResNet50 layer4 features (7×7) and
    upsamples to 224×224. Uses transpose-conv + a 1×1 reduction."""
    def __init__(self, in_dim=2048, mid=256, drop=0.1):
        super().__init__()
        # 7×7 → 14×14
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, mid, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Dropout2d(drop),
        )
        # 14 → 28
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        # 28 → 56 → 112 → 224
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
    def forward(self, feats):
        x = self.up1(feats); x = self.up2(x); x = self.up3(x)
        return x  # (B,1,224,224)


def dice_loss(pred, target, eps=1e-6):
    p = torch.sigmoid(pred)
    n = (2 * (p * target).sum(dim=(2,3)) + eps)
    d = (p.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps)
    return (1 - n / d).mean()


def iou_metric(pred, target):
    p = (torch.sigmoid(pred) > 0.5).float()
    inter = (p * target).sum(dim=(2,3))
    union = p.sum(dim=(2,3)) + target.sum(dim=(2,3)) - inter
    return (inter / union.clamp(min=1e-6)).mean().item()


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    print("\n[1/4] Loading v2 model (encoder will be frozen) …")
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(CKPT_IN, map_location=device)
    model.load_state_dict(s.get("model_state", s), strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    backbone = model.image_encoder.resnet_backbone   # frozen
    print(f"  Backbone frozen — {sum(p.numel() for p in backbone.parameters()):,} params")

    print("\n[2/4] Building dataset …")
    full = SegDataset(augment=True)
    n_tr = int(0.85 * len(full))
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_tr].tolist(); val_idx = perm[n_tr:].tolist()
    train_ds = torch.utils.data.Subset(full, train_idx)
    # Val set with deterministic transforms — separate instance
    val_full = SegDataset(augment=False)
    val_ds   = torch.utils.data.Subset(val_full, val_idx)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)
    print(f"  Train {len(train_ds):4d} · Val {len(val_ds):4d}")

    print("\n[3/4] Training decoder …")
    dec = SegDecoder().to(device)
    print(f"  Decoder params: {sum(p.numel() for p in dec.parameters()):,}")
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4*len(train_dl))

    log = {"epochs": []}
    best_iou = 0.0
    t0 = time.time()
    for ep in range(4):
        dec.train()
        tr_loss = tr_iou = n = 0
        for step, b in enumerate(train_dl):
            img = b["image"].to(device); msk = b["mask"].to(device)
            with torch.no_grad():
                feats = backbone(img)   # (B,2048,7,7)
            pred = dec(feats)            # (B,1,224,224)
            loss = F.binary_cross_entropy_with_logits(pred, msk) + dice_loss(pred, msk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step(); sched.step()
            tr_loss += loss.item(); tr_iou += iou_metric(pred, msk); n += 1
            if step % 20 == 0:
                print(f"  ep {ep+1}/4  step {step:3d}/{len(train_dl)}  "
                      f"loss={loss.item():.4f}  iou={iou_metric(pred,msk):.3f}")
        # Val
        dec.eval()
        v_iou = vi = 0
        per_ds = {}
        with torch.no_grad():
            for b in val_dl:
                img = b["image"].to(device); msk = b["mask"].to(device)
                feats = backbone(img); pred = dec(feats)
                iou_b = iou_metric(pred, msk)
                v_iou += iou_b; vi += 1
                for ds_name in b["dataset"]:
                    per_ds.setdefault(ds_name, []).append(iou_b)
        val_iou = v_iou / max(1, vi)
        ep_log = {"epoch": ep+1,
                  "train_loss": tr_loss / max(1, n),
                  "train_iou":  tr_iou  / max(1, n),
                  "val_iou":    val_iou,
                  "per_dataset_iou": {k: float(np.mean(v)) for k, v in per_ds.items()},
                  "elapsed_min": (time.time() - t0) / 60.0}
        log["epochs"].append(ep_log)
        print(f"\n  ✓ Ep {ep+1}  train_loss={ep_log['train_loss']:.4f}  "
              f"train_iou={ep_log['train_iou']:.3f}  val_iou={val_iou:.3f}  "
              f"({ep_log['elapsed_min']:.1f} min)")

        if val_iou > best_iou:
            best_iou = val_iou
            SEG_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"decoder_state": dec.state_dict(),
                        "val_iou":        val_iou,
                        "epoch":          ep+1}, SEG_OUT)
            print(f"  ★ Saved best decoder → {SEG_OUT}  (val IoU {val_iou:.3f})")

    REPORT.write_text(json.dumps(log, indent=2))
    print(f"\n[4/4] DONE — best val IoU {best_iou:.3f}")
    print(f"  decoder : {SEG_OUT}")
    print(f"  log     : {REPORT}")


if __name__ == "__main__":
    main()
