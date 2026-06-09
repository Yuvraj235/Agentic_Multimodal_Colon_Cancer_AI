"""Does multi-centre PolypGen data close the honest cross-vendor gap?

Olympus-only training gave ETIS (Pentax) held-out IoU = 0.453. This adds the
6-centre PolypGen dataset to training and RE-measures on the SAME fully-held-out
ETIS test set. If cross-vendor IoU rises, the multi-centre data genuinely improved
generalisation (the honest fix for the measured gap).

Train  : Olympus sets (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, Kvasir-test)
         + PolypGen C1-C6 (multi-centre, masks named {stem}_mask.jpg)
Held out: ALL of ETIS-Larib (Pentax) — never trained on.
Saves seg_head_polypgen.pth + seg_polypgen_metrics.json (does not touch live seg_head.pth).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from scripts.train_segmentation_head import SegDecoder, dice_loss

CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
SEG_OUT = Path("outputs/unified_multimodal_v2/seg_head_polypgen.pth")
REPORT = Path("outputs/unified_multimodal_v2/seg_polypgen_metrics.json")

OLYMPUS = [
    ("Kvasir-SEG",   "data/raw/kvasir-seg/Kvasir-SEG/images",                       "data/raw/kvasir-seg/Kvasir-SEG/masks"),
    ("CVC-ClinicDB", "data/raw/CVC-ClinicDB/PNG/Original",                          "data/raw/CVC-ClinicDB/PNG/Ground Truth"),
    ("CVC-ColonDB",  "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/images", "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/masks"),
    ("CVC-300",      "data/raw/test_polyp_datasets/TestDataset/CVC-300/images",     "data/raw/test_polyp_datasets/TestDataset/CVC-300/masks"),
    ("Kvasir-test",  "data/raw/test_polyp_datasets/TestDataset/Kvasir/images",      "data/raw/test_polyp_datasets/TestDataset/Kvasir/masks"),
]
POLYPGEN = [(f"PolypGen-C{c}", f"data/raw/polypgen/data_C{c}/images_C{c}",
             f"data/raw/polypgen/data_C{c}/masks_C{c}") for c in range(1, 7)]
ETIS_HELD = [("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images",
              "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/masks")]
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _find_mask(img_path: Path, mask_dir: Path):
    stem = img_path.stem
    for e in EXTS:
        for cand in (mask_dir / (stem + e), mask_dir / (stem + "_mask" + e)):
            if cand.exists():
                return cand
    return None


class SegSet(Dataset):
    def __init__(self, dataset_list, augment):
        self.samples = []
        for name, ir, mr in dataset_list:
            ip, mp = Path(ir), Path(mr)
            if not ip.exists() or not mp.exists():
                print(f"    · {name}: MISSING"); continue
            n = 0
            for f in sorted(ip.iterdir()):
                if f.suffix.lower() not in EXTS or "_mask" in f.stem:
                    continue
                m = _find_mask(f, mp)
                if m:
                    self.samples.append((str(f), str(m), name)); n += 1
            print(f"    · {name}: {n}")
        self.augment = augment
        if augment:
            self.tf = T.Compose([T.Resize((256, 256)), T.RandomCrop((224, 224)),
                                 T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
                                 T.ColorJitter(0.2, 0.2, 0.15), T.ToTensor(), _NORM])
        else:
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), _NORM])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ip, mp, ds = self.samples[i]
        try:
            pim = Image.open(ip).convert("RGB"); pms = Image.open(mp).convert("L")
        except Exception:
            return self.__getitem__((i + 1) % len(self.samples))
        if self.augment:
            seed = torch.randint(0, 2**31-1, (1,)).item()
            torch.manual_seed(seed); img = self.tf(pim)
            torch.manual_seed(seed)
            m = T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST)(pms)
            m = T.RandomCrop((224, 224))(m); m = T.RandomHorizontalFlip(0.5)(m); m = T.RandomVerticalFlip(0.3)(m)
            msk = (T.ToTensor()(m) > 0.5).float()
        else:
            img = self.tf(pim)
            m = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)(pms)
            msk = (T.ToTensor()(m) > 0.5).float()
        return {"image": img, "mask": msk, "dataset": ds}


@torch.no_grad()
def eval_iou(dec, backbone, dl, device):
    dec.eval(); ious = []
    for b in dl:
        img = b["image"].to(device); msk = b["mask"].to(device)
        p = (torch.sigmoid(dec(backbone(img))) > 0.5).float()
        inter = (p * msk).sum(dim=(2, 3)); union = p.sum(dim=(2, 3)) + msk.sum(dim=(2, 3)) - inter
        dd = p.sum(dim=(2, 3)) + msk.sum(dim=(2, 3))
        for j in range(img.size(0)):
            u = float(union[j]); ious.append((float(inter[j]/u) if u > 1e-6 else 0.0,
                                              float(2*inter[j]/dd[j]) if float(dd[j]) > 1e-6 else 0.0))
    iarr = np.array([x[0] for x in ious]); darr = np.array([x[1] for x in ious])
    return {"n": len(ious), "mean_iou": float(iarr.mean()), "mean_dice": float(darr.mean()),
            "sens_at_iou0.5": float((iarr >= 0.5).mean())}


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {device}")
    model = UnifiedMultiModalTransformer(n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(CKPT, map_location=device); model.load_state_dict(s.get("model_state", s), strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    backbone = model.image_encoder.resnet_backbone

    print("\n[1/3] TRAIN datasets (Olympus + PolypGen multi-centre):")
    train_full = SegSet(OLYMPUS + POLYPGEN, augment=True)
    val_full = SegSet(OLYMPUS + POLYPGEN, augment=False)
    n_tr = int(0.9 * len(train_full))
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(42))
    tr_idx, va_idx = perm[:n_tr].tolist(), perm[n_tr:].tolist()
    tr_dl = DataLoader(Subset(train_full, tr_idx), batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(Subset(val_full, va_idx), batch_size=16, shuffle=False, num_workers=2)
    print("\n[1b] HELD-OUT ETIS (Pentax):")
    etis_dl = DataLoader(SegSet(ETIS_HELD, augment=False), batch_size=16, shuffle=False, num_workers=2)
    print(f"  train {len(tr_idx)} · val {len(va_idx)} · ETIS held-out {len(etis_dl.dataset)}")

    print("\n[2/3] Training decoder …")
    dec = SegDecoder().to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5*len(tr_dl))
    best = 0.0
    for ep in range(5):
        dec.train()
        for step, b in enumerate(tr_dl):
            img = b["image"].to(device); msk = b["mask"].to(device)
            with torch.no_grad(): feats = backbone(img)
            loss = F.binary_cross_entropy_with_logits(dec(feats), msk) + dice_loss(dec(feats), msk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step(); sched.step()
            if step % 40 == 0:
                print(f"  ep{ep+1}/5 step{step:3d}/{len(tr_dl)} loss={loss.item():.4f}")
        v = eval_iou(dec, backbone, va_dl, device)
        print(f"  ✓ ep{ep+1} val IoU={v['mean_iou']:.3f}")
        if v["mean_iou"] > best:
            best = v["mean_iou"]
            torch.save({"decoder_state": dec.state_dict(), "val_iou": best, "epoch": ep+1}, SEG_OUT)

    print("\n[3/3] HONEST held-out ETIS (Pentax) eval …")
    dec.load_state_dict(torch.load(SEG_OUT)["decoder_state"])
    etis = eval_iou(dec, backbone, etis_dl, device); olymp = eval_iou(dec, backbone, va_dl, device)
    print(f"\n  >>> ETIS (Pentax) HELD-OUT:  IoU={etis['mean_iou']:.3f}  Dice={etis['mean_dice']:.3f}  "
          f"sens@IoU0.5={etis['sens_at_iou0.5']:.3f}  (n={etis['n']})")
    print(f"      vs Olympus-only baseline ETIS IoU 0.453")
    REPORT.write_text(json.dumps({
        "method": "trained Olympus + PolypGen (6 centres); ETIS-Larib (Pentax) fully HELD OUT",
        "etis_pentax_heldout": etis, "val_multicentre": olymp,
        "baseline_olympus_only_etis_iou": 0.453, "decoder": str(SEG_OUT),
    }, indent=2))
    print(f"\nSaved → {SEG_OUT}\n      metrics → {REPORT}")


if __name__ == "__main__":
    main()
