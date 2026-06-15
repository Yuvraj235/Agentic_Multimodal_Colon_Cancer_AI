"""CARE rectal-cancer CT tumor segmentation — Colab + Google Drive (free).

Matches the OFFICIAL CARE format (from kanydao/U-SAM dataset/rectum_dataloader.py):
  root/train/train_npz/*.npz  (+ train_bbox.csv)
  root/test/test_npz/*.npz    (+ test_bbox.csv)
Each .npz holds  image  (2D CT slice) and  label  (mask: 0=bg, 1/2 = tissue;
the official loader caps mask>2→2 with num_classes=2). The TUMOUR is label 2
(label 1 ≈ normal rectum) — the INSPECT step prints the values so you can confirm.

Trains a U-Net (trainable encoder) to segment the tumour, using CARE's OWN
train/test split = the honest held-out number. CC BY-NC 4.0 (research use).

Colab:
  from google.colab import drive; drive.mount('/content/drive')
  !pip -q install segmentation-models-pytorch albumentations
  !python train_care_ct_colab.py --root "/content/drive/MyDrive/CARE" --inspect-only
  # confirm tumour label, then drop --inspect-only to train.
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np

TUMOR_LABELS = {2}          # confirm from INSPECT (2 = cancerous rectum; 1 ≈ normal)


def _find_npz(root: Path):
    def pick(split):
        a = sorted(glob.glob(str(root / "**" / f"{split}_npz" / "*.npz"), recursive=True))
        if not a:  # fallback: any .npz whose path mentions the split
            a = sorted(p for p in glob.glob(str(root / "**" / "*.npz"), recursive=True)
                       if split in p.lower())
        return a
    return pick("train"), pick("test")


def _img_to_uint8_3ch(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / (hi - lo + 1e-6) * 255.0 if hi > lo else np.zeros_like(img)
    return np.stack([img.astype(np.uint8)] * 3, axis=-1)


def _mask_tumor(label):
    return np.isin(np.asarray(label), list(TUMOR_LABELS)).astype("float32")


def inspect(root: Path):
    print(f"\n=== INSPECT {root} ===")
    if not root.exists():
        raise SystemExit(f"root not found: {root}")
    tr, te = _find_npz(root)
    print(f"  train_npz files: {len(tr)} | test_npz files: {len(te)}")
    sample = (tr or te)[:1]
    if not sample:
        print("  ⚠ no .npz found — check --root (should hold train/train_npz + test/test_npz)")
        return tr, te
    z = np.load(sample[0])
    print(f"  sample npz: {Path(sample[0]).name}  keys={list(z.files)}")
    if "image" in z.files:
        im = z["image"]; print(f"    image: shape={im.shape} dtype={im.dtype} range=[{float(im.min()):.1f},{float(im.max()):.1f}]")
    if "label" in z.files:
        lb = z["label"]; print(f"    label: shape={lb.shape} unique={sorted(np.unique(lb).tolist())[:8]}  → TUMOR_LABELS={TUMOR_LABELS}")
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--out", default="/content/drive/MyDrive/CARE/care_ct_seg.pt")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    tr_files, te_files = inspect(root)
    if args.inspect_only:
        return
    if not tr_files:
        raise SystemExit("No train .npz found — fix --root, then re-run.")

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} · encoder={args.encoder} · train {len(tr_files)} · test {len(te_files)}")

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    aug = A.Compose([A.Resize(args.imgsz, args.imgsz), A.CLAHE(p=1.0),
                     A.HorizontalFlip(p=.5), A.ShiftScaleRotate(0.05, 0.1, 12, p=.5),
                     A.Normalize(MEAN, STD), ToTensorV2()])
    noa = A.Compose([A.Resize(args.imgsz, args.imgsz), A.CLAHE(p=1.0),
                     A.Normalize(MEAN, STD), ToTensorV2()])

    class DS(Dataset):
        def __init__(s, files, t): s.files, s.t = files, t
        def __len__(s): return len(s.files)
        def __getitem__(s, i):
            z = np.load(s.files[i])
            img = _img_to_uint8_3ch(z["image"]); msk = _mask_tumor(z["label"])
            o = s.t(image=img, mask=msk); return o["image"], o["mask"].unsqueeze(0).float()

    trdl = DataLoader(DS(tr_files, aug), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    tedl = DataLoader(DS(te_files, noa), batch_size=args.batch, shuffle=False, num_workers=2)
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    if args.resume and Path(args.out).exists():
        model.load_state_dict(torch.load(args.out, map_location=dev)["state_dict"]); print("resumed")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dice = smp.losses.DiceLoss(mode="binary"); bce = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def ev():
        model.eval(); ious, dices = [], []
        for x, y in tedl:
            p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
            inter = (p*y).sum((1,2,3)); uni = p.sum((1,2,3))+y.sum((1,2,3))-inter; den = p.sum((1,2,3))+y.sum((1,2,3))
            for j in range(x.size(0)):
                u = float(uni[j]); d = float(den[j])
                ious.append(float(inter[j]/u) if u>1e-6 else 0.0)
                dices.append(float(2*inter[j]/d) if d>1e-6 else 0.0)
        return np.array(ious), np.array(dices)

    def ci(a, B=2000):
        if a.size == 0: return (0,0,0)
        r = np.random.default_rng(0); m = a[r.integers(0,a.size,(B,a.size))].mean(1)
        return float(a.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = 0.0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in trdl:
            x, y = x.to(dev), y.to(dev)
            loss = dice(model(x), y) + bce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        vi, _ = ev(); iou = float(vi.mean()); print(f"  ep{ep+1}/{args.epochs} test_IoU={iou:.4f}")
        if iou > best:
            best = iou
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "iou": best, "imgsz": args.imgsz, "tumor_labels": list(TUMOR_LABELS)}, args.out)

    model.load_state_dict(torch.load(args.out, map_location=dev)["state_dict"])
    ei, ed = ev(); im_, ilo, ihi = ci(ei); dm_, dlo, dhi = ci(ed)
    rep = {"dataset": "CARE rectal CT (official held-out test split)", "encoder": args.encoder,
           "n": int(ei.size), "mean_iou": round(im_,4), "iou_95ci": [round(ilo,4), round(ihi,4)],
           "mean_dice": round(dm_,4), "sens_at_iou0.5": round(float((ei>=0.5).mean()),4),
           "tumor_labels": list(TUMOR_LABELS), "elapsed_min": round((time.time()-t0)/60,1)}
    Path(str(args.out).replace(".pt","_metrics.json")).write_text(json.dumps(rep, indent=2))
    print(f"\n=== CARE held-out test: IoU={rep['mean_iou']} 95%CI{rep['iou_95ci']} "
          f"Dice={rep['mean_dice']} sens@0.5={rep['sens_at_iou0.5']} (n={rep['n']}) ===\nSaved → {args.out}")


if __name__ == "__main__":
    main()
