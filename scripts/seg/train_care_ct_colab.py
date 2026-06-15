"""CARE rectal-cancer CT tumor segmentation — Colab + Google Drive (free).

The CARE dataset (398 patients, 33,024 slice pairs, pixel masks of normal +
cancerous rectum; CC BY-NC 4.0) is ~74 GB, so it lives on your 5 TB Drive and we
train on Colab (Colab can mount Drive; Kaggle can't). This trains a U-Net with a
TRAINABLE encoder to segment the rectal TUMOR on CT, with an honest held-out test.

IMPORTANT — run the INSPECT step first. I can't see your Drive, so the script
auto-discovers the layout and prints it (folders, file types, mask label values).
If auto-detect is wrong, set IMG_GLOB / MASK_DIR_NAME / TUMOR_LABELS at the top from
what the printout shows, and re-run.

Colab usage (paste as cells, or run the file):
  from google.colab import drive; drive.mount('/content/drive')
  !pip -q install segmentation-models-pytorch albumentations
  !python train_care_ct_colab.py --root "/content/drive/MyDrive/CARE"
  # first run prints the structure; then it trains.
"""
from __future__ import annotations
import argparse, glob, os, sys, json, time
from pathlib import Path
import numpy as np

# ── Config you may need to tweak AFTER the first INSPECT printout ────────────
MASK_HINTS = ("mask", "label", "gt", "ground", "anno", "seg")   # folder/name hints for masks
IMG_HINTS  = ("image", "img", "ct", "data", "scan", "slice")
IMG_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
# CARE masks label normal rectum + tumor. We segment the TUMOR. If the printout
# shows mask values like [0,1,2], the tumour is usually the HIGHEST value — set it
# here once you know (None = "any non-zero label = tumour", a safe default).
TUMOR_LABELS = None     # e.g. {2} once you see the values; None = non-zero


def inspect(root: Path):
    print(f"\n=== INSPECT {root} ===")
    if not root.exists():
        sys.exit(f"root not found: {root} — set --root to your CARE folder on Drive")
    # show top 3 levels of the tree (dirs + file-type counts)
    seen = 0
    for dp, dns, fns in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        if depth > 2:
            dns[:] = []
            continue
        exts = {}
        for f in fns:
            exts[Path(f).suffix.lower()] = exts.get(Path(f).suffix.lower(), 0) + 1
        if fns or dns:
            print(f"  {'  '*depth}{Path(dp).name}/  dirs={len(dns)} files={dict(sorted(exts.items()))}")
        seen += 1
        if seen > 60:
            print("  … (truncated)"); break


def _discover(root: Path):
    """Find (image_files, mask_for(image)) by pairing image & mask dirs."""
    # candidate mask dirs/files by hint, image dirs by hint
    all_imgs = [Path(p) for p in glob.glob(str(root / "**" / "*"), recursive=True)
                if Path(p).suffix.lower() in IMG_EXTS and Path(p).is_file()]
    masks = [p for p in all_imgs if any(h in str(p).lower() for h in MASK_HINTS)]
    imgs  = [p for p in all_imgs if p not in set(masks)
             and any(h in str(p).lower() for h in IMG_HINTS)]
    if not imgs:   # fallback: images = everything that isn't a mask
        imgs = [p for p in all_imgs if p not in set(masks)]
    # pair by matching stem
    mask_by_stem = {}
    for m in masks:
        mask_by_stem.setdefault(m.stem.replace("_mask", "").replace("_label", ""), m)
    pairs = []
    for im in imgs:
        m = mask_by_stem.get(im.stem) or mask_by_stem.get(im.stem.replace("_image", ""))
        if m:
            pairs.append((im, m))
    return pairs


def _load_mask(p: Path):
    if p.suffix.lower() == ".npy":
        m = np.load(p)
    else:
        import cv2
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    m = np.asarray(m)
    if m.ndim == 3:
        m = m[..., 0]
    if TUMOR_LABELS is not None:
        return np.isin(m, list(TUMOR_LABELS)).astype("float32")
    return (m > 0).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CARE folder on Drive")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--out", default="/content/drive/MyDrive/CARE/care_ct_seg.pt")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    inspect(root)
    pairs = _discover(root)
    print(f"\nDiscovered {len(pairs)} image/mask pairs.")
    if pairs[:2]:
        print("  sample:", [(str(a.name), str(b.name)) for a, b in pairs[:2]])
    if args.inspect_only or not pairs:
        if not pairs:
            print("\n⚠ No pairs auto-detected. Look at the tree above and set IMG_HINTS /"
                  " MASK_HINTS / the dir names at the top of this file, then re-run.")
        return

    import torch, torch.nn as nn, cv2
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} · encoder={args.encoder}")

    # sanity: print mask label values from a few masks (so you can set TUMOR_LABELS)
    vals = set()
    for _, m in pairs[:30]:
        try: vals |= set(np.unique(_load_mask_raw(m)).tolist())
        except Exception: pass
    print(f"  mask raw label values seen (first 30): {sorted(vals)[:12]}  "
          f"(TUMOR_LABELS={TUMOR_LABELS})")

    # patient-level split if a patient id is in the path, else 85/15 by file
    def pid(p):
        import re; m = re.search(r"(patient|case|pt|p)[_-]?(\d+)", str(p).lower());
        return m.group(0) if m else str(p.parent)
    ids = sorted({pid(a) for a, _ in pairs})
    rng = np.random.default_rng(42); rng.shuffle(ids)
    k = int(0.85 * len(ids)); tr_ids = set(ids[:k])
    tr = [(a, b) for a, b in pairs if pid(a) in tr_ids]
    va = [(a, b) for a, b in pairs if pid(a) not in tr_ids]
    print(f"  split by patient: train {len(tr)} · holdout {len(va)}  ({len(ids)} patients)")

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    aug = A.Compose([A.Resize(args.imgsz, args.imgsz), A.HorizontalFlip(p=.5),
                     A.ShiftScaleRotate(0.05, 0.1, 12, p=.5), A.RandomBrightnessContrast(p=.3),
                     A.Normalize(MEAN, STD), ToTensorV2()])
    noa = A.Compose([A.Resize(args.imgsz, args.imgsz), A.Normalize(MEAN, STD), ToTensorV2()])

    class DS(Dataset):
        def __init__(s, items, t): s.items, s.t = items, t
        def __len__(s): return len(s.items)
        def __getitem__(s, i):
            ip, mp = s.items[i]
            img = cv2.imread(str(ip)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((args.imgsz,args.imgsz,3),"uint8")
            msk = _load_mask(mp)
            o = s.t(image=img, mask=msk); return o["image"], o["mask"].unsqueeze(0).float()

    trdl = DataLoader(DS(tr, aug), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    vadl = DataLoader(DS(va, noa), batch_size=args.batch, shuffle=False, num_workers=2)
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    if args.resume and Path(args.out).exists():
        model.load_state_dict(torch.load(args.out, map_location=dev)["state_dict"]); print("resumed")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dice = smp.losses.DiceLoss(mode="binary"); bce = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def ev():
        model.eval(); ious = []
        for x, y in vadl:
            p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
            inter = (p*y).sum((1,2,3)); uni = p.sum((1,2,3)) + y.sum((1,2,3)) - inter
            for j in range(x.size(0)):
                u = float(uni[j]); ious.append(float(inter[j]/u) if u > 1e-6 else 0.0)
        return np.array(ious)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = 0.0; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, y in trdl:
            x, y = x.to(dev), y.to(dev)
            loss = dice(model(x), y) + bce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        iou = float(ev().mean()); print(f"  ep{ep+1}/{args.epochs} holdout_IoU={iou:.4f}")
        if iou > best:
            best = iou
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "iou": best, "imgsz": args.imgsz}, args.out)

    a = ev(); m_, lo, hi = float(a.mean()), float(np.percentile(a,2.5)), float(np.percentile(a,97.5))
    rep = {"dataset": "CARE rectal CT (held-out patients)", "encoder": args.encoder,
           "n": int(a.size), "mean_iou": round(m_,4), "iou_95ci": [round(lo,4), round(hi,4)],
           "sens_at_iou0.5": round(float((a>=0.5).mean()),4), "elapsed_min": round((time.time()-t0)/60,1)}
    Path(str(args.out).replace(".pt","_metrics.json")).write_text(json.dumps(rep, indent=2))
    print(f"\n=== CARE held-out: IoU={rep['mean_iou']} 95%CI{rep['iou_95ci']} "
          f"sens@0.5={rep['sens_at_iou0.5']} ===\nSaved → {args.out}")


def _load_mask_raw(p: Path):
    if p.suffix.lower() == ".npy": return np.load(p)
    import cv2; m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED);
    return m[...,0] if (m is not None and m.ndim==3) else m


if __name__ == "__main__":
    main()
