"""Dedicated cross-vendor polyp SEGMENTATION fine-tune (trainable encoder) — Kaggle free GPU.

This is the honest "backbone fine-tune": a standalone U-Net with a pretrained,
**trainable** encoder — decoupled from the main 5-class model, so improving
segmentation cannot regress the classifier. Goal: beat the frozen-decoder
cross-vendor IoU (~0.45 on held-out ETIS/Pentax).

Trains on the seg dataset from scripts/seg/prepare_seg_dataset.py (ETIS held out),
reports the honest ETIS IoU/Dice with bootstrap 95% CIs, and exports the weights.

Kaggle:  !pip -q install segmentation-models-pytorch albumentations
         !python train_seg_kaggle.py --data /kaggle/input/<your-seg-dataset>/seg_polyp
Local :  python3 scripts/seg/train_seg_kaggle.py            (uses outputs/seg_polyp)
Resume:  --resume   (continues from the last checkpoint)
"""
from __future__ import annotations
import argparse, glob, os, sys, time, json
from pathlib import Path
import numpy as np


def _device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _find_data(arg):
    if arg and Path(arg).exists():
        return Path(arg)
    for c in [Path("outputs/seg_polyp"),
              *map(Path, glob.glob("/kaggle/input/**/seg_polyp", recursive=True))]:
        if (c / "images").exists():
            return c
    sys.exit("seg dataset not found — run scripts/seg/prepare_seg_dataset.py or pass --data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/seg_polyp")
    ap.add_argument("--encoder", default="resnet34",
                    help="smp encoder (resnet34 safe; timm-efficientnet-b3 = stronger)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=352)
    ap.add_argument("--out", default="outputs/seg_polyp/seg_finetune.pt")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    try:
        import segmentation_models_pytorch as smp
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2
    except ImportError as e:
        sys.exit(f"missing dep: {e}\n  pip install segmentation-models-pytorch albumentations opencv-python")

    data = _find_data(args.data); dev = _device()
    print(f"device={dev} · encoder={args.encoder} · data={data}")
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    class SegSet(Dataset):
        def __init__(self, split, aug):
            self.imgs = sorted(glob.glob(str(data / "images" / split / "*")))
            self.split = split
            if aug:
                self.tf = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.3),
                                     A.RandomRotate90(p=.5), A.ShiftScaleRotate(0.05,0.1,15,p=.5),
                                     A.ColorJitter(0.2,0.2,0.2,0.1,p=.5),
                                     A.Normalize(MEAN, STD), ToTensorV2()])
            else:
                self.tf = A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])
        def __len__(self): return len(self.imgs)
        def __getitem__(self, i):
            ip = self.imgs[i]
            mp = str(data / "masks" / self.split / (Path(ip).stem + ".png"))
            img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            msk = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype("float32")
            t = self.tf(image=img, mask=msk)
            return t["image"], t["mask"].unsqueeze(0)

    tr = DataLoader(SegSet("train", True), batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(SegSet("val", False), batch_size=args.batch, shuffle=False, num_workers=2)
    te = DataLoader(SegSet("test", False), batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"train {len(tr.dataset)} · val {len(va.dataset)} · test(ETIS) {len(te.dataset)}")

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=1).to(dev)
    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and out_p.exists():
        model.load_state_dict(torch.load(out_p, map_location=dev)["state_dict"]); print("resumed")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dice = smp.losses.DiceLoss(mode="binary"); bce = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); ious, dices = [], []
        for x, y in dl:
            p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
            inter = (p * y).sum((1,2,3)); union = p.sum((1,2,3)) + y.sum((1,2,3)) - inter
            den = p.sum((1,2,3)) + y.sum((1,2,3))
            for j in range(x.size(0)):
                u = float(union[j]); d = float(den[j])
                ious.append(float(inter[j]/u) if u>1e-6 else 0.0)
                dices.append(float(2*inter[j]/d) if d>1e-6 else 0.0)
        return np.array(ious), np.array(dices)

    def ci(a, B=2000):
        if a.size == 0: return (0,0,0)
        r = np.random.default_rng(0); m = a[r.integers(0,a.size,(B,a.size))].mean(1)
        return float(a.mean()), float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

    print("Training …"); best = 0.0; t0 = time.time()
    for ep in range(start, args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = dice(model(x), y) + bce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vi, _ = evaluate(va); viou = float(vi.mean())
        print(f"  ep{ep+1}/{args.epochs} val_IoU={viou:.4f}")
        if viou > best:
            best = viou
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "val_iou": best, "imgsz": args.imgsz}, out_p)

    model.load_state_dict(torch.load(out_p, map_location=dev)["state_dict"])
    ei, ed = evaluate(te); im, ilo, ihi = ci(ei); dm, dlo, dhi = ci(ed)
    sens = float((ei >= 0.5).mean())
    rep = {"encoder": args.encoder, "test_set": "ETIS-Larib (Pentax) held-out",
           "n": int(ei.size), "mean_iou": round(im,4), "iou_95ci": [round(ilo,4), round(ihi,4)],
           "mean_dice": round(dm,4), "dice_95ci": [round(dlo,4), round(dhi,4)],
           "sens_at_iou0.5": round(sens,4), "prev_frozen_decoder_iou": 0.45,
           "elapsed_min": round((time.time()-t0)/60,1)}
    Path(str(out_p).replace(".pt","_metrics.json")).write_text(json.dumps(rep, indent=2))
    print(f"\n=== HONEST ETIS (Pentax) held-out ===")
    print(f"  IoU={rep['mean_iou']} 95%CI{rep['iou_95ci']}  Dice={rep['mean_dice']}  "
          f"sens@0.5={rep['sens_at_iou0.5']}  (was ~0.45 frozen)")
    try:
        model.eval()
        torch.onnx.export(model, torch.randn(1,3,args.imgsz,args.imgsz).to(dev),
                          str(out_p).replace(".pt",".onnx"), opset_version=17,
                          input_names=["x"], output_names=["mask"])
        print("  exported ONNX")
    except Exception as e:
        print(f"  onnx export skipped: {e}")
    print(f"Saved → {out_p}")


if __name__ == "__main__":
    main()
