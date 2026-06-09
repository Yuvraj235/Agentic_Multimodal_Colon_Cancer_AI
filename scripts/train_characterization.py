"""Polyp characterization (CADx) — neoplastic vs non-neoplastic (smart win #2).

Trains on BKAI-IGH NeoPolyp (1,000 colonoscopy images with RGB masks: red =
neoplastic, green = non-neoplastic). For each image we derive an image-level label
from the mask, crop to the polyp bounding box (focus on the lesion), and train a
ResNet18 to answer "what kind of polyp": neoplastic (resect/biopsy) vs
non-neoplastic (likely benign). Ambiguous 'both' images are excluded.

Class-weighted loss (data is ~74% neoplastic). Honest held-out metrics; the key
number is neoplastic recall (don't miss the resect-worthy ones) + non-neoplastic
precision (don't wrongly call benign ones neoplastic).

Outputs: outputs/unified_multimodal_v2/cadx_head.pth + cadx_metrics.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

IMG = Path("data/raw/bkai/images")
MSK = Path("data/raw/bkai/masks")
OUT = Path("outputs/unified_multimodal_v2/cadx_head.pth")
MET = Path("outputs/unified_multimodal_v2/cadx_metrics.json")
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLASSES = ["non-neoplastic", "neoplastic"]   # 0, 1


def _label_and_bbox(mask_path):
    m = np.asarray(Image.open(mask_path).convert("RGB"))
    r = (m[:, :, 0] > 100) & (m[:, :, 1] < 80)
    g = (m[:, :, 1] > 100) & (m[:, :, 0] < 80)
    rc, gc = int(r.sum()), int(g.sum())
    if rc < 20 and gc < 20:
        return None, None                      # no polyp
    if rc > 0 and gc > 0 and min(rc, gc) > 0.2 * max(rc, gc):
        return None, None                      # ambiguous 'both' -> exclude
    label = 1 if rc >= gc else 0
    poly = r | g
    ys, xs = np.where(poly)
    H, W = poly.shape
    my, mx = int(0.08 * H), int(0.08 * W)
    y0, y1 = max(0, ys.min() - my), min(H, ys.max() + my)
    x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + mx)
    return label, (x0, y0, x1, y1)


class CadxSet(Dataset):
    def __init__(self, samples, augment):
        self.samples = samples
        if augment:
            self.tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(0.5),
                                 T.RandomVerticalFlip(0.5), T.RandomRotation(15),
                                 T.ColorJitter(0.15, 0.15, 0.10), T.ToTensor(), _NORM])
        else:
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), _NORM])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label, bbox = self.samples[i]
        img = Image.open(path).convert("RGB")
        if bbox:
            img = img.crop(bbox)
        return self.tf(img), label


def build_samples():
    samples = []
    for f in sorted(IMG.glob("*.jpeg")):
        mp = MSK / f.name
        if not mp.exists():
            continue
        label, bbox = _label_and_bbox(mp)
        if label is None:
            continue
        samples.append((str(f), label, bbox))
    return samples


def main():
    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {dev}")
    samples = build_samples()
    labels = np.array([s[1] for s in samples])
    print(f"usable {len(samples)} | non-neoplastic {int((labels==0).sum())} | neoplastic {int((labels==1).sum())}")

    rng = np.random.default_rng(42)
    tr_idx, va_idx = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        k = int(0.8 * len(idx)); tr_idx += idx[:k].tolist(); va_idx += idx[k:].tolist()
    tr = CadxSet([samples[i] for i in tr_idx], augment=True)
    va = CadxSet([samples[i] for i in va_idx], augment=False)
    tr_dl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(va, batch_size=32, shuffle=False, num_workers=2)
    print(f"train {len(tr)} · val {len(va)}")

    # class weights (inverse freq) for the imbalance
    cnt = np.bincount(labels, minlength=2).astype(float)
    w = torch.tensor((cnt.sum() / (2 * cnt)), dtype=torch.float32).to(dev)
    print(f"class weights: {w.tolist()}")

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6*len(tr_dl))

    def evaluate():
        net.eval(); conf = np.zeros((2, 2), int)
        with torch.no_grad():
            for x, y in va_dl:
                p = net(x.to(dev)).argmax(1).cpu().numpy()
                for t, pr in zip(y.numpy(), p):
                    conf[t, pr] += 1
        return conf

    print("Training …"); t0 = time.time(); best = 0.0
    for ep in range(6):
        net.train()
        for step, (x, y) in enumerate(tr_dl):
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(net(x), y, weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        conf = evaluate()
        # balanced acc = mean per-class recall (robust to imbalance)
        recs = [conf[i, i] / max(1, conf[i].sum()) for i in (0, 1)]
        bacc = float(np.mean(recs))
        print(f"  ep{ep+1}/6 balanced_acc={bacc:.4f} (non-neo R={recs[0]:.3f}, neo R={recs[1]:.3f})")
        if bacc > best:
            best = bacc
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "classes": CLASSES,
                        "arch": "resnet18", "balanced_acc": bacc}, OUT)

    net.load_state_dict(torch.load(OUT)["state_dict"]); conf = evaluate()
    per = {}
    for i, c in enumerate(CLASSES):
        tp = conf[i, i]; sup = conf[i].sum(); pred = conf[:, i].sum()
        rec = tp/sup if sup else 0; prec = tp/pred if pred else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        per[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                  "f1": round(float(f1), 4), "support": int(sup)}
    bacc = float(np.mean([per[c]["recall"] for c in CLASSES]))
    print(f"\n  >>> balanced acc={bacc:.4f}")
    print(f"      neoplastic: P={per['neoplastic']['precision']} R={per['neoplastic']['recall']}")
    print(f"      non-neoplastic: P={per['non-neoplastic']['precision']} R={per['non-neoplastic']['recall']}")
    MET.write_text(json.dumps({
        "dataset": "BKAI-IGH NeoPolyp (1000 imgs; 'both' excluded)",
        "task": "neoplastic vs non-neoplastic (CADx); polyp-cropped from mask bbox",
        "classes": CLASSES, "balanced_acc": round(bacc, 4), "per_class": per,
        "confusion_matrix": conf.tolist(),
        "note": "image-level label derived from mask colour; held-out 80/20 stratified split",
        "elapsed_min": round((time.time()-t0)/60, 1),
    }, indent=2))
    print(f"Saved → {OUT}\n      metrics → {MET}")


if __name__ == "__main__":
    main()
