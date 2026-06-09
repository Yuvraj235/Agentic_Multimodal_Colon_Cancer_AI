"""Histology tissue-classifier specialist (NCT-CRC-HE 9 classes).

Demonstrator trained on CRC-VAL-HE-7K (7,180 H&E tiles). Transfer-learned ResNet18.
Nine classes: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM. The clinically key one
is TUM (colorectal adenocarcinoma epithelium) — "is there tumour tissue?" — which
feeds the staging/grading direction (a biopsy-slide branch, separate from colonoscopy).

Honest caveats: (1) trained on the 7K *validation* set (demonstrator) — the full 100K
is the production run, best on Colab; (2) stratified TILE-level split, not strict
patient-level, so held-out numbers are an upper bound. Reported transparently.

Outputs: outputs/unified_multimodal_v2/histology_head.pth + histology_metrics.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

ROOT = "data/raw/nct_crc_he/CRC-VAL-HE-7K"
OUT = Path("outputs/unified_multimodal_v2/histology_head.pth")
MET = Path("outputs/unified_multimodal_v2/histology_metrics.json")
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def main():
    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {dev}")

    base = ImageFolder(ROOT)               # to read targets + class order
    classes = base.classes
    print(f"Classes: {classes}")
    targets = np.array(base.targets)

    # Stratified 80/20 split (per class)
    rng = np.random.default_rng(42)
    tr_idx, va_idx = [], []
    for c in range(len(classes)):
        idx = np.where(targets == c)[0]; rng.shuffle(idx)
        k = int(0.8 * len(idx))
        tr_idx += idx[:k].tolist(); va_idx += idx[k:].tolist()
    print(f"train {len(tr_idx)} · val {len(va_idx)}")

    train_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(0.5),
                          T.RandomVerticalFlip(0.5), T.RandomRotation(15),
                          T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), _NORM])
    val_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), _NORM])
    tr_ds = Subset(ImageFolder(ROOT, transform=train_tf), tr_idx)
    va_ds = Subset(ImageFolder(ROOT, transform=val_tf), va_idx)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2)

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, len(classes))
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4*len(tr_dl))

    print("Training …"); t0 = time.time(); best = 0.0
    for ep in range(4):
        net.train()
        for step, (x, y) in enumerate(tr_dl):
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(net(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 40 == 0:
                print(f"  ep{ep+1}/4 step{step:3d}/{len(tr_dl)} loss={loss.item():.3f}")
        # val
        net.eval(); n = len(classes); conf = np.zeros((n, n), int)
        with torch.no_grad():
            for x, y in va_dl:
                p = net(x.to(dev)).argmax(1).cpu().numpy()
                for t, pr in zip(y.numpy(), p):
                    conf[t, pr] += 1
        acc = float(np.trace(conf) / conf.sum())
        print(f"  ✓ ep{ep+1} val_acc={acc:.4f}")
        if acc > best:
            best = acc
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "arch": "resnet18", "val_acc": acc}, OUT)

    # Final per-class metrics from best
    ck = torch.load(OUT); net.load_state_dict(ck["state_dict"]); net.eval()
    n = len(classes); conf = np.zeros((n, n), int)
    with torch.no_grad():
        for x, y in va_dl:
            p = net(x.to(dev)).argmax(1).cpu().numpy()
            for t, pr in zip(y.numpy(), p):
                conf[t, pr] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp = conf[i, i]; sup = conf[i].sum(); pred = conf[:, i].sum()
        rec = tp / sup if sup else 0.0; prec = tp / pred if pred else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        per_class[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                        "f1": round(float(f1), 4), "support": int(sup)}
    acc = float(np.trace(conf) / conf.sum())
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    print(f"\n  >>> overall acc={acc:.4f}  macro-F1={macro_f1:.4f}")
    print(f"      TUM (tumour) — precision {per_class['TUM']['precision']}, recall {per_class['TUM']['recall']}")
    MET.write_text(json.dumps({
        "dataset": "CRC-VAL-HE-7K (demonstrator; full 100K is the production run)",
        "split": "stratified tile-level 80/20 (not strict patient-level)",
        "classes": classes, "overall_acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class": per_class, "confusion_matrix": conf.tolist(),
        "elapsed_min": round((time.time()-t0)/60, 1),
    }, indent=2))
    print(f"Saved → {OUT}\n      metrics → {MET}")


if __name__ == "__main__":
    main()
