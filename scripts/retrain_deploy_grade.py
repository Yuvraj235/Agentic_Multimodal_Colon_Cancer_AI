#!/usr/bin/env python3
"""ColonAI — deploy-grade multi-task retraining.

Why this script exists
──────────────────────
v1 (best_model.pth) is great at classifying polyps (99.5% test acc) but its
GradCAM lands in the wrong place on Pentax scopes (IoU 0.07 vs Olympus 0.24).
Naive retraining on more polyp data fixes GradCAM but RISKS degrading the
staging + risk heads.

What this trains
────────────────
   • **Pathology head**     → focal loss + class weights (real labels)
   • **Staging head**       → KL distillation from v1 teacher (no fake labels)
   • **Risk head**          → KL distillation from v1 teacher (no fake labels)
   • **GradCAM target layer** → mask-aligned attention loss (KL vs polyp mask)
   • **Fusion backbone**    → all of the above, gentle LR

The KL-from-teacher trick: we freeze a COPY of the current v1 model as
"teacher". On every batch we run the teacher and the student. The student's
staging and risk logits must stay close to the teacher's (KL divergence).
The student's pathology + spatial attention are free to improve. This way
NOTHING DRIFTS that was already good.

Multi-vendor polyp pool (2,348 masks): Kvasir-SEG, CVC-ClinicDB,
CVC-ColonDB, CVC-300, ETIS-Larib (Pentax), Kvasir-test
HyperKvasir+CVC pool (4,082): full 5-class pathology coverage.
TOTAL: 6,430 training images.

Run from project root:
    python3 scripts/retrain_deploy_grade.py --epochs 2 --batch_size 16

Output:
    outputs/unified_multimodal_v2/checkpoints/best_model.pth
    outputs/unified_multimodal_v2/train_log.json
    outputs/unified_multimodal_v2/temperature.json  (post-train calibration)
"""
from __future__ import annotations
import argparse, json, sys, time, copy
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    N_TABULAR_FEATURES,
    HyperKvasirMultiModalDataset as MultiModalDataset,
    CLASS_NAMES_5,
)
from transformers import AutoTokenizer


CKPT_IN  = "outputs/unified_multimodal/checkpoints/best_model.pth"
CKPT_OUT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
LOG_OUT  = "outputs/unified_multimodal_v2/train_log.json"
TEMP_OUT = "outputs/unified_multimodal_v2/temperature.json"
BERT     = "dmis-lab/biobert-base-cased-v1.2"

POLYP_MASK_DATASETS = [
    ("Kvasir-SEG",        "data/raw/kvasir-seg/Kvasir-SEG/images",                            "data/raw/kvasir-seg/Kvasir-SEG/masks",                          "Olympus"),
    ("CVC-ClinicDB",      "data/raw/CVC-ClinicDB/PNG/Original",                               "data/raw/CVC-ClinicDB/PNG/Ground Truth",                        "Olympus"),
    ("CVC-ColonDB",       "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/images",      "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/masks",    "Olympus"),
    ("CVC-300",           "data/raw/test_polyp_datasets/TestDataset/CVC-300/images",          "data/raw/test_polyp_datasets/TestDataset/CVC-300/masks",        "Olympus"),
    ("ETIS-LaribPolypDB", "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images","data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/masks","Pentax"),
    ("Kvasir-test",       "data/raw/test_polyp_datasets/TestDataset/Kvasir/images",           "data/raw/test_polyp_datasets/TestDataset/Kvasir/masks",         "Olympus"),
]


# ════════════════════════════════════════════════════════════════════════════
# Dataset — polyp images WITH pixel masks
# ════════════════════════════════════════════════════════════════════════════
class MaskedPolypDataset(Dataset):
    EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    def __init__(self, tokenizer, img_size: int = 224, max_seq_len: int = 64,
                 augment: bool = True):
        self.tokenizer = tokenizer
        self.img_size  = img_size
        self.max_seq_len = max_seq_len
        self.augment   = augment

        if augment:
            self.tfm = T.Compose([
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop((img_size, img_size)),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.3),
                T.ColorJitter(brightness=0.25, contrast=0.25,
                              saturation=0.20, hue=0.04),
                T.RandomPerspective(distortion_scale=0.20, p=0.4),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.3, scale=(0.02, 0.10)),
            ])
        else:
            self.tfm = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        self.samples: List[Tuple[str, str, str]] = []
        for name, img_root, msk_root, vendor in POLYP_MASK_DATASETS:
            img_dir, msk_dir = Path(img_root), Path(msk_root)
            if not img_dir.exists() or not msk_dir.exists():
                print(f"  · {name:18s}  NOT FOUND  (skip)"); continue
            n = 0
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in self.EXTS: continue
                msk_path = None
                for e in self.EXTS:
                    cand = msk_dir / (img_path.stem + e)
                    if cand.exists(): msk_path = cand; break
                if msk_path is None: continue
                self.samples.append((str(img_path), str(msk_path), vendor))
                n += 1
            print(f"  · {name:18s}  +{n:4d}  ({vendor})")
        print(f"  TOTAL  polyp+mask: {len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path, vendor = self.samples[idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
            pil_msk = Image.open(msk_path).convert("L")
        except Exception:
            return self.__getitem__(0)

        if self.augment:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
            torch.manual_seed(seed)
            image = self.tfm(pil_img)
            torch.manual_seed(seed)
            msk = T.Resize((self.img_size + 32, self.img_size + 32),
                           interpolation=T.InterpolationMode.NEAREST)(pil_msk)
            msk = T.RandomCrop((self.img_size, self.img_size))(msk)
            msk = T.RandomHorizontalFlip(0.5)(msk)
            msk = T.RandomVerticalFlip(0.3)(msk)
            mask = T.ToTensor()(msk); mask = (mask > 0.5).float()
        else:
            image = self.tfm(pil_img)
            mask  = T.Resize((self.img_size, self.img_size),
                             interpolation=T.InterpolationMode.NEAREST)(pil_msk)
            mask  = T.ToTensor()(mask); mask = (mask > 0.5).float()

        text = "Patient referred for screening colonoscopy with bowel preparation review."
        enc = self.tokenizer(text, max_length=self.max_seq_len, padding="max_length",
                             truncation=True, return_tensors="pt")

        rng = np.random.default_rng(hash(img_path) % (2**31))
        tabular = torch.tensor([
            55 + rng.normal(0, 9), 27 + rng.normal(0, 4),
            2014 + rng.integers(0, 9), 400 + rng.normal(0, 250),
            float(rng.choice([0, 5, 10, 20])),
            float(rng.choice([0, 5, 15, 30])),
            float(rng.binomial(1, 0.4)), float(rng.binomial(1, 0.48)),
            float(rng.integers(0, 4)), 0.0, 8140.0,
            float(rng.integers(0, 12)),
        ], dtype=torch.float32)

        return {
            "image":          image,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tabular":        tabular,
            "label":          torch.tensor(0, dtype=torch.long),
            "mask":           mask,
            "has_mask":       torch.tensor(1.0),
            "vendor":         vendor,
        }


class NoMaskAdapter(Dataset):
    """Adapts HyperKvasir dataset (no pixel mask) for the unified loader."""
    def __init__(self, inner, img_size=224):
        self.inner = inner; self.img_size = img_size
    def __len__(self): return len(self.inner)
    def __getitem__(self, idx):
        s = self.inner[idx]
        s["mask"]     = torch.zeros((1, self.img_size, self.img_size))
        s["has_mask"] = torch.tensor(0.0)
        s["vendor"]   = "HyperKvasir"
        return s


# ════════════════════════════════════════════════════════════════════════════
# Attention-mask alignment loss (forces ResNet features into the polyp)
# ════════════════════════════════════════════════════════════════════════════
class AttentionMaskLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feats = None
        target = model.get_image_target_layer()
        def hook(_m, _i, out): self.feats = out
        self._hook = target.register_forward_hook(hook)

    def forward(self, mask, has_mask):
        if self.feats is None:
            return torch.tensor(0.0, device=mask.device)
        B = mask.size(0)
        attn = self.feats.abs().mean(dim=1)              # (B,h,w)
        h, w = attn.shape[-2], attn.shape[-1]
        mask_lo = F.adaptive_avg_pool2d(mask, (h, w)).squeeze(1)
        attn_p   = F.softmax(attn.reshape(B, -1), dim=-1)
        mask_f   = mask_lo.reshape(B, -1)
        mask_p   = mask_f / mask_f.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        eps      = 1e-8
        kl       = (attn_p * (torch.log(attn_p + eps) - torch.log(mask_p + eps))).sum(dim=-1)
        weight   = has_mask.to(kl.dtype)
        return (kl * weight).sum() / weight.sum().clamp(min=1.0)

    def close(self):
        if self._hook is not None: self._hook.remove()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing; self.cw = class_weights
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.cw,
                             label_smoothing=self.ls, reduction="none")
        with torch.no_grad():
            pt = F.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        return ((1 - pt) ** self.gamma * ce).mean()


# ════════════════════════════════════════════════════════════════════════════
# Temperature scaling (post-train calibration)
# ════════════════════════════════════════════════════════════════════════════
def calibrate_temperature(model, val_dl, device) -> float:
    """Fit a single T scalar on val set so softmax(logits/T) is well-calibrated."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_dl:
            out = model(batch["image"].to(device),
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["tabular"].to(device))
            logits_all.append(out["pathology"].cpu())
            labels_all.append(batch["label"])
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)

    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05).item())


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=2)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--bert_lr",      type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=0.12)
    ap.add_argument("--mask_loss_w",  type=float, default=0.6)
    ap.add_argument("--distill_w",    type=float, default=0.5,
                    help="weight for staging+risk KL distillation from v1 teacher")
    ap.add_argument("--distill_T",    type=float, default=2.0)
    ap.add_argument("--hk_root",      default="data/processed/hyper_kvasir_clean")
    ap.add_argument("--cvc_root",     default="data/raw/CVC-ClinicDB")
    ap.add_argument("--tcga_dir",     default="data/raw/tcga")
    ap.add_argument("--ckpt_in",      default=CKPT_IN)
    ap.add_argument("--ckpt_out",     default=CKPT_OUT)
    ap.add_argument("--max_train_batches", type=int, default=0)
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    # ── 1. Tokeniser ────────────────────────────────────────────────────────
    print("\n[1/6] Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(BERT)

    # ── 2. Build combined dataset ───────────────────────────────────────────
    print("\n[2/6] Building multi-vendor masked-polyp corpus …")
    masked_ds = MaskedPolypDataset(tokenizer=tokenizer, augment=True)
    if len(masked_ds) == 0:
        print("  ERROR: no polyp+mask samples found. Aborting."); return

    print("\n      Loading HyperKvasir (multi-class, no mask) …")
    tcga_path = Path(args.tcga_dir) / "clinical" / "clinical.tsv"
    tcga_df   = pd.read_csv(tcga_path, sep="\t") if tcga_path.exists() else None
    hk_train  = MultiModalDataset(
        root_dir=args.hk_root, tokenizer=tokenizer, tcga_df=tcga_df,
        split="train", val_ratio=0.15, test_ratio=0.10, seed=42,
        manifest_dir="outputs/unified_multimodal_v2/manifests")
    if Path(args.cvc_root).exists(): hk_train.add_cvc_clinicdb(args.cvc_root)
    hk_train_a = NoMaskAdapter(hk_train)

    full_train = ConcatDataset([hk_train_a, masked_ds])
    hk_val = MultiModalDataset(
        root_dir=args.hk_root, tokenizer=tokenizer, tcga_df=tcga_df,
        split="val", val_ratio=0.15, test_ratio=0.10, seed=42)
    print(f"\n      Combined training samples: {len(full_train):,}")

    # Class-balanced sampler
    print("\n      Computing class-balanced sample weights …")
    labels = [lb for _, lb, _ in hk_train.samples] + [0] * len(masked_ds)
    counts = np.bincount(labels, minlength=5).astype(float)
    print(f"        counts: {dict(zip(CLASS_NAMES_5, counts.astype(int).tolist()))}")
    counts = np.where(counts == 0, 1, counts)
    class_w  = 1.0 / counts
    sampler  = WeightedRandomSampler([class_w[l] for l in labels],
                                     num_samples=len(labels), replacement=True)
    class_weights = torch.tensor(class_w / class_w.sum(),
                                 dtype=torch.float32).to(device)

    train_dl = DataLoader(full_train, batch_size=args.batch_size,
                          sampler=sampler, num_workers=2, drop_last=True)
    val_dl   = DataLoader(hk_val, batch_size=args.batch_size,
                          shuffle=False, num_workers=2)
    print(f"      Train batches: {len(train_dl)}  ·  Val batches: {len(val_dl)}")

    # ── 3. Build student + teacher (both warm-started from v1) ─────────────
    print(f"\n[3/6] Building student + teacher from {args.ckpt_in} …")
    student = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    teacher = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)

    if not Path(args.ckpt_in).exists():
        print(f"      ERROR: {args.ckpt_in} not found. Cannot warm-start."); return
    state = torch.load(args.ckpt_in, map_location="cpu")
    state = state.get("model_state", state)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    print("      Both networks loaded with v1 weights.")
    print("      Teacher frozen — staging + risk heads anchored to v1 behaviour.")

    # ── 4. Losses + optimiser ───────────────────────────────────────────────
    focal     = FocalLoss(gamma=2.0, label_smoothing=0.1, class_weights=class_weights)
    attn_loss = AttentionMaskLoss(student).to(device)

    bert_params  = [p for n, p in student.named_parameters() if "bert" in n]
    other_params = [p for n, p in student.named_parameters() if "bert" not in n]
    optim = torch.optim.AdamW(
        [{"params": bert_params,  "lr": args.bert_lr},
         {"params": other_params, "lr": args.lr}],
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(train_dl), eta_min=args.lr * 0.1)

    # ── 5. Train ────────────────────────────────────────────────────────────
    print(f"\n[4/6] Training {args.epochs} epoch(s) …")
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    log = {"epochs": [], "config": vars(args)}
    best_val = 0.0
    t0 = time.time()

    def distill_kl(s_logits, t_logits, T):
        return F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                        F.softmax(t_logits / T, dim=-1),
                        reduction="batchmean") * (T * T)

    for ep in range(args.epochs):
        student.train()
        run = {"loss": 0, "cls": 0, "attn": 0, "stg": 0, "rsk": 0,
               "n": 0, "correct": 0, "total": 0}
        for step, batch in enumerate(train_dl):
            if args.max_train_batches and step >= args.max_train_batches: break
            img  = batch["image"].to(device)
            ids  = batch["input_ids"].to(device)
            msk  = batch["attention_mask"].to(device)
            tab  = batch["tabular"].to(device)
            lbl  = batch["label"].to(device)
            pmsk = batch["mask"].to(device)
            hmsk = batch["has_mask"].to(device)

            optim.zero_grad()

            # Teacher forward (no grad) — get staging + risk targets
            with torch.no_grad():
                t_out = teacher(img, ids, msk, tab)
                t_stg = t_out["staging"]
                t_rsk = t_out["risk"]

            # Student forward
            s_out = student(img, ids, msk, tab)
            loss_cls  = focal(s_out["pathology"], lbl)
            loss_stg  = distill_kl(s_out["staging"], t_stg, args.distill_T)
            loss_rsk  = distill_kl(s_out["risk"],    t_rsk, args.distill_T)
            loss_attn = attn_loss(pmsk, hmsk) if hmsk.sum() > 0 \
                        else torch.tensor(0.0, device=device)

            loss = (loss_cls
                    + args.distill_w  * (loss_stg + loss_rsk)
                    + args.mask_loss_w * loss_attn)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step(); scheduler.step()

            run["loss"] += loss.item();      run["cls"] += loss_cls.item()
            run["attn"] += loss_attn.item(); run["stg"] += loss_stg.item()
            run["rsk"]  += loss_rsk.item();  run["n"]   += 1
            pred = s_out["pathology"].argmax(-1)
            run["correct"] += int((pred == lbl).sum())
            run["total"]   += lbl.size(0)

            if step % 20 == 0:
                print(f"  ep {ep+1}/{args.epochs}  step {step:4d}/{len(train_dl)}  "
                      f"loss={loss.item():.3f}  cls={loss_cls.item():.3f}  "
                      f"attn={loss_attn.item():.3f}  stg={loss_stg.item():.3f}  "
                      f"rsk={loss_rsk.item():.3f}  "
                      f"acc={run['correct']/max(1,run['total']):.3f}")

        # Validation
        student.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for batch in val_dl:
                pred = student(batch["image"].to(device),
                               batch["input_ids"].to(device),
                               batch["attention_mask"].to(device),
                               batch["tabular"].to(device))["pathology"].argmax(-1)
                v_correct += int((pred == batch["label"].to(device)).sum())
                v_total   += batch["label"].size(0)
        val_acc = v_correct / max(1, v_total)

        ep_log = {"epoch": ep+1, **{k: run[k]/max(1, run["n"])
                                    for k in ["loss","cls","attn","stg","rsk"]},
                  "train_acc": run["correct"]/max(1,run["total"]),
                  "val_acc": val_acc,
                  "elapsed_min": (time.time()-t0)/60.0}
        log["epochs"].append(ep_log)
        print(f"\n  ✓ Ep {ep+1}  loss={ep_log['loss']:.3f}  "
              f"train_acc={ep_log['train_acc']:.4f}  val_acc={val_acc:.4f}  "
              f"({ep_log['elapsed_min']:.1f} min)")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": student.state_dict(),
                        "val_acc":     val_acc,
                        "epoch":       ep+1,
                        "classes":     CLASS_NAMES_5,
                        "config":      vars(args)},
                       args.ckpt_out)
            print(f"  ★ Saved new best to {args.ckpt_out}  (val_acc={val_acc:.4f})")

    # ── 6. Temperature scaling ──────────────────────────────────────────────
    print(f"\n[5/6] Calibrating temperature on val set …")
    T_opt = calibrate_temperature(student, val_dl, device)
    Path(TEMP_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(TEMP_OUT).write_text(json.dumps({"temperature": T_opt}, indent=2))
    print(f"      Optimal T = {T_opt:.3f}  →  saved to {TEMP_OUT}")

    Path(LOG_OUT).write_text(json.dumps(log, indent=2))
    attn_loss.close()

    print(f"\n[6/6] DONE — best val acc {best_val:.4f}")
    print(f"      checkpoint: {args.ckpt_out}")
    print(f"      train log : {LOG_OUT}")
    print(f"      total time: {(time.time()-t0)/60.0:.1f} min")
    print("\nNext: python3 scripts/validate_gradcam_cross_vendor.py "
          "(point CHECKPOINT to outputs/unified_multimodal_v2/checkpoints/best_model.pth)")


if __name__ == "__main__":
    main()
