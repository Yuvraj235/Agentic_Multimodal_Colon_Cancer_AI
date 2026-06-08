#!/usr/bin/env python3
"""Fix uc-moderate-sev recall (baseline 0.154) by fine-tuning on LIMUC + a
UC-focused loss. Warm-starts from the current v2 checkpoint and writes to a NEW
checkpoint so the working model is untouched until the fix is verified.

Data:
  - HyperKvasir (5-class)            — class stability
  - polyp+mask sets                  — polyp stability
  - LIMUC (real Mayo-graded UC)      — Mayo 1 -> uc-mild, Mayo 2/3 -> uc-mod-sev
                                       (Mayo 0 = normal/remission excluded;
                                        split BY PATIENT, no leakage)
Loss:
  - focal (gamma=3) + inverse-freq class weights with uc-mod-sev boost
  - distillation KL anchors staging/risk heads to the teacher (no drift)
  - auxiliary mild<->mod-sev binary loss on the two UC logits, with the
    mod-sev class weighted 2x (asymmetric: under-calling severity is the
    dangerous error)

Eval: per-class precision/recall on the SAME HyperKvasir val split as the
baseline (seed=42), so uc-mod-sev recall is directly comparable to 0.154.

Usage:
  python3 scripts/retrain_uc_fix.py --max_train_batches 3   # smoke test
  python3 scripts/retrain_uc_fix.py --epochs 3              # full run
"""
import argparse, json, sys, time, random, copy
from pathlib import Path

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
from scripts.retrain_deploy_grade import (
    MaskedPolypDataset, NoMaskAdapter, AttentionMaskLoss, calibrate_temperature,
)

UC_MILD, UC_MODSEV = 1, 2
CKPT_IN  = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
CKPT_OUT = "outputs/unified_multimodal_v2/checkpoints/best_model_ucfix.pth"
METRICS_OUT = "outputs/unified_multimodal_v2/metrics_ucfix.json"
LIMUC_ROOT = "data/raw/limuc/patient_based_classified_images"
BERT = "dmis-lab/biobert-base-cased-v1.2"
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════
class LIMUCDataset(Dataset):
    """LIMUC UC-severity frames. Mayo 1 -> uc-mild, Mayo 2/3 -> uc-mod-sev.
    Mayo 0 (normal/remission) is NOT a model class and is excluded."""
    def __init__(self, tokenizer, patients, img_size=224, max_seq_len=64, augment=True):
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.samples = []
        root = Path(LIMUC_ROOT)
        for pid in patients:
            for mayo, label in (("Mayo 1", UC_MILD), ("Mayo 2", UC_MODSEV), ("Mayo 3", UC_MODSEV)):
                d = root / pid / mayo
                if not d.exists():
                    continue
                for f in d.glob("*.bmp"):
                    self.samples.append((str(f), label))
        if augment:
            self.tfm = T.Compose([
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop((img_size, img_size)),
                T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3),
                T.ColorJitter(0.25, 0.25, 0.20, 0.04),
                T.RandomPerspective(0.20, p=0.4),
                T.ToTensor(), T.Normalize(IMG_MEAN, IMG_STD),
                T.RandomErasing(p=0.3, scale=(0.02, 0.10)),
            ])
        else:
            self.tfm = T.Compose([T.Resize((img_size, img_size)),
                                  T.ToTensor(), T.Normalize(IMG_MEAN, IMG_STD)])

    def label_counts(self):
        c = {UC_MILD: 0, UC_MODSEV: 0}
        for _, lb in self.samples:
            c[lb] += 1
        return c

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        image = self.tfm(img)
        text = ("Patient with ulcerative colitis undergoing surveillance colonoscopy; "
                "mucosal inflammation severity assessment.")
        enc = self.tokenizer(text, max_length=self.max_seq_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        # Synthetic tabular filler (same pattern as MaskedPolypDataset — the UC
        # signal is in the IMAGE; tabular is just plausible non-constant filler).
        rng = np.random.default_rng(hash(path) % (2**31))
        tabular = torch.tensor([
            45 + rng.normal(0, 12), 25 + rng.normal(0, 4),
            2016 + rng.integers(0, 7), 300 + rng.normal(0, 200),
            float(rng.choice([0, 5, 10])), float(rng.choice([0, 5, 15])),
            float(rng.binomial(1, 0.3)), float(rng.binomial(1, 0.4)),
            float(rng.integers(0, 4)), 0.0, 8140.0, float(rng.integers(0, 12)),
        ], dtype=torch.float32)
        return {
            "image": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tabular": tabular,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.zeros((1, self.img_size, self.img_size)),
            "has_mask": torch.tensor(0.0),
            "vendor": "LIMUC",
        }


def limuc_patient_split(val_frac=0.15, seed=42):
    root = Path(LIMUC_ROOT)
    if not root.exists():
        raise SystemExit(f"LIMUC not found at {root} — run the download/extract first.")
    usable = []
    for p in sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
                    key=lambda d: int(d.name)):
        if any((p / m).exists() and any((p / m).glob("*.bmp"))
               for m in ("Mayo 1", "Mayo 2", "Mayo 3")):
            usable.append(p.name)
    rng = random.Random(seed)
    rng.shuffle(usable)
    n_val = int(len(usable) * val_frac)
    return usable[n_val:], usable[:n_val]


def uc_aux_loss(logits, target, device, modsev_w=1.3):
    """Train the mild<->mod-sev boundary directly on the two UC logits. mod-sev is
    weighted slightly higher (missing severity is the more dangerous error), but
    NOT so high that the model collapses everything into mod-sev — the goal is to
    DISTINGUISH the two grades, not flip the bias."""
    uc = (target == UC_MILD) | (target == UC_MODSEV)
    if uc.sum() == 0:
        return torch.tensor(0.0, device=device)
    uc_logits = logits[uc][:, [UC_MILD, UC_MODSEV]]
    uc_target = (target[uc] == UC_MODSEV).long()
    w = torch.tensor([1.0, modsev_w], device=device)
    return F.cross_entropy(uc_logits, uc_target, weight=w)


class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing; self.cw = class_weights
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.cw,
                             label_smoothing=self.ls, reduction="none")
        with torch.no_grad():
            pt = F.softmax(logits, -1).gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def evaluate(model, val_dl, device, T_scale=1.0):
    model.eval()
    n = len(CLASS_NAMES_5)
    conf = np.zeros((n, n), dtype=int)
    for batch in val_dl:
        out = model(batch["image"].to(device), batch["input_ids"].to(device),
                    batch["attention_mask"].to(device), batch["tabular"].to(device))
        pred = (out["pathology"] / T_scale).argmax(-1).cpu().numpy()
        lbl = batch["label"].numpy()
        for t, p in zip(lbl, pred):
            conf[t, p] += 1
    per_class = {}
    for i, name in enumerate(CLASS_NAMES_5):
        tp = conf[i, i]; support = conf[i].sum(); pred_i = conf[:, i].sum()
        recall = tp / support if support else 0.0
        prec = tp / pred_i if pred_i else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
        per_class[name] = {"precision": round(float(prec), 4),
                           "recall": round(float(recall), 4),
                           "f1": round(float(f1), 4), "support": int(support)}
    acc = float(np.trace(conf) / conf.sum()) if conf.sum() else 0.0
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return {"overall_acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per_class, "confusion_matrix": conf.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bert_lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=0.12)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--modsev_weight", type=float, default=1.0)
    ap.add_argument("--aux_w", type=float, default=0.5)
    ap.add_argument("--aux_modsev_w", type=float, default=1.3)
    ap.add_argument("--mask_loss_w", type=float, default=0.6)
    ap.add_argument("--distill_w", type=float, default=0.5)
    ap.add_argument("--distill_T", type=float, default=2.0)
    ap.add_argument("--hk_root", default="data/processed/hyper_kvasir_clean")
    ap.add_argument("--cvc_root", default="data/raw/CVC-ClinicDB")
    ap.add_argument("--tcga_dir", default="data/raw/tcga")
    ap.add_argument("--ckpt_in", default=CKPT_IN)
    ap.add_argument("--ckpt_out", default=CKPT_OUT)
    ap.add_argument("--max_train_batches", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[uc-fix] device={device}")
    tok = AutoTokenizer.from_pretrained(BERT)

    # ── Data ────────────────────────────────────────────────────────────────
    print("[1/6] Building datasets …")
    masked_ds = MaskedPolypDataset(tokenizer=tok, augment=True)
    tcga_path = Path(args.tcga_dir) / "clinical" / "clinical.tsv"
    tcga_df = pd.read_csv(tcga_path, sep="\t") if tcga_path.exists() else None
    hk_train = MultiModalDataset(root_dir=args.hk_root, tokenizer=tok, tcga_df=tcga_df,
                                 split="train", val_ratio=0.15, test_ratio=0.10, seed=42,
                                 manifest_dir="outputs/unified_multimodal_v2/manifests")
    if Path(args.cvc_root).exists():
        hk_train.add_cvc_clinicdb(args.cvc_root)
    hk_val = MultiModalDataset(root_dir=args.hk_root, tokenizer=tok, tcga_df=tcga_df,
                               split="val", val_ratio=0.15, test_ratio=0.10, seed=42)

    tr_pats, va_pats = limuc_patient_split(val_frac=0.15, seed=42)
    limuc_train = LIMUCDataset(tok, tr_pats, augment=True)
    print(f"      LIMUC train patients={len(tr_pats)} images={len(limuc_train)} "
          f"counts={limuc_train.label_counts()}")

    hk_train_a = NoMaskAdapter(hk_train)
    full_train = ConcatDataset([hk_train_a, masked_ds, limuc_train])

    # Sampler labels (inverse-freq, + uc-mod-sev boost)
    hk_labels = [lb for _, lb, *_ in hk_train.samples]
    labels = hk_labels + [0] * len(masked_ds) + [lb for _, lb in limuc_train.samples]
    counts = np.bincount(labels, minlength=5).astype(float)
    print(f"      combined class counts: {dict(zip(CLASS_NAMES_5, counts.astype(int).tolist()))}")
    counts = np.where(counts == 0, 1, counts)
    class_w = 1.0 / counts
    class_w[UC_MODSEV] *= args.modsev_weight     # extra push on the failing class
    sampler = WeightedRandomSampler([class_w[l] for l in labels],
                                    num_samples=len(labels), replacement=True)
    class_weights = torch.tensor(class_w / class_w.sum(), dtype=torch.float32).to(device)

    train_dl = DataLoader(full_train, batch_size=args.batch_size, sampler=sampler,
                          num_workers=2, drop_last=True)
    val_dl = DataLoader(hk_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"      train batches={len(train_dl)} val batches={len(val_dl)}")

    # ── Models (student fine-tuned, teacher frozen for staging/risk anchor) ──
    print(f"[2/6] Loading student + teacher from {args.ckpt_in} …")
    student = UnifiedMultiModalTransformer(n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    teacher = UnifiedMultiModalTransformer(n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    state = torch.load(args.ckpt_in, map_location="cpu")
    state = state.get("model_state", state)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Losses + optim ───────────────────────────────────────────────────────
    focal = FocalLoss(gamma=args.gamma, label_smoothing=0.1, class_weights=class_weights)
    attn_loss = AttentionMaskLoss(student).to(device)
    bert_params = [p for n, p in student.named_parameters() if "bert" in n]
    other = [p for n, p in student.named_parameters() if "bert" not in n]
    optim = torch.optim.AdamW([{"params": bert_params, "lr": args.bert_lr},
                               {"params": other, "lr": args.lr}],
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs * len(train_dl)), eta_min=args.lr * 0.1)

    def distill_kl(s, t, Tk):
        return F.kl_div(F.log_softmax(s / Tk, -1), F.softmax(t / Tk, -1),
                        reduction="batchmean") * (Tk * Tk)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"[3/6] Fine-tuning {args.epochs} epoch(s) …")
    Path(args.ckpt_out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(args.epochs):
        student.train()
        run = {"loss": 0, "cls": 0, "aux": 0, "n": 0}
        for step, b in enumerate(train_dl):
            if args.max_train_batches and step >= args.max_train_batches:
                break
            img = b["image"].to(device); ids = b["input_ids"].to(device)
            msk = b["attention_mask"].to(device); tab = b["tabular"].to(device)
            lbl = b["label"].to(device); pmsk = b["mask"].to(device); hmsk = b["has_mask"].to(device)
            optim.zero_grad()
            with torch.no_grad():
                t_out = teacher(img, ids, msk, tab)
            s_out = student(img, ids, msk, tab)
            l_cls = focal(s_out["pathology"], lbl)
            l_aux = uc_aux_loss(s_out["pathology"], lbl, device, args.aux_modsev_w)
            l_stg = distill_kl(s_out["staging"], t_out["staging"], args.distill_T)
            l_rsk = distill_kl(s_out["risk"], t_out["risk"], args.distill_T)
            l_attn = attn_loss(pmsk, hmsk) if hmsk.sum() > 0 else torch.tensor(0.0, device=device)
            loss = (l_cls + args.aux_w * l_aux + args.distill_w * (l_stg + l_rsk)
                    + args.mask_loss_w * l_attn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step(); sched.step()
            run["loss"] += loss.item(); run["cls"] += l_cls.item()
            run["aux"] += float(l_aux); run["n"] += 1
            if step % 20 == 0:
                print(f"  ep{ep+1} step{step:4d}/{len(train_dl)} "
                      f"loss={loss.item():.3f} cls={l_cls.item():.3f} aux={float(l_aux):.3f}")
        if run["n"]:
            print(f"  ep{ep+1} mean loss={run['loss']/run['n']:.3f} "
                  f"cls={run['cls']/run['n']:.3f} aux={run['aux']/run['n']:.3f}")

    # ── Calibrate + evaluate ──────────────────────────────────────────────────
    print("[4/6] Calibrating temperature …")
    Tcal = calibrate_temperature(student, val_dl, device)
    print(f"      T = {Tcal:.3f}")
    print("[5/6] Evaluating on HyperKvasir val (same split as baseline) …")
    metrics = evaluate(student, val_dl, device, T_scale=Tcal)
    metrics["temperature"] = Tcal
    metrics["config"] = vars(args)
    print(json.dumps(metrics["per_class"], indent=2))
    uc = metrics["per_class"]["uc-moderate-sev"]
    print(f"\n  >>> uc-moderate-sev recall: {uc['recall']:.3f} "
          f"(baseline 0.154) | overall acc {metrics['overall_acc']:.3f} "
          f"macro-F1 {metrics['macro_f1']:.3f}")

    # ── Save (to a NEW path — does not overwrite the working v2) ───────────────
    print(f"[6/6] Saving -> {args.ckpt_out}")
    torch.save({"model_state": student.state_dict(),
                "classes": list(CLASS_NAMES_5), "temperature": Tcal,
                "note": "uc-fix fine-tune (LIMUC + UC-focused loss)"}, args.ckpt_out)
    Path(METRICS_OUT).write_text(json.dumps(metrics, indent=2))
    attn_loss.close()
    print(f"      metrics -> {METRICS_OUT}  ·  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
