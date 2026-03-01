"""
Training Script — Unified Multi-Modal Transformer
──────────────────────────────────────────────────
Anti-overfitting measures:
  ✓ Label smoothing (0.1)
  ✓ Mixup (alpha=0.2)
  ✓ Stochastic depth / DropPath
  ✓ WeightedRandomSampler (balanced classes)
  ✓ AdamW + cosine annealing with warmup
  ✓ Gradient clipping (max_norm=1.0)
  ✓ Early stopping (patience=10)
  ✓ Multi-task regularisation
  ✓ BioBERT bottom layers frozen
  ✓ Progressive unfreezing of text encoder
  ✓ EMA (Exponential Moving Average) weights

Run:
    python experiments/train_unified_multimodal.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from src.models.unified_transformer import (
    UnifiedMultiModalTransformer, MultiTaskLoss, mixup_batch)
from src.data.multimodal_dataset import (
    build_dataloaders, N_TABULAR_FEATURES, HYPERKVASIR_CLASS_MAP)


# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed/hyper_kvasir_clean",
                   help="Path to HyperKvasir processed directory")
    p.add_argument("--tcga_dir", default="data/raw/tcga",
                   help="Path to TCGA data directory")
    p.add_argument("--output_dir", default="outputs/unified_multimodal")
    p.add_argument("--bert_model", default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bert_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_fusion_heads", type=int, default=8)
    p.add_argument("--n_fusion_layers", type=int, default=4)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--warmup_pct", type=float, default=0.1)
    p.add_argument("--unfreeze_epoch", type=int, default=5,
                   help="Epoch to unfreeze top BERT layers")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--no_pretrained", action="store_true")
    args = p.parse_args()
    return args


# ──────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ──────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def progressive_unfreeze(model: UnifiedMultiModalTransformer, epoch: int,
                         unfreeze_epoch: int):
    """Unfreeze top 4 BERT layers after unfreeze_epoch."""
    if epoch == unfreeze_epoch:
        for i, layer in enumerate(model.text_encoder.bert.encoder.layer):
            if i >= 8:   # unfreeze layers 8-11
                for p in layer.parameters():
                    p.requires_grad = True
        print(f"[Train] Epoch {epoch}: Unfroze BERT layers 8-11.")


# ──────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion,
                device, mixup_alpha, ema):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        labels = batch["label"].to(device)

        # Mixup
        if mixup_alpha > 0:
            batch_mixed, lam, idx = mixup_batch(
                {"image": image, "tabular": tabular,
                 "input_ids": input_ids, "attention_mask": attention_mask,
                 "label": labels}, alpha=mixup_alpha)
            image = batch_mixed["image"]
            tabular = batch_mixed["tabular"]
        else:
            lam, idx = 1.0, None

        optimizer.zero_grad()

        out = model(image, input_ids, attention_mask, tabular)
        loss_dict = criterion(out, labels, lam=lam, idx=idx)
        loss = loss_dict["total"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema.update()

        total_loss += loss.item()

        preds = out["pathology"].argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if step % 20 == 0:
            print(f"  [Step {step:4d}] loss={loss.item():.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


# ──────────────────────────────────────────────────
# VALIDATE
# ──────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, ema=None, use_ema=False):
    if use_ema and ema is not None:
        ema.apply_shadow()

    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        labels = batch["label"].to(device)

        out = model(image, input_ids, attention_mask, tabular)
        loss_dict = criterion(out, labels)
        total_loss += loss_dict["total"].item()

        probs = torch.softmax(out["pathology"], dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    if use_ema and ema is not None:
        ema.restore()

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr",
                            average="macro")
    except Exception:
        auc = 0.0

    return total_loss / len(loader), acc, f1, auc, all_preds, all_labels, all_probs


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
def main():
    cfg = get_config()
    seed_everything(cfg.seed)
    device = get_device()
    print(f"[Train] Device: {device}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # ── Tokenizer ────────────────────────────────────
    print(f"[Train] Loading tokenizer: {cfg.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model)

    # ── Data ─────────────────────────────────────────
    print(f"[Train] Building dataloaders from {cfg.data_dir}")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = build_dataloaders(
        hyperkvasir_dir=cfg.data_dir,
        tokenizer=tokenizer,
        tcga_dir=cfg.tcga_dir,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        max_seq_len=cfg.max_seq_len,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"[Train] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")

    class_weights = train_ds.get_class_weights().to(device)

    # ── Model ─────────────────────────────────────────
    print("[Train] Building UnifiedMultiModalTransformer ...")
    model = UnifiedMultiModalTransformer(
        bert_model_name=cfg.bert_model,
        n_tabular_features=N_TABULAR_FEATURES,
        d_model=cfg.d_model,
        n_fusion_heads=cfg.n_fusion_heads,
        n_fusion_layers=cfg.n_fusion_layers,
        pretrained_backbone=not cfg.no_pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {n_params:,}")

    # ── Criterion ─────────────────────────────────────
    criterion = MultiTaskLoss(
        w_path=0.5, w_stage=0.3, w_risk=0.2,
        smoothing=cfg.label_smoothing,
        class_weights=class_weights,
    )

    # ── Optimizer: different LRs for backbone vs BERT vs rest ─
    bert_params = list(model.text_encoder.bert.parameters())
    bert_ids = {id(p) for p in bert_params}
    backbone_params = list(model.image_encoder.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters()
                    if id(p) not in bert_ids | backbone_ids]

    optimizer = AdamW([
        {"params": bert_params, "lr": cfg.bert_lr, "weight_decay": 0.01},
        {"params": backbone_params, "lr": cfg.lr * 0.5, "weight_decay": cfg.weight_decay},
        {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
    ], eps=1e-8)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=[cfg.bert_lr, cfg.lr * 0.5, cfg.lr],
        total_steps=total_steps,
        pct_start=cfg.warmup_pct,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    ema = EMA(model, decay=cfg.ema_decay)

    # ── Training Loop ──────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [],
               "val_acc": [], "val_f1": [], "val_auc": []}
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\n[Train] Starting training ...")
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n── Epoch {epoch}/{cfg.epochs} ──")

        # Progressive BERT unfreezing
        progressive_unfreeze(model, epoch, cfg.unfreeze_epoch)

        # Train
        t_loss, t_acc, t_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, cfg.mixup_alpha, ema)

        # Validate (with EMA weights)
        v_loss, v_acc, v_f1, v_auc, _, _, _ = validate(
            model, val_loader, criterion, device, ema=ema, use_ema=True)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["val_auc"].append(v_auc)

        print(f"  Train — loss={t_loss:.4f}  acc={t_acc:.4f}  f1={t_f1:.4f}")
        print(f"  Val   — loss={v_loss:.4f}  acc={v_acc:.4f}  f1={v_f1:.4f}  auc={v_auc:.4f}")

        # Checkpoint
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = epoch
            patience_counter = 0

            ema.apply_shadow()
            ck_path = out_dir / "checkpoints" / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": v_f1,
                "val_auc": v_auc,
                "config": vars(cfg),
            }, ck_path)
            ema.restore()
            print(f"  ✓ New best model saved (val_f1={v_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(best was epoch {best_epoch})")
                break

    # ── Test Evaluation ─────────────────────────────────
    print("\n[Train] Loading best model for test evaluation ...")
    ck = torch.load(out_dir / "checkpoints" / "best_model.pth", map_location=device)
    model.load_state_dict(ck["model_state"])

    te_loss, te_acc, te_f1, te_auc, te_preds, te_labels, te_probs = validate(
        model, test_loader, criterion, device)

    print(f"\n{'═'*50}")
    print("TEST RESULTS")
    print(f"  Loss     : {te_loss:.4f}")
    print(f"  Accuracy : {te_acc:.4f}")
    print(f"  F1 Macro : {te_f1:.4f}")
    print(f"  AUC-ROC  : {te_auc:.4f}")
    print(f"{'═'*50}")

    cls_names = list(HYPERKVASIR_CLASS_MAP.keys())
    print("\n" + classification_report(te_labels, te_preds,
                                        target_names=cls_names, zero_division=0))

    # ── Save Artefacts ──────────────────────────────────
    # Metrics JSON
    metrics = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": te_acc,
        "test_f1_macro": te_f1,
        "test_auc_roc": te_auc,
        "test_loss": te_loss,
        "config": vars(cfg),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_x = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_x, history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(epochs_x, history["val_loss"], label="Val", color="#f44336")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs_x, history["train_acc"], label="Train", color="#2196F3")
    axes[1].plot(epochs_x, history["val_acc"], label="Val", color="#f44336")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs_x, history["val_f1"], label="F1", color="#4CAF50")
    axes[2].plot(epochs_x, history["val_auc"], label="AUC", color="#FF9800")
    axes[2].set_title("Val F1 & AUC"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(te_labels, te_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    tick_marks = range(len(cls_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(cls_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(cls_names)
    for i in range(len(cls_names)):
        for j in range(len(cls_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\n[Train] All artefacts saved to {out_dir}")
    return model, metrics


if __name__ == "__main__":
    main()
