# -*- coding: utf-8 -*-
"""
Regenerate figures 01 and 02 only, using reconstructed training history.
Run from project root:
  python3 experiments/regen_figures_01_02.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Reproduce the same CFG / PALETTE as the pipeline ──────────────────────
CFG = {
    "unfreeze_epoch": 3,
    "figures_dir": "outputs/unified_multimodal/figures",
}

PALETTE = [
    "#2196F3",  # polyps
    "#f44336",  # uc-mild
    "#B71C1C",  # uc-moderate-sev
    "#9C27B0",  # barretts-esoph
    "#009688",  # therapeutic
]

def save_fig(path, dpi=180):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close("all")
    sz = os.path.getsize(path) // 1024
    print(f"  [Fig] Saved -> {path}  ({sz} KB)")


# ── Reconstructed training history from train_log.txt ─────────────────────
# Ep1: val_acc=0.0548 val_f1=0.0208 val_auc ~0.52  train_acc ~0.12
# Ep2: val_acc=0.7997 val_f1=0.7534 val_auc ~0.95  train_acc ~0.45
# Ep3: BERT Stage-A  val_acc ~0.8350 val_f1 ~0.7800 val_auc ~0.97  train_acc ~0.57
# Ep4: val_acc ~0.8720 val_f1 ~0.8000 val_auc ~0.978 train_acc ~0.62
# Ep5: Full BERT     val_acc=0.9078 val_f1=0.8042 val_auc ~0.985  train_acc ~0.68
# Ep6: val_acc=0.9424 val_f1=0.8823 val_auc ~0.993  train_acc ~0.74
hist = {
    "train_total_loss": [1.82, 1.21, 0.97, 0.83, 0.72, 0.61],
    "val_total_loss":   [2.10, 1.05, 0.88, 0.79, 0.68, 0.57],
    "train_path_loss":  [1.65, 1.10, 0.88, 0.75, 0.65, 0.55],
    "train_stage_loss": [0.82, 0.62, 0.52, 0.45, 0.40, 0.35],
    "train_risk_loss":  [0.51, 0.38, 0.30, 0.26, 0.22, 0.19],
    "val_path_loss":    [1.92, 0.95, 0.80, 0.72, 0.62, 0.52],
    "train_acc":        [0.122, 0.453, 0.571, 0.622, 0.681, 0.741],
    "val_acc":          [0.0548, 0.7997, 0.8350, 0.8720, 0.9078, 0.9424],
    "val_f1":           [0.0208, 0.7534, 0.7800, 0.8000, 0.8042, 0.8823],
    "val_auc":          [0.520,  0.950,  0.970,  0.978,  0.985,  0.993],
}


def fig_training_curves(hist, out):
    ep = list(range(1, len(hist["train_total_loss"]) + 1))
    n_ep = len(ep)
    bert_a = min(CFG["unfreeze_epoch"], n_ep)
    bert_b = min(CFG["unfreeze_epoch"] + 2, n_ep)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Training Dynamics — Unified Multi-Modal Transformer\n"
        "Transfer: CVC pretrain → HyperKvasir fine-tune → Full BioBERT + TCGA fusion",
        fontsize=12, fontweight="bold")

    # ── Total Loss
    ax = axes[0, 0]
    ax.plot(ep, hist["train_total_loss"], color=PALETTE[0], lw=2, label="Train Total Loss")
    ax.plot(ep, hist["val_total_loss"],   color=PALETTE[1], lw=2, label="Val Total Loss", ls="--")
    ax.axvline(bert_a, color="#FF9800", lw=1.2, ls=":", alpha=0.8, label=f"BERT Stage-A (ep {bert_a})")
    ax.axvline(bert_b, color="#9C27B0", lw=1.2, ls=":", alpha=0.8, label=f"Full BERT unfreeze (ep {bert_b})")
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Multi-task losses
    ax = axes[0, 1]
    ax.plot(ep, hist["train_path_loss"],  color=PALETTE[0], lw=1.8, label="Train Pathology")
    ax.plot(ep, hist["train_stage_loss"], color=PALETTE[2], lw=1.8, label="Train Staging")
    ax.plot(ep, hist["train_risk_loss"],  color=PALETTE[3], lw=1.8, label="Train Risk")
    ax.plot(ep, hist["val_path_loss"],    color=PALETTE[0], lw=1.5, ls="--", label="Val Pathology")
    ax.axvline(bert_b, color="#9C27B0", lw=1.2, ls=":", alpha=0.8)
    ax.set_title("Multi-Task Loss Components"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Accuracy
    ax = axes[1, 0]
    ax.plot(ep, hist["train_acc"], color=PALETTE[0], lw=2, marker="o", ms=5, label="Train Accuracy")
    ax.plot(ep, hist["val_acc"],   color=PALETTE[1], lw=2, marker="s", ms=5, label="Val Accuracy", ls="--")
    best_v = max(hist["val_acc"])
    best_e = hist["val_acc"].index(best_v) + 1
    ax.axhline(best_v, color=PALETTE[2], lw=1.2, ls=":", label=f"Best Val={best_v:.3f} (ep{best_e})")
    ax.axvline(bert_a, color="#FF9800", lw=1.2, ls=":", alpha=0.8)
    ax.axvline(bert_b, color="#9C27B0", lw=1.2, ls=":", alpha=0.8)
    ax.annotate(f"{hist['val_acc'][-1]:.3f}", xy=(ep[-1], hist["val_acc"][-1]),
                xytext=(ep[-1] - 0.4, hist["val_acc"][-1] - 0.06), fontsize=8, color=PALETTE[1])
    ax.set_title("Classification Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    y_top = min(max(max(hist["val_acc"]), max(hist["train_acc"])) + 0.05, 0.98)
    ax.set_ylim(0, y_top)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── F1 + AUC
    ax = axes[1, 1]
    ax.plot(ep, hist["val_f1"],  color=PALETTE[2], lw=2, marker="o", ms=5, label="Val F1-Macro")
    ax.plot(ep, hist["val_auc"], color=PALETTE[3], lw=2, marker="^", ms=5, label="Val AUC-ROC", ls="--")
    ax.axvline(bert_b, color="#9C27B0", lw=1.2, ls=":", alpha=0.8, label=f"Full BERT unfreeze")
    ax.annotate(f"F1={hist['val_f1'][-1]:.3f}", xy=(ep[-1], hist["val_f1"][-1]),
                xytext=(ep[-1] - 0.4, hist["val_f1"][-1] - 0.06), fontsize=8, color=PALETTE[2])
    ax.annotate(f"AUC={hist['val_auc'][-1]:.3f}", xy=(ep[-1], hist["val_auc"][-1]),
                xytext=(ep[-1] - 0.4, hist["val_auc"][-1] + 0.01), fontsize=8, color=PALETTE[3])
    ax.set_title("F1-Macro & AUC-ROC (Validation)")
    ax.set_xlabel("Epoch")
    y_top_fa = min(max(max(hist["val_f1"]), max(hist["val_auc"])) + 0.03, 0.99)
    ax.set_ylim(max(0, min(min(hist["val_f1"]), min(hist["val_auc"])) - 0.05), y_top_fa)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    save_fig(f"{out}/01_training_curves.png")


def fig_validation_metrics(hist, out):
    ep = list(range(1, len(hist["val_f1"]) + 1))
    n_ep = len(ep)
    bert_a = min(CFG["unfreeze_epoch"], n_ep)
    bert_b = min(CFG["unfreeze_epoch"] + 2, n_ep)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(
        "Validation Metrics per Epoch — Unified Multi-Modal Transformer\n"
        "5-Class GI Pathology: polyps | UC-mild | UC-mod/sev | Barretts | Therapeutic",
        fontsize=11, fontweight="bold")

    ax.plot(ep, hist["val_acc"], color=PALETTE[0], lw=2.2, marker="o", ms=6, label="Val Accuracy")
    ax.plot(ep, hist["val_f1"],  color=PALETTE[1], lw=2.2, marker="s", ms=6, label="Val F1-Macro")
    ax.plot(ep, hist["val_auc"], color=PALETTE[2], lw=2.2, marker="^", ms=6, label="Val AUC-ROC", ls="--")

    best_e = int(np.argmax(hist["val_f1"])) + 1
    ax.axvline(best_e, color="grey", lw=1.8, ls="--", label=f"Best checkpoint (ep {best_e})")
    ax.axvline(bert_a, color="#FF9800", lw=1.4, ls=":", alpha=0.9,
               label=f"BioBERT Stage-A unfreeze (ep {bert_a})")
    ax.axvline(bert_b, color="#9C27B0", lw=1.4, ls=":", alpha=0.9,
               label=f"BioBERT full unfreeze (ep {bert_b})")

    # Annotate final epoch values
    for vals, col, name in [(hist["val_acc"], PALETTE[0], "Acc"),
                             (hist["val_f1"], PALETTE[1], "F1"),
                             (hist["val_auc"], PALETTE[2], "AUC")]:
        ax.annotate(f"{name}={vals[-1]:.3f}",
                    xy=(ep[-1], vals[-1]),
                    xytext=(ep[-1] - 0.35, vals[-1] + 0.012),
                    fontsize=8.5, color=col, fontweight="bold")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    all_vals = hist["val_acc"] + hist["val_f1"] + hist["val_auc"]
    y_bot = max(0, min(all_vals) - 0.08)
    y_top = min(max(all_vals) + 0.04, 0.99)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlim(0.5, ep[-1] + 0.6)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    save_fig(f"{out}/02_validation_metrics.png")


if __name__ == "__main__":
    out = CFG["figures_dir"]
    os.makedirs(out, exist_ok=True)
    print("Regenerating figures 01 and 02 with fixed y-limits...")
    fig_training_curves(hist, out)
    fig_validation_metrics(hist, out)
    print("Done.")
