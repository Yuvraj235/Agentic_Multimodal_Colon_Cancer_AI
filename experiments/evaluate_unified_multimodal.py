"""
Evaluation & Full Agentic Pipeline Demo
─────────────────────────────────────────
Loads the best trained UnifiedMultiModalTransformer checkpoint,
runs the complete 6-agent pipeline on the test set, and generates:
  • Per-sample XAI reports (Grad-CAM++, SHAP, modality weights)
  • Clinical recommendation letters
  • Aggregate metrics (accuracy, F1, AUC, per-class report)
  • Confusion matrix + ROC curves
  • Modality ablation study

Run:
    python experiments/evaluate_unified_multimodal.py \
        --checkpoint outputs/unified_multimodal/checkpoints/best_model.pth
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, auc as sklearn_auc)
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    build_dataloaders, N_TABULAR_FEATURES, HYPERKVASIR_CLASS_MAP)
from src.agents.multimodal_orchestrator import MultiModalOrchestrator


CLASS_NAMES = list(HYPERKVASIR_CLASS_MAP.keys())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="outputs/unified_multimodal/checkpoints/best_model.pth")
    p.add_argument("--data_dir", default="data/processed/hyper_kvasir_clean")
    p.add_argument("--tcga_dir", default="data/raw/tcga")
    p.add_argument("--output_dir", default="outputs/unified_multimodal/evaluation")
    p.add_argument("--bert_model", default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--n_demo_cases", type=int, default=3,
                   help="Number of cases to run full agentic pipeline on")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_fusion_heads", type=int, default=8)
    p.add_argument("--n_fusion_layers", type=int, default=4)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────
# BATCH METRICS
# ──────────────────────────────────────────────────
@torch.no_grad()
def run_batch_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    modality_weights_list = []

    for batch in tqdm(loader, desc="Evaluating"):
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        labels = batch["label"]

        out = model(image, input_ids, attention_mask, tabular)
        probs = F.softmax(out["pathology"], dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

        if "mod_weights" in out:
            modality_weights_list.extend(
                out["mod_weights"].cpu().numpy().tolist())

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), np.array(modality_weights_list))


# ──────────────────────────────────────────────────
# MODALITY ABLATION
# ──────────────────────────────────────────────────
@torch.no_grad()
def modality_ablation(model, loader, device, n_batches=20):
    """Ablate each modality and measure accuracy drop."""
    results = {}
    model.eval()

    def run_acc(ablate_img=False, ablate_txt=False, ablate_tab=False):
        preds_all, labs_all = [], []
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["label"]

            if ablate_img:
                image = torch.zeros_like(image)
            if ablate_txt:
                input_ids = torch.ones_like(input_ids)  # all [UNK]
            if ablate_tab:
                tabular = torch.zeros_like(tabular)

            out = model(image, input_ids, attention_mask, tabular)
            preds = out["pathology"].argmax(dim=-1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labs_all.extend(labels.numpy().tolist())

        from sklearn.metrics import accuracy_score
        return accuracy_score(labs_all, preds_all)

    results["all_modalities"] = run_acc()
    results["ablate_image"] = run_acc(ablate_img=True)
    results["ablate_text"] = run_acc(ablate_txt=True)
    results["ablate_tabular"] = run_acc(ablate_tab=True)

    print("\n[Ablation Study]")
    for k, v in results.items():
        print(f"  {k:25s}: {v:.4f}")

    return results


# ──────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────
def plot_roc_curves(labels, probs, class_names, out_path):
    from sklearn.preprocessing import label_binarize
    lb = label_binarize(labels, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"]
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Per Class")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Eval] ROC curves saved to {out_path}")


def plot_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_modality_weights(weights_arr, out_path):
    mean_w = weights_arr.mean(axis=0)
    labels = ["Image\n(ConvNeXt-V2)", "Text\n(BioBERT)", "Tabular\n(TabTransformer)"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, mean_w, color=colors)
    ax.set_ylabel("Average Modality Weight")
    ax.set_title("Cross-Modal Attention — Modality Importance")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, mean_w):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ablation(ablation, out_path):
    keys = list(ablation.keys())
    vals = [ablation[k] for k in keys]
    colors = ["#4CAF50", "#f44336", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title("Modality Ablation Study")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
def main():
    args = get_args()
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # ── Data ─────────────────────────────────────────
    train_loader, val_loader, test_loader, _, _, test_ds = build_dataloaders(
        hyperkvasir_dir=args.data_dir,
        tokenizer=tokenizer,
        tcga_dir=args.tcga_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_seq_len=args.max_seq_len,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────
    model = UnifiedMultiModalTransformer(
        bert_model_name=args.bert_model,
        n_tabular_features=N_TABULAR_FEATURES,
        d_model=args.d_model,
        n_fusion_heads=args.n_fusion_heads,
        n_fusion_layers=args.n_fusion_layers,
        pretrained_backbone=False,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ck["model_state"])
        print(f"[Eval] Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[Eval] WARNING: Checkpoint not found at {args.checkpoint}. "
              f"Using randomly initialised weights for demo.")

    # ── Batch Evaluation ─────────────────────────────
    print("\n[Eval] Running batch evaluation on test set ...")
    labels, preds, probs, mod_weights = run_batch_evaluation(
        model, test_loader, device)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    print(f"\n{'═'*50}")
    print("FINAL TEST METRICS")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Macro  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"{'═'*50}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    # Save metrics
    metrics = {"accuracy": acc, "f1_macro": f1, "auc_roc": auc}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────
    plot_roc_curves(labels, probs, CLASS_NAMES, out_dir / "roc_curves.png")
    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                          out_dir / "confusion_matrix.png")
    if len(mod_weights) > 0:
        plot_modality_weights(mod_weights, out_dir / "modality_weights.png")

    # ── Modality Ablation ─────────────────────────────
    print("\n[Eval] Running modality ablation study ...")
    ablation = modality_ablation(model, test_loader, device)
    with open(out_dir / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
    plot_ablation(ablation, out_dir / "ablation.png")

    # ── Demo: Full Agentic Pipeline ───────────────────
    print(f"\n[Eval] Running full agentic pipeline on {args.n_demo_cases} demo cases ...")
    orchestrator = MultiModalOrchestrator(
        model, tokenizer, device,
        output_dir=str(out_dir / "agent_cases"))

    for i, batch in enumerate(test_loader):
        if i >= args.n_demo_cases:
            break
        image = batch["image"][[0]]
        input_ids = batch["input_ids"][[0]]
        attention_mask = batch["attention_mask"][[0]]
        tabular = batch["tabular"][[0]]
        label = batch["label"][0].item()

        text = f"Colonoscopy sample from test case {i}. Label: {CLASS_NAMES[label]}."

        orchestrator.run(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tabular=tabular,
            text=text,
            case_id=f"case_{i:03d}",
            save=True,
        )

    print(f"\n[Eval] All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
