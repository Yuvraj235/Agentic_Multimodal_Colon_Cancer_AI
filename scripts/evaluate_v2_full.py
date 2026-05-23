"""ColonAI v2 — full evaluation suite.

Runs the v2 checkpoint on the HyperKvasir validation split and produces:

  * Per-class precision / recall / F1 / support
  * Confusion matrix (raw counts + normalised)
  * Reliability diagram (raw + temperature-scaled)
  * Expected Calibration Error (ECE) before and after T-scaling
  * Brier score (multiclass)

Saves:
  outputs/unified_multimodal_v2/figures/fig_v2_confusion_matrix.png
  outputs/unified_multimodal_v2/figures/fig_v2_reliability.png
  outputs/unified_multimodal_v2/metrics_v2.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             classification_report)
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    N_TABULAR_FEATURES, HyperKvasirMultiModalDataset, CLASS_NAMES_5,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
TEMP = "outputs/unified_multimodal_v2/temperature.json"
FIG  = Path("outputs/unified_multimodal_v2/figures")
OUT  = Path("outputs/unified_multimodal_v2/metrics_v2.json")
FIG.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def expected_calibration_error(probs, labels, n_bins=15) -> float:
    """Standard ECE: weighted absolute confidence-accuracy gap."""
    confs = probs.max(axis=1); preds = probs.argmax(axis=1)
    accs  = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; N = len(confs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        in_bin = (confs > lo) & (confs <= hi)
        if in_bin.sum() == 0: continue
        ece += (in_bin.sum() / N) * abs(accs[in_bin].mean() - confs[in_bin].mean())
    return float(ece)


def brier_multiclass(probs, labels) -> float:
    n_cls = probs.shape[1]
    one_hot = np.zeros_like(probs); one_hot[np.arange(len(labels)), labels] = 1
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    T = json.loads(Path(TEMP).read_text()).get("temperature", 1.0) \
        if Path(TEMP).exists() else 1.0
    print(f"Temperature: {T:.3f}")

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    tcga_p = Path("data/raw/tcga/clinical/clinical.tsv")
    tcga_df = pd.read_csv(tcga_p, sep="\t") if tcga_p.exists() else None
    val_ds = HyperKvasirMultiModalDataset(
        root_dir="data/processed/hyper_kvasir_clean",
        tokenizer=tok, tcga_df=tcga_df, split="val",
        val_ratio=0.15, test_ratio=0.10, seed=42)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"\nInferring on {len(val_ds)} validation samples …")
    raw_probs, cal_probs, labels = [], [], []
    with torch.no_grad():
        for i, b in enumerate(val_dl):
            out = model(b["image"].to(device), b["input_ids"].to(device),
                        b["attention_mask"].to(device), b["tabular"].to(device))
            lo = out["pathology"]
            raw_probs.append(F.softmax(lo,        dim=-1).cpu().numpy())
            cal_probs.append(F.softmax(lo / T,    dim=-1).cpu().numpy())
            labels.append(b["label"].numpy())
            if (i+1) % 10 == 0:
                print(f"  batch {i+1}/{len(val_dl)}")
    raw_probs = np.vstack(raw_probs)
    cal_probs = np.vstack(cal_probs)
    labels    = np.concatenate(labels)
    preds     = raw_probs.argmax(axis=1)

    # ── Per-class P/R/F1 + confusion matrix ────────────────────────────
    print("\n--- Per-class metrics ---")
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(5)), zero_division=0)
    rows = []
    print(f"  {'class':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>8}")
    for i, name in enumerate(CLASS_NAMES_5):
        rows.append({"class": name, "precision": float(p[i]),
                     "recall": float(r[i]), "f1": float(f1[i]),
                     "support": int(sup[i])})
        print(f"  {name:<22}  {p[i]:6.3f}  {r[i]:6.3f}  {f1[i]:6.3f}  {sup[i]:8d}")

    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES_5, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES_5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("v2 — Confusion matrix (row-normalised)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if cm_norm[i,j] > 0.5 else "#1e293b")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v2_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {FIG/'fig_v2_confusion_matrix.png'}")

    # ── Calibration ─────────────────────────────────────────────────────
    print("\n--- Calibration ---")
    ece_raw = expected_calibration_error(raw_probs, labels)
    ece_cal = expected_calibration_error(cal_probs, labels)
    brier_raw = brier_multiclass(raw_probs, labels)
    brier_cal = brier_multiclass(cal_probs, labels)
    print(f"  ECE raw    : {ece_raw:.4f}")
    print(f"  ECE T={T:.3f}: {ece_cal:.4f}  (Δ {ece_cal-ece_raw:+.4f})")
    print(f"  Brier raw  : {brier_raw:.4f}")
    print(f"  Brier cal  : {brier_cal:.4f}")

    # Reliability diagram
    bins = np.linspace(0, 1, 16)
    def bin_acc(probs):
        conf = probs.max(axis=1); pred = probs.argmax(axis=1)
        acc  = (pred == labels).astype(float)
        accs, confs, sizes = [], [], []
        for i in range(len(bins) - 1):
            in_b = (conf > bins[i]) & (conf <= bins[i+1])
            if in_b.sum() == 0:
                accs.append(np.nan); confs.append(np.nan); sizes.append(0)
            else:
                accs.append(acc[in_b].mean())
                confs.append(conf[in_b].mean())
                sizes.append(int(in_b.sum()))
        return np.array(accs), np.array(confs), np.array(sizes)

    a_raw, c_raw, s_raw = bin_acc(raw_probs)
    a_cal, c_cal, s_cal = bin_acc(cal_probs)

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect calibration")
    # mask NaN bins (no samples)
    ok_r = ~np.isnan(c_raw)
    ok_c = ~np.isnan(c_cal)
    ax.plot(c_raw[ok_r], a_raw[ok_r], "o-", color="#94a3b8",
            label=f"Raw  (ECE = {ece_raw:.3f})", markersize=7)
    ax.plot(c_cal[ok_c], a_cal[ok_c], "o-", color="#0ea5e9",
            label=f"After T = {T:.3f}  (ECE = {ece_cal:.3f})", markersize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Actual accuracy in bin")
    ax.set_title("v2 — Reliability diagram on held-out val set")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v2_reliability.png", dpi=200, bbox_inches="tight")
    plt.close()
    # Replace the earlier synthetic calibration figure
    import shutil
    shutil.copy(FIG / "fig_v2_reliability.png",
                FIG / "fig_v2_calibration_curve.png")
    print(f"  ✓ {FIG/'fig_v2_reliability.png'}")

    # ── Save metrics ────────────────────────────────────────────────────
    metrics = {
        "checkpoint": CKPT,
        "temperature": T,
        "n_val_samples": len(labels),
        "overall_acc": float((preds == labels).mean()),
        "ece_raw": ece_raw, "ece_calibrated": ece_cal,
        "brier_raw": brier_raw, "brier_calibrated": brier_cal,
        "macro_f1": float(f1.mean()),
        "per_class": rows,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalised": cm_norm.tolist(),
    }
    OUT.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics → {OUT}")

    print(f"\n{'='*60}\nv2 EVALUATION SUMMARY\n{'='*60}")
    print(f"  Overall accuracy : {metrics['overall_acc']:.4f}")
    print(f"  Macro-F1         : {metrics['macro_f1']:.4f}")
    print(f"  ECE raw          : {ece_raw:.4f}")
    print(f"  ECE calibrated   : {ece_cal:.4f}")
    print(f"  Brier raw        : {brier_raw:.4f}")
    print(f"  Brier calibrated : {brier_cal:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
