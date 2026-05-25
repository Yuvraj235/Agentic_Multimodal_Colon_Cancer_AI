"""ColonAI — fit per-class abstention thresholds from val data.

The patient-safety policy currently uses a single global `min_confidence
= 0.75` to decide show/abstain. That's coarse — different classes have
different precision-recall trade-offs. For example, uc-mild has high
recall (0.95) but low precision (0.29) so we should require a HIGHER
confidence before showing a uc-mild call.

This script fits a per-class threshold that targets a specified
precision floor (default 0.85). It runs on the held-out val set,
sweeps thresholds for each class, and picks the one where precision
crosses 0.85 (and recall doesn't collapse).

Output: outputs/unified_multimodal_v2/per_class_thresholds.json

Run:
    python3 scripts/calibrate_per_class_thresholds.py
    python3 scripts/calibrate_per_class_thresholds.py --target_precision 0.90
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (
    N_TABULAR_FEATURES, HyperKvasirMultiModalDataset, CLASS_NAMES_5)
from src.app.security import safe_torch_load


CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
TEMP = "outputs/unified_multimodal_v2/temperature.json"
OUT  = Path("outputs/unified_multimodal_v2/per_class_thresholds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_precision", type=float, default=0.85)
    ap.add_argument("--min_recall_floor", type=float, default=0.10,
                    help="Don't pick a threshold that drops recall below this")
    ap.add_argument("--global_floor",     type=float, default=0.50,
                    help="Never recommend a threshold below this")
    ap.add_argument("--global_ceiling",   type=float, default=0.95,
                    help="Cap to keep at least some predictions")
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    # Load model + temperature
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    state = safe_torch_load(CKPT, map_location=device, allow_unsafe=True)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    T = (float(json.loads(Path(TEMP).read_text()).get("temperature", 1.0))
         if Path(TEMP).exists() else 1.0)
    print(f"Temperature: {T:.3f}")

    # Val set
    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    tcga_p = Path("data/raw/tcga/clinical/clinical.tsv")
    tcga_df = pd.read_csv(tcga_p, sep="\t") if tcga_p.exists() else None
    val_ds = HyperKvasirMultiModalDataset(
        root_dir="data/processed/hyper_kvasir_clean", tokenizer=tok,
        tcga_df=tcga_df, split="val", val_ratio=0.15, test_ratio=0.10, seed=42)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Collect calibrated probs + labels
    print(f"\nInferring on {len(val_ds)} val samples …")
    all_p, all_y = [], []
    with torch.no_grad():
        for i, b in enumerate(val_dl):
            out = model(b["image"].to(device), b["input_ids"].to(device),
                        b["attention_mask"].to(device), b["tabular"].to(device))
            p = F.softmax(out["pathology"] / T, dim=-1).cpu().numpy()
            all_p.append(p)
            all_y.append(b["label"].numpy())
            if (i + 1) % 10 == 0: print(f"  batch {i+1}/{len(val_dl)}")
    P = np.vstack(all_p)
    Y = np.concatenate(all_y)
    print(f"P.shape = {P.shape}, n_classes = {P.shape[1]}")

    # ── Per-class threshold sweep ──────────────────────────────────────
    print(f"\nFitting thresholds (target_precision = {args.target_precision}) …")
    thresholds = {}
    summary = {}
    for c_idx, c_name in enumerate(CLASS_NAMES_5):
        # For each sample, what's the probability of this class?
        probs_c = P[:, c_idx]
        # True positives = where Y == c_idx
        y_is_c = (Y == c_idx)
        if y_is_c.sum() == 0:
            thresholds[c_name] = args.global_floor
            summary[c_name] = {"chosen": args.global_floor,
                               "reason": "no positive examples in val",
                               "support": 0}
            continue
        # Sweep threshold candidates
        best_t = args.global_floor
        best_score = -1.0
        candidates = np.linspace(args.global_floor, args.global_ceiling, 100)
        per_t = []
        for t in candidates:
            pred_c = probs_c >= t
            tp = int(((pred_c) & (y_is_c)).sum())
            fp = int(((pred_c) & (~y_is_c)).sum())
            fn = int(((~pred_c) & (y_is_c)).sum())
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            per_t.append({"t": float(t), "prec": float(prec), "rec": float(rec),
                          "tp": tp, "fp": fp, "fn": fn})
            # Pick the SMALLEST threshold whose precision ≥ target AND recall ≥ floor.
            # Smallest threshold preserves the most predictions while meeting precision.
            if prec >= args.target_precision and rec >= args.min_recall_floor:
                # We want the smallest t — so break on first hit
                if best_score < 0:
                    best_t = float(t)
                    best_score = prec
                    break
        # Fallback: if no threshold meets target, pick the highest-precision one above
        # min_recall_floor.
        if best_score < 0:
            for r in per_t:
                if r["rec"] >= args.min_recall_floor and r["prec"] > best_score:
                    best_t = r["t"]; best_score = r["prec"]
        thresholds[c_name] = float(best_t)
        # Find what we'd get at this threshold
        at_chosen = next((r for r in per_t if abs(r["t"] - best_t) < 1e-4), per_t[-1])
        summary[c_name] = {
            "chosen":        best_t,
            "precision":     at_chosen["prec"],
            "recall":        at_chosen["rec"],
            "support":       int(y_is_c.sum()),
            "tp_at_chosen":  at_chosen["tp"],
            "fp_at_chosen":  at_chosen["fp"],
            "fn_at_chosen":  at_chosen["fn"],
        }
        print(f"  {c_name:18s}  t = {best_t:.3f}  "
              f"prec = {at_chosen['prec']:.3f}  rec = {at_chosen['rec']:.3f}  "
              f"({int(y_is_c.sum())} positives)")

    out_obj = {
        "method":            "per-class precision-targeting threshold sweep",
        "target_precision":  args.target_precision,
        "min_recall_floor":  args.min_recall_floor,
        "global_min":        args.global_floor,
        "global_max":        args.global_ceiling,
        "temperature":       T,
        "n_val_samples":     int(len(Y)),
        "thresholds":        thresholds,
        "per_class_details": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_obj, indent=2))
    print(f"\n✓ Saved per-class thresholds → {OUT}")
    print(f"\nTo use in the safety policy, set the env var:")
    print(f"   COLONAI_PER_CLASS_THRESHOLDS=outputs/unified_multimodal_v2/per_class_thresholds.json")


if __name__ == "__main__":
    main()
