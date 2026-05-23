"""Grid-search the temperature T that minimises Expected Calibration
Error (ECE) on the val set, then save it as temperature.json.

The training-time T was fit to minimise NLL (the standard recipe), but
that doesn't necessarily minimise ECE. This script does the direct
optimisation."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (N_TABULAR_FEATURES,
    HyperKvasirMultiModalDataset, CLASS_NAMES_5)

CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
OUT  = Path("outputs/unified_multimodal_v2/temperature.json")


def softmax_T(x, T):
    z = x / T; e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ece(p, y, n_bins=15):
    c = p.max(1); pr = p.argmax(1); acc = (pr == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(c)
    for i in range(n_bins):
        m = (c > bins[i]) & (c <= bins[i+1])
        if m.sum() == 0: continue
        e += (m.sum() / N) * abs(acc[m].mean() - c[m].mean())
    return float(e)


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    m = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    s = torch.load(CKPT, map_location=device)
    m.load_state_dict(s.get("model_state", s), strict=False); m.eval()

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    tcga = pd.read_csv("data/raw/tcga/clinical/clinical.tsv", sep="\t") \
           if Path("data/raw/tcga/clinical/clinical.tsv").exists() else None
    ds = HyperKvasirMultiModalDataset(
        root_dir="data/processed/hyper_kvasir_clean",
        tokenizer=tok, tcga_df=tcga, split="val",
        val_ratio=0.15, test_ratio=0.10, seed=42)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    logits, labels = [], []
    with torch.no_grad():
        for b in dl:
            out = m(b["image"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["tabular"].to(device))
            logits.append(out["pathology"].cpu().numpy())
            labels.append(b["label"].numpy())
    logits = np.vstack(logits); labels = np.concatenate(labels)

    best_T, best_e = 1.0, 1.0
    for T in np.linspace(0.3, 4.0, 75):
        e = ece(softmax_T(logits, T), labels)
        if e < best_e: best_e, best_T = e, float(T)
    raw_ece = ece(softmax_T(logits, 1.0), labels)
    print(f"Raw ECE        : {raw_ece:.4f}")
    print(f"Best T (ECE-min): {best_T:.3f}   ECE = {best_e:.4f}")

    OUT.write_text(json.dumps({
        "temperature":   float(best_T),
        "method":        "grid-search to minimise ECE",
        "ece_at_T":      float(best_e),
        "ece_raw":       float(raw_ece),
        "previous_T":    1.309,
    }, indent=2))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
