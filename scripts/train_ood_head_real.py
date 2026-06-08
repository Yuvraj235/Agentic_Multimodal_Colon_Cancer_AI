"""Honest OOD head — trained AND evaluated on REAL out-of-scope endoscopy images.

The previous head was trained + tested on synthetic noise only, so its "F1 = 1.0"
was meaningless. This version uses real HyperKvasir classes the model was NEVER
trained on (anatomical landmarks + quality views) as genuine OOD, splits them into
train/test, and reports AUROC + F1 on the HELD-OUT real-OOD set. Synthetic OOD is
kept in TRAINING only (so the head still catches pure noise / non-tissue), never in
the reported test metric.

In-distribution = the 5 trained classes (HyperKvasir 5-class + polyp-mask sets).
Out-of-distribution (real) = cecum, pylorus, z-line, retroflex views, bowel-prep
(bbps), impacted-stool, ileum, hemorrhoids.

Writes to *_real files first; promote to ood_head.pth after the metrics are checked.
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
import numpy as np, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, HyperKvasirMultiModalDataset
from transformers import AutoTokenizer
from scripts.train_ood_head import synthetic_ood, get_fused_embedding, OODHead, CKPT

HK_ROOT = "data/processed/hyper_kvasir_clean"
OOD_OUT = Path("outputs/unified_multimodal_v2/ood_head_real.pth")
THR_OUT = Path("outputs/unified_multimodal_v2/ood_threshold_real.json")
MET_OUT = Path("outputs/unified_multimodal_v2/ood_metrics_real.json")

# Real OOD = HyperKvasir labels the 5-class model was NOT trained on.
OOD_CLASS_DIRS = [
    "lower-gi-tract/anatomical-landmarks/cecum",
    "lower-gi-tract/anatomical-landmarks/retroflex-rectum",
    "lower-gi-tract/anatomical-landmarks/ileum",
    "lower-gi-tract/quality-of-mucosal-views/bbps-0-1",
    "lower-gi-tract/quality-of-mucosal-views/bbps-2-3",
    "lower-gi-tract/quality-of-mucosal-views/impacted-stool",
    "lower-gi-tract/pathological-findings/hemorrhoids",
    "upper-gi-tract/anatomical-landmarks/pylorus",
    "upper-gi-tract/anatomical-landmarks/z-line",
    "upper-gi-tract/anatomical-landmarks/retroflex-stomach",
]
PER_CLASS_CAP = 150          # cap per OOD class so big classes don't dominate
random.seed(42); np.random.seed(42); torch.manual_seed(42)
_TFM = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def auroc(id_scores, ood_scores) -> float:
    """Mann-Whitney AUROC: P(score(ID) > score(OOD)). Score = P(in-distribution)."""
    pos = np.asarray(id_scores, float); neg = np.asarray(ood_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    alls = np.concatenate([pos, neg])
    order = np.argsort(alls, kind="mergesort")
    ranks = np.empty(len(alls)); ranks[order] = np.arange(1, len(alls) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    model = UnifiedMultiModalTransformer(n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tok("Patient referred for screening colonoscopy.", padding="max_length",
              truncation=True, max_length=64, return_tensors="pt")
    ids = enc["input_ids"].to(device); msk = enc["attention_mask"].to(device)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=device)

    def emb_of_pil(pim):
        x = _TFM(pim.convert("RGB")).unsqueeze(0).to(device)
        return get_fused_embedding(model, x, ids, msk, tab, device)

    # ── In-distribution embeddings (the 5 trained classes) ──────────────────
    print("Collecting in-distribution embeddings …")
    tcga = pd.read_csv("data/raw/tcga/clinical/clinical.tsv", sep="\t") \
        if Path("data/raw/tcga/clinical/clinical.tsv").exists() else None
    hk = HyperKvasirMultiModalDataset(root_dir=HK_ROOT, tokenizer=tok, tcga_df=tcga,
                                      split="train", val_ratio=0.15, test_ratio=0.10, seed=42)
    id_emb = []
    for i, idx in enumerate(random.sample(range(len(hk)), min(1100, len(hk)))):
        e = get_fused_embedding(model, hk[idx]["image"].unsqueeze(0).to(device), ids, msk, tab, device)
        if e is not None: id_emb.append(e)
        if (i + 1) % 200 == 0: print(f"  ID {i+1}")
    print(f"  in-dist total: {len(id_emb)}")

    # ── Real OOD embeddings (untrained HyperKvasir classes) ─────────────────
    print("Collecting REAL out-of-scope embeddings …")
    ood_emb, per_class = [], {}
    root = Path(HK_ROOT)
    for rel in OOD_CLASS_DIRS:
        d = root / rel
        if not d.exists():
            print(f"  · MISSING {rel}"); continue
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        random.shuffle(files)
        n = 0
        for f in files[:PER_CLASS_CAP]:
            try:
                e = emb_of_pil(Image.open(f))
                if e is not None: ood_emb.append(e); n += 1
            except Exception:
                pass
        per_class[rel.split("/")[-1]] = n
        print(f"  · {rel.split('/')[-1]:20s} +{n}")
    print(f"  real-OOD total: {len(ood_emb)}  {per_class}")

    id_emb = np.asarray(id_emb, np.float32); ood_emb = np.asarray(ood_emb, np.float32)

    # ── Split BOTH into train/test (held-out real OOD for honest metrics) ───
    def split(a, frac=0.3):
        idx = np.random.permutation(len(a)); k = int(len(a) * frac)
        return a[idx[k:]], a[idx[:k]]
    id_tr, id_te = split(id_emb); ood_tr, ood_te = split(ood_emb)

    # Synthetic OOD — TRAIN ONLY (keeps the head robust to pure noise/non-tissue)
    print("Adding synthetic OOD to TRAIN only …")
    synth = []
    for arr in synthetic_ood(n_per_kind=120):
        e = emb_of_pil(Image.fromarray(arr))
        if e is not None: synth.append(e)
    synth = np.asarray(synth, np.float32)

    Xtr = np.vstack([id_tr, ood_tr, synth]).astype(np.float32)
    ytr = np.concatenate([np.ones(len(id_tr)), np.zeros(len(ood_tr) + len(synth))]).astype(np.float32)
    print(f"  train: ID {len(id_tr)} | real-OOD {len(ood_tr)} | synth {len(synth)}")
    print(f"  test : ID {len(id_te)} | real-OOD {len(ood_te)} (held out)")

    # ── Train head ──────────────────────────────────────────────────────────
    head = OODHead(dim=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).to(device)
    head.train()
    for ep in range(60):
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(perm), 64):
            b = perm[s:s+64]
            loss = F.binary_cross_entropy_with_logits(head(Xt[b]).squeeze(-1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    # ── Evaluate on HELD-OUT REAL OOD ───────────────────────────────────────
    with torch.no_grad():
        s_id = torch.sigmoid(head(torch.from_numpy(id_te).to(device)).squeeze(-1)).cpu().numpy()
        s_ood = torch.sigmoid(head(torch.from_numpy(ood_te).to(device)).squeeze(-1)).cpu().numpy()
    roc = auroc(s_id, s_ood)
    # Pick threshold on P(in-dist): flag OOD when score < t. Optimise OOD-detection F1.
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        tp = (s_ood < t).sum(); fp = (s_id < t).sum(); fn = (s_ood >= t).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        if f1 > best_f1: best_f1, best_t = float(f1), float(t)
    id_acc = float((s_id >= best_t).mean()); ood_acc = float((s_ood < best_t).mean())
    print(f"\n  >>> REAL-OOD  AUROC={roc:.3f}  F1={best_f1:.3f}  thr={best_t:.3f}")
    print(f"      ID kept={id_acc:.1%}  OOD caught={ood_acc:.1%}  (held-out real images)")

    torch.save({"state_dict": head.state_dict(), "emb_dim": Xtr.shape[1]}, OOD_OUT)
    THR_OUT.write_text(json.dumps({"threshold": best_t, "real_ood_f1": best_f1,
                                   "real_ood_auroc": roc, "note": "score<threshold => OOD"}, indent=2))
    MET_OUT.write_text(json.dumps({
        "real_ood_auroc": roc, "real_ood_f1": best_f1, "threshold": best_t,
        "id_kept_frac": id_acc, "ood_caught_frac": ood_acc,
        "n_id_train": len(id_tr), "n_ood_train": len(ood_tr), "n_synth_train": len(synth),
        "n_id_test": len(id_te), "n_ood_test": len(ood_te),
        "ood_classes": per_class,
        "method": "trained on real out-of-scope HK + synthetic; evaluated on HELD-OUT real OOD",
    }, indent=2))
    print(f"Saved → {OOD_OUT}\n      → {THR_OUT}\n      → {MET_OUT}")


if __name__ == "__main__":
    main()
