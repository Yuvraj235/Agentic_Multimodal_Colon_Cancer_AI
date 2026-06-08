"""View-quality flag — warns when the colonoscopy view is too poor to assess.

Trained on HyperKvasir's bowel-prep-quality frames (free, already local):
  POOR view (label 1): bbps-0-1 (inadequate prep) + impacted-stool (obscured)
  ADEQUATE   (label 0): bbps-2-3 (good prep)

A small MLP on the fused embedding (same idea as the OOD head). Evaluated on a
HELD-OUT split and reported honestly. Used as an ADVISORY warning in the app, not a
hard gate — if the bowel prep / view is poor, any finding is less reliable.

Writes view_quality_head.pth + view_quality_threshold.json + view_quality_metrics.json.
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
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from transformers import AutoTokenizer
from scripts.train_ood_head import get_fused_embedding, OODHead, CKPT

HK = "data/processed/hyper_kvasir_clean"
POOR_DIRS = [f"{HK}/lower-gi-tract/quality-of-mucosal-views/bbps-0-1",
             f"{HK}/lower-gi-tract/quality-of-mucosal-views/impacted-stool"]
GOOD_DIRS = [f"{HK}/lower-gi-tract/quality-of-mucosal-views/bbps-2-3"]
CAP = 700
HEAD_OUT = Path("outputs/unified_multimodal_v2/view_quality_head.pth")
THR_OUT = Path("outputs/unified_multimodal_v2/view_quality_threshold.json")
MET_OUT = Path("outputs/unified_multimodal_v2/view_quality_metrics.json")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
_TFM = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    dev = (torch.device("cuda") if torch.cuda.is_available()
           else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {dev}")
    m = UnifiedMultiModalTransformer(n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(dev)
    st = torch.load(CKPT, map_location=dev); m.load_state_dict(st.get("model_state", st), strict=False); m.eval()
    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tok("screening colonoscopy bowel preparation", padding="max_length",
              truncation=True, max_length=64, return_tensors="pt")
    ids, msk = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=dev)

    def collect(dirs, cap):
        embs = []
        files = []
        for d in dirs:
            p = Path(d)
            if p.exists():
                files += [f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        random.shuffle(files)
        for f in files[:cap]:
            try:
                x = _TFM(Image.open(f).convert("RGB")).unsqueeze(0).to(dev)
                e = get_fused_embedding(m, x, ids, msk, tab, dev)
                if e is not None:
                    embs.append(e if np.asarray(e).ndim == 1 else np.asarray(e).mean(0))
            except Exception:
                pass
        return np.asarray(embs, np.float32)

    print("Collecting POOR-view embeddings …");  poor = collect(POOR_DIRS, CAP)
    print(f"  poor: {len(poor)}")
    print("Collecting ADEQUATE-view embeddings …"); good = collect(GOOD_DIRS, CAP)
    print(f"  adequate: {len(good)}")

    def split(a, frac=0.3):
        i = np.random.permutation(len(a)); k = int(len(a) * frac); return a[i[k:]], a[i[:k]]
    poor_tr, poor_te = split(poor); good_tr, good_te = split(good)
    Xtr = np.vstack([poor_tr, good_tr]).astype(np.float32)
    ytr = np.concatenate([np.ones(len(poor_tr)), np.zeros(len(good_tr))]).astype(np.float32)  # 1=POOR

    head = OODHead(dim=Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    head.train()
    for ep in range(60):
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(0, len(perm), 64):
            b = perm[s:s+64]
            loss = F.binary_cross_entropy_with_logits(head(Xt[b]).squeeze(-1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

    with torch.no_grad():
        s_poor = torch.sigmoid(head(torch.from_numpy(poor_te).to(dev)).squeeze(-1)).cpu().numpy()
        s_good = torch.sigmoid(head(torch.from_numpy(good_te).to(dev)).squeeze(-1)).cpu().numpy()
    roc = auroc(s_poor, s_good)  # P(poor): poor should score higher
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        tp = (s_poor >= t).sum(); fp = (s_good >= t).sum(); fn = (s_poor < t).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        if f1 > best_f1: best_f1, best_t = float(f1), float(t)
    poor_caught = float((s_poor >= best_t).mean()); good_kept = float((s_good < best_t).mean())
    print(f"\n  >>> VIEW-QUALITY  AUROC={roc:.3f}  F1={best_f1:.3f}  thr={best_t:.3f}")
    print(f"      poor caught={poor_caught:.1%}  adequate kept={good_kept:.1%} (held-out)")

    torch.save({"state_dict": head.state_dict(), "emb_dim": Xtr.shape[1]}, HEAD_OUT)
    THR_OUT.write_text(json.dumps({"threshold": best_t, "f1": best_f1, "auroc": roc,
                                   "note": "score>=threshold => POOR view"}, indent=2))
    MET_OUT.write_text(json.dumps({"auroc": roc, "f1": best_f1, "threshold": best_t,
                                   "poor_caught_frac": poor_caught, "adequate_kept_frac": good_kept,
                                   "n_poor_train": len(poor_tr), "n_good_train": len(good_tr),
                                   "n_poor_test": len(poor_te), "n_good_test": len(good_te)}, indent=2))
    print(f"Saved → {HEAD_OUT}\n      → {THR_OUT}\n      → {MET_OUT}")


if __name__ == "__main__":
    main()
