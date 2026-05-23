"""ColonAI — Out-of-Distribution (OOD) head.

A lightweight classifier trained on top of the unified model's fused
embeddings. Given a fused embedding, it outputs P(in-distribution).

Training data
─────────────
   In-distribution (label 1):
     * 1,000 HyperKvasir train images (random sample)
     * 600  Polyp-with-mask images (vendor-diverse)

   Out-of-distribution (label 0):
     * 800 synthetic noise images (multiple textures + colours that look
       superficially endoscopy-ish but aren't real tissue)
     * 800 inverted/heavily-augmented HK images
     * 400 random RGB checkerboards + gradients

Output
──────
   outputs/unified_multimodal_v2/ood_head.pth         — sklearn-style MLP
   outputs/unified_multimodal_v2/ood_threshold.json   — best F1 threshold
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
from typing import List
import numpy as np, torch, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import (N_TABULAR_FEATURES,
    HyperKvasirMultiModalDataset, CLASS_NAMES_5)
from transformers import AutoTokenizer

CKPT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
OOD_OUT  = Path("outputs/unified_multimodal_v2/ood_head.pth")
THR_OUT  = Path("outputs/unified_multimodal_v2/ood_threshold.json")
EMB_DIM  = 256

random.seed(42); np.random.seed(42); torch.manual_seed(42)


def synthetic_ood(n_per_kind: int = 200) -> List[np.ndarray]:
    """Generate fake OOD images that look plausibly endoscopy-ish."""
    out = []
    rng = np.random.default_rng(42)
    # 1. Pure Gaussian noise
    for _ in range(n_per_kind):
        out.append(np.clip(rng.normal(127, 60, (224, 224, 3)), 0, 255).astype(np.uint8))
    # 2. Coloured gradients
    for _ in range(n_per_kind):
        h = rng.integers(0, 360)
        s = rng.uniform(0.3, 0.9); v = rng.uniform(0.4, 0.9)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            for x in range(224):
                img[y, x] = [int(127 + 80*np.sin(x/30 + y/40 + h/10)),
                             int(100 + 70*np.cos(x/25 - y/35 + h/12)),
                             int(140 + 60*np.sin((x+y)/45 + h/8))]
        out.append(img)
    # 3. Random checkerboards
    for _ in range(n_per_kind):
        c = rng.integers(8, 32)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, c):
            for j in range(0, 224, c):
                col = rng.integers(0, 255, 3)
                img[i:i+c, j:j+c] = col
        out.append(img)
    return out


def get_fused_embedding(model, x, ids, msk, tab, device):
    with torch.no_grad():
        out = model(x.to(device), ids, msk, tab)
        fused = out.get("fused")
        if fused is None: return None
        if fused.dim() == 3:
            fused = fused.mean(dim=1)
        return fused.cpu().numpy().squeeze()


class OODHead(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"Device: {device}")

    print("Loading v2 model …")
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tok("Patient referred for screening colonoscopy.",
              padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    ids = enc["input_ids"].to(device); msk = enc["attention_mask"].to(device)
    tab = torch.zeros((1, N_TABULAR_FEATURES), device=device)

    # ── 1. Collect in-distribution embeddings ──────────────────────────
    print("\nCollecting in-distribution embeddings …")
    tcga = pd.read_csv("data/raw/tcga/clinical/clinical.tsv", sep="\t") \
           if Path("data/raw/tcga/clinical/clinical.tsv").exists() else None
    hk_ds = HyperKvasirMultiModalDataset(
        root_dir="data/processed/hyper_kvasir_clean",
        tokenizer=tok, tcga_df=tcga, split="train",
        val_ratio=0.15, test_ratio=0.10, seed=42)
    # Random subset of 1000 HK
    idxs = random.sample(range(len(hk_ds)), min(1000, len(hk_ds)))
    in_emb = []
    for i, idx in enumerate(idxs):
        x = hk_ds[idx]["image"].unsqueeze(0).to(device)
        emb = get_fused_embedding(model, x, ids, msk, tab, device)
        if emb is not None: in_emb.append(emb)
        if (i+1) % 100 == 0: print(f"  HK {i+1}/{len(idxs)}")

    # Polyp-with-mask (vendor-diverse, sampled)
    polyp_dirs = [
        "data/raw/kvasir-seg/Kvasir-SEG/images",
        "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images",
        "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/images",
    ]
    tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    n_pp = 0
    for d in polyp_dirs:
        p = Path(d)
        if not p.exists(): continue
        files = sorted(p.iterdir())[:200]
        for f in files:
            if f.suffix.lower() not in (".jpg",".jpeg",".png",".tif",".tiff",".bmp"):
                continue
            try:
                pim = Image.open(f).convert("RGB")
                x = tfm(pim).unsqueeze(0).to(device)
                emb = get_fused_embedding(model, x, ids, msk, tab, device)
                if emb is not None: in_emb.append(emb); n_pp += 1
            except Exception: pass
        print(f"  {d}: +{n_pp}")
    print(f"  Total in-dist: {len(in_emb)}")

    # ── 2. Collect OOD embeddings ───────────────────────────────────────
    print("\nGenerating synthetic OOD images …")
    ood_imgs = synthetic_ood(n_per_kind=200)
    print(f"  {len(ood_imgs)} OOD images")
    ood_emb = []
    for i, arr in enumerate(ood_imgs):
        try:
            x = tfm(Image.fromarray(arr)).unsqueeze(0).to(device)
            emb = get_fused_embedding(model, x, ids, msk, tab, device)
            if emb is not None: ood_emb.append(emb)
        except Exception: pass
        if (i+1) % 100 == 0: print(f"  OOD {i+1}/{len(ood_imgs)}")

    # ── 3. Train the head ──────────────────────────────────────────────
    print(f"\nIn-dist: {len(in_emb)}   OOD: {len(ood_emb)}")
    X = np.vstack([np.asarray(in_emb), np.asarray(ood_emb)]).astype(np.float32)
    y = np.concatenate([np.ones(len(in_emb)), np.zeros(len(ood_emb))]).astype(np.float32)
    # Shuffle + 80/20 split
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    n_tr = int(0.8 * len(X))
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    head = OODHead(dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).to(device); ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)

    print("\nTraining OOD head …")
    head.train(); best_va_acc = 0.0
    for ep in range(40):
        idxs = torch.randperm(len(Xtr_t), device=device)
        for s in range(0, len(idxs), 64):
            b = idxs[s:s+64]
            logits = head(Xtr_t[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        # Val
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(Xva_t).squeeze(-1))
            pred = (p > 0.5).float()
            va_acc = (pred == yva_t).float().mean().item()
        head.train()
        if va_acc > best_va_acc:
            best_va_acc = va_acc
            torch.save({"state_dict": head.state_dict(),
                        "emb_dim":    X.shape[1]}, OOD_OUT)
        if (ep+1) % 10 == 0:
            print(f"  ep {ep+1:2d}  loss={loss.item():.4f}  val_acc={va_acc:.4f}")

    # ── 4. Find best F1 threshold on val ────────────────────────────────
    head.load_state_dict(torch.load(OOD_OUT)["state_dict"])
    head.eval()
    with torch.no_grad():
        scores = torch.sigmoid(head(Xva_t).squeeze(-1)).cpu().numpy()
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (scores > t).astype(float)
        tp = ((pred == 1) & (yva == 1)).sum()
        fp = ((pred == 1) & (yva == 0)).sum()
        fn = ((pred == 0) & (yva == 1)).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        if f1 > best_f1: best_f1, best_t = float(f1), float(t)
    print(f"\nBest F1 threshold: {best_t:.3f}   F1 = {best_f1:.4f}")
    THR_OUT.write_text(json.dumps({
        "threshold":   best_t, "val_f1": best_f1,
        "val_acc":     best_va_acc, "n_in": len(in_emb), "n_ood": len(ood_emb)
    }, indent=2))
    print(f"Saved → {OOD_OUT}\nSaved → {THR_OUT}")


if __name__ == "__main__":
    main()
