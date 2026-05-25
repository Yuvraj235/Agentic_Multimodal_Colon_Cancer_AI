"""ColonAI — retrain the pathology head from accumulated user feedback.

The Streamlit results page lets clinicians click "AI was right / wrong"
on each prediction. Their answers (plus the model's internal feature
embedding for that case — NEVER the raw image or patient identifiers)
are stored in outputs/learning_log/. This script reads them back,
fine-tunes ONLY the pathology head (cheap and fast), and saves a new
checkpoint.

The encoder / fusion / staging / risk heads are NOT touched — so the
hard work (mask-aware vision features, BioBERT text understanding,
TabTransformer tabular fusion) is preserved.

Privacy contract — see src/app/learning_log.py for details. tl;dr:
   • input = (256-d fused embedding, label)
   • we never see the raw image, the patient name, or the symptoms text

Run:
    python3 scripts/retrain_from_feedback.py
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.learning_log import load_training_set, stats
from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES, CLASS_NAMES_5
from src.app.security import safe_torch_load

CKPT_IN  = Path("outputs/unified_multimodal_v2/checkpoints/best_model.pth")
CKPT_OUT = Path("outputs/unified_multimodal_v2/checkpoints/best_model_feedback.pth")


def main():
    # ── 1. Load the user-feedback dataset ───────────────────────────────
    print("Loading user-feedback training set …")
    print(f"  Status: {stats()}")
    data = load_training_set()
    X = data["X"]                          # (N, 256)
    y_labels = data["y"]                   # list[str]
    w = data["weights"]                    # (N,)
    n = X.shape[0]
    if n < 10:
        print(f"\n  ✗ only {n} labelled cases — need ≥ 10 to retrain.")
        print(f"  Encourage more feedback through the UI, then re-run.")
        return
    print(f"\n  ✓ {n} labelled cases  ·  weights mean = {w.mean():.2f}")
    # Map string labels → class indices
    cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES_5)}
    y = np.array([cls_to_idx.get(lbl, -1) for lbl in y_labels])
    valid = y >= 0
    X, y, w = X[valid], y[valid], w[valid]
    print(f"  Per-class breakdown:")
    for c in CLASS_NAMES_5:
        c_idx = cls_to_idx[c]
        print(f"    {c:20s} {(y == c_idx).sum():3d}")

    # ── 2. Build the model + warm-start ─────────────────────────────────
    device = (torch.device("cuda") if torch.cuda.is_available()
              else (torch.device("mps") if torch.backends.mps.is_available()
                    else torch.device("cpu")))
    print(f"\nDevice: {device}")
    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    if not CKPT_IN.exists():
        print(f"  ✗ checkpoint missing: {CKPT_IN}")
        return
    state = safe_torch_load(str(CKPT_IN), map_location=device, allow_unsafe=True)
    state = state.get("model_state", state)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded {CKPT_IN.name}")

    # ── 3. Freeze everything except the pathology head ──────────────────
    for p in model.parameters(): p.requires_grad = False
    head = model.head.pathology
    for p in head.parameters(): p.requires_grad = True
    print(f"  Trainable params: "
          f"{sum(p.numel() for p in head.parameters()):,} (head only)")

    # ── 4. Fine-tune ────────────────────────────────────────────────────
    X_t = torch.from_numpy(X).to(device).float()
    y_t = torch.from_numpy(y).to(device).long()
    w_t = torch.from_numpy(w).to(device).float()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-3)
    print(f"\nFine-tuning the pathology head (50 epochs, AdamW lr=1e-4)…")
    model.train()
    for ep in range(50):
        opt.zero_grad()
        logits = head(X_t)
        loss   = F.cross_entropy(logits, y_t, reduction="none")
        loss   = (loss * w_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y_t).float().mean().item()
                print(f"  ep {ep+1:2d}/50  loss={loss.item():.4f}  acc={acc:.3f}")

    # ── 5. Save ─────────────────────────────────────────────────────────
    out = {"model_state": model.state_dict(),
           "from_user_feedback_n": int(n),
           "classes": CLASS_NAMES_5,
           "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "warm_start_from": str(CKPT_IN)}
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, CKPT_OUT)
    print(f"\n✓ Saved feedback-tuned checkpoint to {CKPT_OUT}")
    print(f"\nNext steps:")
    print(f"  1. Run scripts/validate_gradcam_v2.py to verify no regression")
    print(f"  2. If happy, replace best_model.pth with this file:")
    print(f"     mv {CKPT_OUT} outputs/unified_multimodal_v2/checkpoints/best_model.pth")
    print(f"  3. Push to HF Space + restart")


if __name__ == "__main__":
    main()
