"""Trained OOD gate — flags inputs that don't match the model's known findings.

Uses the OOD head trained on REAL out-of-scope endoscopy images
(scripts/train_ood_head_real.py; held-out real-OOD AUROC ~0.996). Score is
P(in-distribution) from the fused embedding, MEAN-POOLED over tokens to match
exactly how the head was trained. is_ood = score < threshold.

Fail-open: if the head/threshold can't load, returns not-OOD (the app still has
the endoscopy gate, prototype distance and cross-check as other safety layers).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

_HEAD_PATH = Path("outputs/unified_multimodal_v2/ood_head.pth")
_THR_PATH = Path("outputs/unified_multimodal_v2/ood_threshold.json")
_cache: dict = {}


def _load():
    if "loaded" in _cache:
        return _cache["head"], _cache["thr"]
    import torch
    import torch.nn as nn
    thr = 0.5
    try:
        thr = float(json.loads(_THR_PATH.read_text()).get("threshold", 0.5))
    except Exception:
        pass
    head = None
    try:
        ckpt = torch.load(_HEAD_PATH, map_location="cpu")
        dim = int(ckpt.get("emb_dim", 256))

        class OODHead(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(64, 1))

            def forward(self, x):
                return self.net(x)

        head = OODHead(dim)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
    except Exception:
        head = None
    _cache["loaded"] = True
    _cache["head"] = head
    _cache["thr"] = thr
    return head, thr


def _to_vec(fused_embedding) -> np.ndarray:
    a = np.asarray(fused_embedding, dtype=np.float32)
    if a.ndim == 2:        # (tokens, dim) -> mean-pool over tokens (training form)
        a = a.mean(axis=0)
    return a.squeeze()


def ood_check(fused_embedding) -> dict:
    """Return {is_ood, p_in_dist, available[, threshold]}.

    Fail-open (is_ood=False) if the head is unavailable or anything errors."""
    head, thr = _load()
    if head is None or fused_embedding is None:
        return {"is_ood": False, "p_in_dist": 1.0, "available": False}
    try:
        import torch
        v = _to_vec(fused_embedding)
        with torch.no_grad():
            logit = head(torch.from_numpy(v).float().unsqueeze(0)).squeeze()
            p = float(torch.sigmoid(logit).item())
        return {"is_ood": bool(p < thr), "p_in_dist": round(p, 4),
                "available": True, "threshold": thr}
    except Exception:
        return {"is_ood": False, "p_in_dist": 1.0, "available": False}
