"""View-quality advisory — warns when the colonoscopy view is too poor to assess.

Head trained on HyperKvasir bowel-prep frames (scripts/train_view_quality.py;
held-out AUROC ~0.985). Score = P(poor view) from the fused embedding (mean-pooled
to match training). is_poor = score >= threshold.

Advisory ONLY — it adds a "view may be unreliable" warning, it does not block a
result. Fail-open if the head is unavailable.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

_HEAD = Path("outputs/unified_multimodal_v2/view_quality_head.pth")
_THR = Path("outputs/unified_multimodal_v2/view_quality_threshold.json")
_cache: dict = {}


def _load():
    if "loaded" in _cache:
        return _cache["head"], _cache["thr"]
    import torch
    import torch.nn as nn
    thr = 0.5
    try:
        thr = float(json.loads(_THR.read_text()).get("threshold", 0.5))
    except Exception:
        pass
    head = None
    try:
        ck = torch.load(_HEAD, map_location="cpu")
        dim = int(ck.get("emb_dim", 256))

        class _H(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(64, 1))

            def forward(self, x):
                return self.net(x)

        head = _H(dim)
        head.load_state_dict(ck["state_dict"])
        head.eval()
    except Exception:
        head = None
    _cache.update(loaded=True, head=head, thr=thr)
    return head, thr


def _vec(e) -> np.ndarray:
    a = np.asarray(e, np.float32)
    if a.ndim == 2:
        a = a.mean(0)
    return a.squeeze()


def view_quality_check(fused_embedding) -> dict:
    """Return {is_poor, p_poor, available[, threshold]}. Fail-open."""
    head, thr = _load()
    if head is None or fused_embedding is None:
        return {"is_poor": False, "p_poor": 0.0, "available": False}
    try:
        import torch
        v = _vec(fused_embedding)
        with torch.no_grad():
            p = float(torch.sigmoid(head(torch.from_numpy(v).float().unsqueeze(0)).squeeze()).item())
        return {"is_poor": bool(p >= thr), "p_poor": round(p, 4),
                "available": True, "threshold": thr}
    except Exception:
        return {"is_poor": False, "p_poor": 0.0, "available": False}
