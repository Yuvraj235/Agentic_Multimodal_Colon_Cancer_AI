"""Histology tissue-classifier specialist (NCT-CRC-HE, 9 classes).

Classifies an H&E biopsy/resection TILE into one of 9 tissue types; the clinically
key output is P(tumour epithelium). This is the microscopy branch that feeds the
staging/grading direction — separate from the colonoscopy pipeline.

Trained by scripts/train_histology.py (demonstrator on CRC-VAL-HE-7K). Fail-open:
returns available=False if the head isn't present (e.g. fresh clone without weights).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_HEAD = Path("outputs/unified_multimodal_v2/histology_head.pth")
_cache: dict = {}

CLASS_FULL = {
    "ADI": "adipose (fat)", "BACK": "background", "DEB": "debris",
    "LYM": "lymphocytes (immune cells)", "MUC": "mucus", "MUS": "smooth muscle",
    "NORM": "normal colon mucosa", "STR": "cancer-associated stroma",
    "TUM": "tumour epithelium (adenocarcinoma)",
}


def _load():
    if "loaded" in _cache:
        return _cache["net"], _cache["classes"]
    net, classes = None, []
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        ck = torch.load(_HEAD, map_location="cpu")
        classes = list(ck.get("classes", []))
        net = models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(classes))
        net.load_state_dict(ck["state_dict"])
        net.eval()
    except Exception:
        net = None
    _cache.update(loaded=True, net=net, classes=classes)
    return net, classes


def classify_tile(pil_image) -> dict:
    """Return {available, tissue, tissue_full, p_tumour, probs}. Fail-open."""
    net, classes = _load()
    if net is None or pil_image is None or not classes:
        return {"available": False}
    try:
        import torch
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(pil_image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(net(x), dim=-1).squeeze().numpy()
        top = int(p.argmax())
        tum = float(p[classes.index("TUM")]) if "TUM" in classes else 0.0
        return {"available": True, "tissue": classes[top],
                "tissue_full": CLASS_FULL.get(classes[top], classes[top]),
                "p_tumour": round(tum, 4),
                "probs": {c: round(float(p[i]), 4) for i, c in enumerate(classes)}}
    except Exception:
        return {"available": False}
