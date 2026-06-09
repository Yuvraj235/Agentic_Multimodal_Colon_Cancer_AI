"""Polyp characterization (CADx) specialist — "what kind of polyp?".

Answers the optical-diagnosis question a clinician asks once a polyp is seen:
does it look NEOPLASTIC (adenomatous/serrated — resect & send for histology) or
NON-NEOPLASTIC (e.g. hyperplastic — usually benign)? This is decision support that
mirrors the resect-and-discard / leave-in-situ judgement; it never replaces histology.

Trained by scripts/train_characterization.py on BKAI-IGH NeoPolyp (image-level
neoplastic/non-neoplastic labels derived from the segmentation masks; the lesion is
cropped from the mask bounding box). Fail-open: returns available=False if the head
isn't present (e.g. a fresh clone without weights).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_HEAD = Path("outputs/unified_multimodal_v2/cadx_head.pth")
_cache: dict = {}


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


def _bbox_from_mask(mask, margin=0.08):
    """(x0,y0,x1,y1) bounding box of a binary/greyscale polyp mask, or None."""
    try:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m.max(axis=2)
        ys, xs = np.where(m > 8)
        if ys.size == 0:
            return None
        H, W = m.shape[:2]
        my, mx = int(margin * H), int(margin * W)
        return (max(0, int(xs.min()) - mx), max(0, int(ys.min()) - my),
                min(W, int(xs.max()) + mx), min(H, int(ys.max()) + my))
    except Exception:
        return None


def characterize(pil_image, mask=None, bbox=None) -> dict:
    """Classify a polyp as neoplastic vs non-neoplastic.

    pil_image : the colonoscopy frame (PIL).
    mask      : optional segmentation mask (PIL/np) — used to crop to the lesion.
    bbox      : optional explicit (x0,y0,x1,y1) crop, overrides mask.

    Returns {available, kind, kind_full, p_neoplastic, confidence, probs}. Fail-open.
    """
    net, classes = _load()
    if net is None or pil_image is None or not classes:
        return {"available": False}
    try:
        import torch
        import torchvision.transforms as T
        img = pil_image.convert("RGB")
        crop = bbox or (_bbox_from_mask(mask) if mask is not None else None)
        if crop:
            img = img.crop(crop)
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(net(x), dim=-1).squeeze().numpy()
        top = int(p.argmax())
        kind = classes[top]
        p_neo = float(p[classes.index("neoplastic")]) if "neoplastic" in classes else 0.0
        full = {"neoplastic": "looks neoplastic (adenoma/serrated — resect & send for histology)",
                "non-neoplastic": "looks non-neoplastic (e.g. hyperplastic — usually benign)"}
        return {"available": True, "kind": kind,
                "kind_full": full.get(kind, kind),
                "p_neoplastic": round(p_neo, 4),
                "confidence": round(float(p[top]), 4),
                "probs": {c: round(float(p[i]), 4) for i, c in enumerate(classes)},
                "cropped": bool(crop)}
    except Exception:
        return {"available": False}


if __name__ == "__main__":
    from PIL import Image
    img = Image.open("data/raw/bkai/images/" + __import__("os").listdir("data/raw/bkai/images")[0])
    print(characterize(img))
