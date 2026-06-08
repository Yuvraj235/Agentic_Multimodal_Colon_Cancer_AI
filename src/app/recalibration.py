"""Per-site temperature recalibration for ColonAI.

The shipped temperature (T=0.45) was fit on the public HyperKvasir val split.
Real-world confidence will not transfer perfectly to a new site (different scope
brands, lighting, demographics). This module lets a hospital upload a small set of
their own *labelled* colonoscopy images and re-fit the temperature so the
confidence numbers are honest for THEIR data.

Pure logic only (no model import) so it stays unit-testable and does not create a
circular import with app.py. The model forward lives in app.py:page_recalibration,
which feeds logits + labels into fit_temperature() here.
"""
from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

CLASS_NAMES = ["polyps", "uc-mild", "uc-moderate-sev", "barretts-esoph", "therapeutic"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SITE_TEMP_PATH = Path("outputs/unified_multimodal_v2/temperature_site.json")


def _softmax_T(x: np.ndarray, T: float) -> np.ndarray:
    z = x / max(T, 1e-3)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (lower = better calibrated)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(conf)
    if N == 0:
        return 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Grid-search the temperature that minimises ECE on the provided logits.

    Returns {temperature, ece_raw, ece_calibrated, accuracy, n_samples}.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or len(logits) == 0:
        raise ValueError("logits must be a non-empty (N, n_classes) array")
    best_T, best_e = 1.0, float("inf")
    for T in np.linspace(0.3, 4.0, 75):
        e = ece(_softmax_T(logits, T), labels)
        if e < best_e:
            best_e, best_T = e, float(T)
    raw = ece(_softmax_T(logits, 1.0), labels)
    acc = float((logits.argmax(1) == labels).mean())
    return {
        "temperature": round(float(best_T), 3),
        "ece_raw": round(float(raw), 4),
        "ece_calibrated": round(float(best_e), 4),
        "accuracy": round(acc, 4),
        "n_samples": int(len(labels)),
    }


def load_labeled_zip(zip_bytes: bytes,
                     max_files: int = 4000,
                     max_img_bytes: int = 20 * 1024 * 1024) -> Tuple[List, dict]:
    """Parse a ZIP whose images live under class-named subfolders.

    Expected layout (folder names must match CLASS_NAMES):
        polyps/img1.jpg
        uc-mild/...
        uc-moderate-sev/...
        barretts-esoph/...
        therapeutic/...

    Returns (samples, summary) where samples = [(PIL.Image, label_idx), ...].
    Safe against path traversal and oversized members. PIL import is local so the
    module has no hard image dependency for unit tests of fit_temperature().
    """
    from PIL import Image  # local import — keeps fit_temperature testable without PIL
    samples: List[Tuple["Image.Image", int]] = []
    per_class = {c: 0 for c in CLASS_NAMES}
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                skipped += 1
                continue
            if Path(name).suffix.lower() not in IMG_EXTS:
                continue
            if info.file_size > max_img_bytes:
                skipped += 1
                continue
            parts = [p.strip().lower() for p in Path(name).parts]
            label = next((i for i, cn in enumerate(CLASS_NAMES) if cn.lower() in parts), None)
            if label is None:
                skipped += 1
                continue
            if len(samples) >= max_files:
                break
            try:
                with zf.open(info) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                samples.append((img, label))
                per_class[CLASS_NAMES[label]] += 1
            except Exception:
                skipped += 1
                continue
    return samples, {"loaded": len(samples), "skipped": skipped, "per_class": per_class}


def save_site_temperature(result: dict, site_name: str = "") -> Path:
    """Persist a per-site temperature so the app can use it instead of the
    public-data default. Returns the path written."""
    SITE_TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["site_name"] = site_name
    payload["source"] = "per_site_recalibration"
    SITE_TEMP_PATH.write_text(json.dumps(payload, indent=2))
    return SITE_TEMP_PATH


def load_site_temperature():
    """Return the active per-site temperature dict, or None if not set."""
    if SITE_TEMP_PATH.exists():
        try:
            return json.loads(SITE_TEMP_PATH.read_text())
        except Exception:
            return None
    return None


if __name__ == "__main__":
    # Self-test of the pure calibration maths (no model / no PIL needed).
    rng = np.random.default_rng(0)
    N = 400
    labels = rng.integers(0, 5, N)
    # Over-confident logits (large magnitude) → high raw ECE, T>1 should help
    logits = np.full((N, 5), -2.0)
    for i, y in enumerate(labels):
        logits[i, y] = 6.0 if rng.random() < 0.8 else -2.0  # 80% correct, over-confident
    r = fit_temperature(logits, labels)
    print("fit_temperature self-test:", r)
    assert r["ece_calibrated"] <= r["ece_raw"] + 1e-9, "calibration must not worsen ECE"
    assert 0.3 <= r["temperature"] <= 4.0
    print("OK — calibrated ECE", r["ece_calibrated"], "<= raw ECE", r["ece_raw"])
