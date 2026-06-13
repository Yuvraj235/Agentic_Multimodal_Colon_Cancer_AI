"""Convert our local polyp segmentation datasets → YOLO detection format.

A real-time polyp DETECTOR (YOLO) needs bounding boxes, but our local data is
segmentation masks. A mask → box is an exact, lossless conversion: threshold the
mask, find each connected blob, take its bounding rectangle. A detector trained on
these images runs frame-by-frame on live video — exactly how clinical real-time
polyp detection works at inference.

Uses ONLY data we already have (zero download, zero cost). Leak-free split BY
SOURCE so the honest test number means something:
  test : ETIS-Larib (Pentax — a scanner brand held out everywhere else too)
  val  : CVC-300     (distinct source, for tuning)
  train: Kvasir-SEG + CVC-ClinicDB + CVC-ColonDB + Kvasir-test + BKAI + PolypGen C1-C6

Output: outputs/yolo_polyp/{images,labels}/{train,val,test}/ + data.yaml
Run: python3 scripts/video/masks_to_yolo.py
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/yolo_polyp"
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
MIN_AREA_FRAC = 0.0008          # ignore blobs smaller than 0.08% of the image (noise)

# (source_name, images_dir, masks_dir, split)
DATASETS = [
    ("kvasirseg",  "data/raw/kvasir-seg/Kvasir-SEG/images",                       "data/raw/kvasir-seg/Kvasir-SEG/masks",                       "train"),
    ("clinicdb",   "data/raw/CVC-ClinicDB/PNG/Original",                          "data/raw/CVC-ClinicDB/PNG/Ground Truth",                     "train"),
    ("colondb",    "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/images", "data/raw/test_polyp_datasets/TestDataset/CVC-ColonDB/masks", "train"),
    ("kvasirtest", "data/raw/test_polyp_datasets/TestDataset/Kvasir/images",      "data/raw/test_polyp_datasets/TestDataset/Kvasir/masks",      "train"),
    ("bkai",       "data/raw/bkai/images",                                        "data/raw/bkai/masks",                                        "train"),
    ("cvc300",     "data/raw/test_polyp_datasets/TestDataset/CVC-300/images",     "data/raw/test_polyp_datasets/TestDataset/CVC-300/masks",     "val"),
    ("etis",       "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/images",
                   "data/raw/test_polyp_datasets/TestDataset/ETIS-LaribPolypDB/masks", "test"),
] + [(f"polypgen_c{c}", f"data/raw/polypgen/data_C{c}/images_C{c}",
      f"data/raw/polypgen/data_C{c}/masks_C{c}", "train") for c in range(1, 7)]


def _find_mask(stem: str, mdir: Path):
    for e in EXTS:
        for cand in (mdir / (stem + e), mdir / (stem + "_mask" + e)):
            if cand.exists():
                return cand
    return None


def _boxes_from_mask(mask_path: Path):
    """Return list of YOLO boxes [(cx,cy,w,h) normalised] from a polyp mask.

    Binarise as 'any non-black pixel' (max over RGB channels), NOT luminance —
    BKAI masks are colour-coded (red=neoplastic, green=non-neoplastic) and a
    luminance threshold silently drops the red ones (red L~76 < 127). For a
    detector every polyp counts regardless of type, so any coloured blob = polyp.
    """
    m = np.asarray(Image.open(mask_path).convert("RGB"))
    H, W = m.shape[:2]
    binm = (m.max(axis=2) > 64).astype(np.uint8)
    if binm.sum() == 0:
        return []
    min_area = max(20, int(MIN_AREA_FRAC * H * W))
    boxes = []
    try:
        import cv2
        n, _, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
        comps = [(stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]) for i in range(1, n)]
    except Exception:
        from scipy import ndimage
        lab, n = ndimage.label(binm)
        comps = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            comps.append((xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(xs)))
    for x, y, w, h, area in comps:
        if area < min_area:
            continue
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        boxes.append((cx, cy, w / W, h / H))
    return boxes


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    boxcount = {"train": 0, "val": 0, "test": 0}
    empties = 0
    for name, idir, mdir, split in DATASETS:
        ip, mp = ROOT / idir, ROOT / mdir
        if not ip.exists() or not mp.exists():
            print(f"  · {name}: MISSING ({idir})"); continue
        n = 0
        for f in sorted(ip.iterdir()):
            if f.suffix.lower() not in EXTS or "_mask" in f.stem:
                continue
            mk = _find_mask(f.stem, mp)
            if mk is None:
                continue
            boxes = _boxes_from_mask(mk)
            if not boxes:
                empties += 1
                continue
            uid = f"{name}_{f.stem}{f.suffix.lower()}"
            shutil.copy(f, OUT / "images" / split / uid)
            lbl = OUT / "labels" / split / (f"{name}_{f.stem}.txt")
            lbl.write_text("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes))
            counts[split] += 1; boxcount[split] += len(boxes); n += 1
        print(f"  · {name:14} → {split:5} : {n} images")

    yaml = OUT / "data.yaml"
    yaml.write_text(
        "# Polyp detector — built from local seg masks (masks_to_yolo.py)\n"
        f"path: {OUT}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: [polyp]\n")
    print("\n=== YOLO dataset ready ===")
    for s in ("train", "val", "test"):
        print(f"  {s:5}: {counts[s]:5} images · {boxcount[s]:5} boxes")
    print(f"  (skipped {empties} mask-empty images)")
    print(f"  data.yaml → {yaml}")
    print(f"  output    → {OUT}")


if __name__ == "__main__":
    main()
