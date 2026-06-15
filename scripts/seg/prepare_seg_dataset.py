"""Package our local polyp image+mask pairs into a compact segmentation dataset.

For the dedicated cross-vendor segmentation fine-tune (trainable encoder). Reuses
the SAME sources + leak-free split as the detector (scripts/video/masks_to_yolo.py)
but keeps the pixel MASK instead of deriving a box:

  test : ETIS-Larib (Pentax)  — held out everywhere (the honest cross-vendor number)
  val  : CVC-300               — distinct source, for tuning
  train: Kvasir-SEG + CVC-ClinicDB + CVC-ColonDB + Kvasir-test + BKAI + PolypGen C1-6

Resizes image (bilinear) + mask (nearest) to 352x352 and binarises the mask as
"any non-black pixel" (so BKAI's red/green colour masks are kept — same fix as the
detector prep). Output: outputs/seg_polyp/{images,masks}/{train,val,test}/.
Run: python3 scripts/seg/prepare_seg_dataset.py
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/seg_polyp"
SIZE = 352
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

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


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "masks" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    empties = 0
    for name, idir, mdir, split in DATASETS:
        ip, mp = ROOT / idir, ROOT / mdir
        if not ip.exists() or not mp.exists():
            print(f"  · {name}: MISSING"); continue
        n = 0
        for f in sorted(ip.iterdir()):
            if f.suffix.lower() not in EXTS or "_mask" in f.stem:
                continue
            mk = _find_mask(f.stem, mp)
            if mk is None:
                continue
            try:
                img = Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
                m = np.asarray(Image.open(mk).convert("RGB").resize((SIZE, SIZE), Image.NEAREST))
                binm = (m.max(axis=2) > 64).astype(np.uint8) * 255   # any non-black = polyp
            except Exception:
                continue
            if binm.sum() == 0:
                empties += 1
                continue
            uid = f"{name}_{f.stem}"
            img.save(OUT / "images" / split / f"{uid}.jpg", quality=92)
            Image.fromarray(binm).save(OUT / "masks" / split / f"{uid}.png")
            counts[split] += 1; n += 1
        print(f"  · {name:14} → {split:5} : {n}")

    print("\n=== seg dataset ready ===")
    for s in ("train", "val", "test"):
        print(f"  {s:5}: {counts[s]} image/mask pairs")
    print(f"  (skipped {empties} mask-empty)")
    print(f"  output → {OUT}  (zip + upload to Kaggle as the seg input)")


if __name__ == "__main__":
    main()
