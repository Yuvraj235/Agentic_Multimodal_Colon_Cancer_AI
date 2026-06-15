# Cross-vendor segmentation fine-tune (free, on Kaggle)

Goal: beat the frozen-decoder cross-vendor mask IoU (**~0.45** on held-out ETIS/Pentax)
by training a **dedicated U-Net with a trainable encoder** — decoupled from the main
5-class model, so it cannot regress the classifier. 100% free (Kaggle GPU).

> Honest expectation: unfreezing the encoder is the real lever, so this *should* beat
> 0.45 — but cross-vendor generalisation is hard; a move to ~0.55–0.65 is a realistic
> hope, not a guarantee. And localisation is already strong via the YOLO **detector**
> (0.88 mAP) — this just sharpens pixel masks. Worth trying since it costs only time.

## Step 1 — build + zip the dataset (local, ~1 min)
```
python3 scripts/seg/prepare_seg_dataset.py
cd outputs && zip -rq seg_polyp.zip seg_polyp && cd ..
```
Produces `outputs/seg_polyp/` (4,502 train / 60 val / 196 ETIS test image+mask pairs,
352×352) and `outputs/seg_polyp.zip` to upload. ETIS is held out = the honest number.

## Step 2 — Kaggle (same as the detector)
1. kaggle.com → **New Dataset** → upload `seg_polyp.zip` (title e.g. `seg-polyp`).
2. **New Notebook** → add the dataset → **Settings**: GPU **T4 x2**, **Internet On**.
3. One cell:
   ```python
   !pip -q install segmentation-models-pytorch albumentations
   import glob; data = glob.glob("/kaggle/input/**/seg_polyp", recursive=True)[0]
   !python /kaggle/input/<your-dataset>/scripts/seg/train_seg_kaggle.py --data {data}
   ```
   (or upload `train_seg_kaggle.py` and point `--data` at the dataset's `seg_polyp` folder.)
4. **Save & Run All (Commit)** → runs in the background (~1–2 hr for 40 epochs on a T4).
   Re-run with `--resume` if a session ends mid-run.

Defaults: U-Net + **resnet34** trainable encoder, 352 px, 40 epochs, Dice+BCE. For a
stronger model swap `--encoder timm-efficientnet-b3` (a bit slower).

## Step 3 — honest result (automatic)
Prints the **ETIS (Pentax) held-out** IoU/Dice with bootstrap **95% CIs** + sens@IoU0.5,
next to the old 0.45. Saved to `seg_finetune_metrics.json`.

## Step 4 — bring it back (only if it actually beats 0.45)
Download `seg_finetune.pt` (+ `.onnx`). If the held-out IoU is clearly better than 0.45,
tell me and I'll wire it into `src/app/segmentation.py` as the localisation model (with
a fail-open fallback to the current decoder). If it's *not* better, we keep the current
one and report honestly — no pretending.
