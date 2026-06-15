# CARE rectal-cancer CT segmentation — Colab + your 5 TB Drive (free)

Builds a CT rectal-**tumor** segmenter from the **CARE** dataset (398 patients,
pixel masks, CC BY-NC 4.0 — research/non-commercial). ~74 GB, so it stays on Drive
and trains on Colab (Colab mounts Drive; Kaggle can't).

> Honest scope: this is **CT tumor segmentation**, not MRI local-staging. It's the
> only free + labelled rectal-imaging dataset we could actually download. Worth it
> only if a CT tumor-segmenter is useful to you.

## Step 1 — get CARE onto your Drive (no Mac download)
1. Open the CARE Google-Drive link from the U-SAM repo: https://github.com/kanydao/U-SAM
   (it's CC BY-NC 4.0 — agree to the non-commercial research terms).
2. **Add it to your own Drive** (Drive → right-click → "Add shortcut / Make a copy to My Drive"),
   so it lands at e.g. `MyDrive/CARE`. Nothing touches your Mac.

## Step 2 — Colab
New Colab notebook → **Runtime → Change runtime type → GPU (T4)**. Then cells:
```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install segmentation-models-pytorch albumentations
# upload train_care_ct_colab.py to the notebook (or !wget it from your GitHub), then:
# FIRST — inspect the layout (I can't see your Drive):
!python train_care_ct_colab.py --root "/content/drive/MyDrive/CARE" --inspect-only
```
The inspect step prints the folder tree, file types, **mask label values**, and how
many image/mask pairs it auto-detected.

## Step 3 — adjust if needed, then train
CARE ships as **`.npz` files** (`train/train_npz/*.npz`, `test/test_npz/*.npz`; each
npz holds `image` + `label`). The script already knows this format and uses CARE's
**own train/test split** as the honest held-out.
- The inspect step prints the label values (expect `[0, 1, 2]`). The tumour is
  **label 2** (1 ≈ normal rectum) and is the **pre-set default** (`TUMOR_LABELS = {2}`).
  Only change it if the printout shows the tumour is a different value.
- If `test_npz files: 0`, the data unzipped somewhere else — fix `--root` to the folder
  that contains `train/` and `test/`.
- Then train (saves to Drive, resumable):
```python
!python train_care_ct_colab.py --root "/content/drive/MyDrive/CARE" \
        --out "/content/drive/MyDrive/CARE/care_ct_seg.pt"
```
Defaults: U-Net + **resnet34** trainable encoder, 384 px, 40 epochs, Dice+BCE,
**patient-level** holdout split, honest IoU + 95% CI. Colab free sessions time out
(~12 h / idle disconnects) — re-run with `--resume` to continue from the last save.

## Step 4 — bring it back
Download `care_ct_seg.pt` + `care_ct_seg_metrics.json` from Drive. Tell me the
held-out IoU; if it's solid we add it as a **CT tumor-segmentation specialist**
(new modality path — it won't touch the colonoscopy pipeline). If it's weak, we
report it honestly and don't ship it.
