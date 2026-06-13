# Real-time polyp detector — the free (Kaggle) video phase

Goal: a YOLO detector that flags polyps **live, frame-by-frame** during colonoscopy —
the most clinically proven form of colonoscopy AI (it helps doctors not miss polyps).
Built end-to-end on **free** compute: training on Kaggle's free GPU, the live demo on
your Mac.

## Why this approach
We don't download a giant video dataset to start. We already have **~4,900 polyp
images with segmentation masks** locally (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB,
CVC-300, ETIS, BKAI, PolypGen). A mask → a bounding box is an exact conversion, and a
detector trained on these **runs on live video frame-by-frame** — which is exactly how
real-time polyp detection works at inference. True video datasets
(LDPolypVideo / SUN-SEG) are a later robustness add-on, not a blocker.

## Step 1 — Build the YOLO dataset (local, free, ~1 min)
```
python3 scripts/video/masks_to_yolo.py
```
Produces `outputs/yolo_polyp/` with `images/{train,val,test}`, `labels/...`, `data.yaml`.
Split is **leak-free by source**: ETIS-Larib (Pentax) is the held-out **test** (honest
cross-vendor number), CVC-300 is **val**, everything else is **train** (~4.5k images).

## Step 2 — Put the dataset on Kaggle (free)
1. Zip it: `cd outputs && zip -r yolo_polyp.zip yolo_polyp` (≈1–2 GB).
2. kaggle.com → **Datasets → New Dataset** → upload the zip. (Free hosting; data sits
   next to the GPU so no re-download each session.)

## Step 3 — Train on Kaggle's free GPU
1. **New Notebook** → Add your dataset (right panel) → **Settings → Accelerator → GPU T4 x2**
   (one-time phone verification unlocks it). Free quota: **30 GPU-hrs/week**.
2. In a cell:
   ```python
   !pip -q install ultralytics
   !python /kaggle/input/<your-dataset>/scripts/video/train_polyp_yolo.py \
        --data /kaggle/input/<your-dataset>/yolo_polyp/data.yaml
   ```
   (or upload `train_polyp_yolo.py` and point `--data` at the dataset's `data.yaml`.)
3. Use **Save Version → Save & Run All (Commit)** so training runs in the **background**
   (survives closing the tab, up to ~9–12 hr).
4. If a session ends mid-run, re-run with `--resume` to continue from the last checkpoint.

Defaults: `yolo11s` pretrained, 640 px, 100 epochs with early-stop. On a free T4 this is
roughly 2–3 hr — well inside one session and the weekly quota. Swap `--model yolov8n.pt`
for a smaller/faster model, or `--epochs` to shorten.

## Step 4 — Honest evaluation (automatic)
After training it evaluates on the **held-out ETIS (Pentax)** test split and prints
`mAP50`, `mAP50-95`, precision, and **recall (sensitivity)** — recall is the one that
matters clinically (don't miss polyps). This is the real cross-vendor number, not a
same-data score.

## Step 5 — Bring it back to the Mac
Training exports **`best.onnx`** and **`best.mlpackage` (CoreML)**. Download them.
The CoreML model runs real-time on your M3 Pro for the live-video mode (next step:
wiring it into the app's "Live Video Mode" — `src/app/video_pipeline.py`).

## Step 6 — Phase 3: run it live on the Mac
Once training is done, **download `best.pt`** from the Kaggle Output panel and drop it at:
```
outputs/unified_multimodal_v2/polyp_detector.pt
```
That single file activates the detector everywhere — both paths auto-detect it (and
fall back to the old GradCAM method if it's missing, so nothing breaks before then):

- **In the app — "Live Video Mode" (Step 6):** upload a colonoscopy clip → it runs the
  detector frame-by-frame, draws polyp boxes, tracks each polyp across frames, and
  returns an annotated video + a per-polyp summary. (`src/app/video_pipeline.py`.)
- **Standalone real-time (webcam / capture card):**
  ```
  python3 scripts/video/live_detect.py                   # webcam
  python3 scripts/video/live_detect.py --source clip.mp4 # a recorded clip
  ```
  Draws live boxes with an FPS counter — the true real-time demo on Apple Silicon.

Both apply the endoscopy gate first (so a non-colon frame is ignored) and require a
polyp to persist across a few frames before it's "confirmed" (kills single-frame
false positives).

## Free-tier playbook
- **Kaggle** = primary (30 GPU-hrs/week, background runs). **Colab free** = overflow.
  **Lightning's 15 credits** = save for an occasional A100 boost. **Mac** = runs the demo.
- Always `--resume`-friendly; always check the **ETIS** number, not the train/val score.
- Nothing here costs money.
