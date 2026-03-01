"""
=========================================================
DAY 9B — SEGMENTATION VISUALIZATION (CVC-ClinicDB)
=========================================================
Visualizes ConvNeXt segmentation performance.
Outputs side-by-side clinical comparison panels.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from src.data.cvc_seg_dataset import CVCClinicSegDataset
from src.models.multitask_convnext import MultiTaskConvNeXt

# =========================================================
# CONFIG
# =========================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DATA_PATH = "data/raw/CVC-ClinicDB/PNG"
MODEL_PATH = "outputs/day9/convnext_segmentation.pth"
OUT_DIR = "outputs/day9/segmentation_visuals"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SAMPLES = 15
IMG_SIZE = 224

# =========================================================
# DATASET
# =========================================================

dataset = CVCClinicSegDataset(DATA_PATH)
print("Total samples:", len(dataset))

# =========================================================
# MODEL
# =========================================================

model = MultiTaskConvNeXt(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Segmentation model loaded")

# =========================================================
# DICE SCORE FUNCTION
# =========================================================

def dice_score(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# =========================================================
# VISUALIZATION LOOP
# =========================================================

for i in range(NUM_SAMPLES):

    image, mask = dataset[i]
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, pred_mask = model(image)

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()

    # Move to CPU
    image_np = image[0].cpu().permute(1,2,0).numpy()
    gt_np = mask.squeeze().numpy()
    pred_np = pred_mask[0].cpu().squeeze().numpy()

    # Dice per image
    d = dice_score(torch.tensor(pred_np), torch.tensor(gt_np)).item()

    # Create overlay
    overlay = image_np.copy()
    overlay[pred_np == 1] = [1, 0, 0]  # red for predicted

    # =====================================================
    # Plot
    # =====================================================

    fig, axs = plt.subplots(1,4, figsize=(18,5))

    axs[0].imshow(image_np)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(gt_np, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_np, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    axs[3].imshow(overlay)
    axs[3].set_title(f"Overlay (Dice={d:.3f})")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"seg_result_{i}.png"), dpi=300)
    plt.close()

    print(f"Saved sample {i} | Dice: {d:.3f}")

print("\n✅ Visualization complete.")