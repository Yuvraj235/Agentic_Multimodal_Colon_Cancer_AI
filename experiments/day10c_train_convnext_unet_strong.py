"""
=========================================================
DAY 10C – STRONG POLYP SEGMENTATION TRAINING (FIXED)
ConvNeXt Encoder + UNet Decoder
Balanced BCE + Dice Loss
=========================================================
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.convnext_unet import ConvNeXtUNet
from src.data.cvc_seg_dataset import CVCClinicSegDataset

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
DATA_PATH = "data/raw/CVC-ClinicDB/PNG"
SAVE_PATH = "outputs/day10/convnext_unet_strong.pth"

print("Using device:", DEVICE)

# ==========================================================
# DATA
# ==========================================================

dataset = CVCClinicSegDataset(DATA_PATH, img_size=IMG_SIZE)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total: {len(dataset)} | Train: {train_size} | Val: {val_size}")

# ==========================================================
# MODEL
# ==========================================================

model = ConvNeXtUNet(pretrained=True).to(DEVICE)

# ==========================================================
# LOSS FUNCTIONS
# ==========================================================

# Foreground ≈ 14% → weight positive pixels higher
POS_WEIGHT = torch.tensor([5.0]).to(DEVICE)

bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target):
    bce = bce_loss(pred, target)
    d = dice_loss(pred, target)
    return 0.6 * bce + 0.4 * d


# ==========================================================
# METRICS
# ==========================================================

def compute_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + 1e-6) / (union + 1e-6)

    total = pred.numel()
    iou = (intersection + 1e-6) / (pred.sum() + target.sum() - intersection + 1e-6)

    return dice.item(), iou.item()


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

best_val_loss = float("inf")

# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = combined_loss(outputs, masks)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0
    val_dice = 0
    val_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            val_loss += loss.item()

            dice, iou = compute_metrics(outputs, masks)
            val_dice += dice
            val_iou += iou

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    scheduler.step(val_loss)

    print("\n===================================")
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Dice:   {val_dice:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")
    print("===================================\n")

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("outputs/day10", exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best model saved.")

print("\n🔥 Training Complete")
print("Best Model Saved At:", SAVE_PATH)