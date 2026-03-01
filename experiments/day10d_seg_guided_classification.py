"""
=========================================================
DAY 10D – SEGMENTATION GUIDED CLASSIFICATION (UPGRADED)
ConvNeXt-Small Backbone
Strong Augmentation + Scheduler + Label Smoothing
=========================================================
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm

from src.models.convnext_unet import ConvNeXtUNet
from src.data.dataset import HyperKvasirDataset


# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 16
LR = 3e-4

CSV_PATH = "data/processed/clean_labels.csv"
IMAGE_ROOT = "data/processed/hyper_kvasir_clean"

SEG_MODEL_PATH = "outputs/day10/convnext_unet_strong.pth"
SAVE_PATH = "outputs/day10/seg_guided_classifier_best.pth"

print("Using device:", DEVICE)


# ==========================================================
# TRANSFORMS (VERY IMPORTANT)
# ==========================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# DATASET
# ==========================================================

dataset = HyperKvasirDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
    transform=train_transform
)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total samples: {len(dataset)}")


# ==========================================================
# LOAD SEGMENTATION MODEL (FROZEN)
# ==========================================================

seg_model = ConvNeXtUNet(pretrained=False).to(DEVICE)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
seg_model.eval()

for param in seg_model.parameters():
    param.requires_grad = False

print("✅ Segmentation model loaded")


# ==========================================================
# CLASSIFICATION MODEL (UPGRADED)
# ==========================================================

model = timm.create_model(
    "convnext_small",
    pretrained=True,
    num_classes=4
).to(DEVICE)


# Freeze early layers (stability)
for name, param in model.named_parameters():
    if "stages.0" in name or "stages.1" in name:
        param.requires_grad = False


# ==========================================================
# LOSS + OPTIMIZER
# ==========================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Segmentation mask
        with torch.no_grad():
            masks = seg_model(images)

        images = images * masks

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total


    # ---------------- VALIDATION ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            masks = seg_model(images)
            images = images * masks

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    scheduler.step()

    print("\n===================================")
    print(f"Epoch {epoch+1}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print("===================================\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("✅ Best classification model saved.")


print("\n🔥 Training Complete")
print("Best Val Accuracy:", best_val_acc)