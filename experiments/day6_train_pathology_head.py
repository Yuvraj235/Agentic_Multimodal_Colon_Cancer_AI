# ============================================
# Day 6A — Train Pathology Head (HyperKvasir)
# ============================================

import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny

# --------------------------------------------
# Project root & paths (CRITICAL FIX)
# --------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_ROOT = PROCESSED_DIR / "hyper_kvasir_clean"
LABELS_CSV = PROCESSED_DIR / "clean_labels.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# --------------------------------------------
# Imports (after path resolution)
# --------------------------------------------
from src.data.dataset import HyperKvasirDataset
from src.models.pathology_head import PathologyHead

# --------------------------------------------
# Device
# --------------------------------------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------
# Config
# --------------------------------------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

# --------------------------------------------
# Transforms
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------------------------
# Dataset & Loader
# --------------------------------------------
dataset = HyperKvasirDataset(
    image_root=str(IMAGE_ROOT),
    labels_csv=str(LABELS_CSV),
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,        # IMPORTANT for macOS / Python 3.13
    pin_memory=False
)

# --------------------------------------------
# Backbone (Frozen ConvNeXt)
# --------------------------------------------
backbone = convnext_tiny(weights="IMAGENET1K_V1")
backbone.classifier = nn.Identity()

for param in backbone.parameters():
    param.requires_grad = False

backbone.to(DEVICE)
backbone.eval()

# --------------------------------------------
# Pathology Head
# --------------------------------------------
NUM_CLASSES = len(dataset.LABEL_MAP)

head = PathologyHead(
    in_dim=768,          # ConvNeXt-Tiny feature dim
    num_classes=NUM_CLASSES
)

head.to(DEVICE)
head.train()

# --------------------------------------------
# Optimizer & Loss
# --------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(head.parameters(), lr=LR)

# --------------------------------------------
# Training Loop
# --------------------------------------------
for epoch in range(EPOCHS):
    running_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.no_grad():
            features = backbone(images)      # (B, 768, 1, 1)
            features = features.flatten(1)   # (B, 768)

        logits = head(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# --------------------------------------------
# Save Model (SAFE)
# --------------------------------------------
save_path = MODELS_DIR / "pathology_head_hyperkvasir.pth"
torch.save(head.state_dict(), save_path)

print(f"\n✅ Pathology head saved to: {save_path}")
print("🎉 Day 6A training completed successfully.")