import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from src.data.dataset import HyperKvasirDataset
from src.models.backbones.convnext import ConvNeXtBackbone
from src.models.backbones.efficientnet import EfficientNetBackbone
from src.models.backbones.swin import SwinBackbone
from src.models.fusion.fusion_model import MultimodalFusionModel


# =========================
# CONFIG (MINIMAL CHANGES)
# =========================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 3                 # 🔧 reduced from 5
LR = 3e-5                  # 🔧 reduced from 1e-4

CSV_PATH = "data/processed/clean_labels.csv"
IMAGE_ROOT = "data/processed/hyper_kvasir_clean"
OUTPUT_DIR = "outputs/day8"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)


# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# DATASET
# =========================
dataset = HyperKvasirDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
    transform=transform
)

print(f"✅ Loaded {len(dataset)} samples")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,        # 🔧 prevents BN crash
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================
model = MultimodalFusionModel(
    ConvNeXtBackbone(device=DEVICE),
    EfficientNetBackbone(device=DEVICE),
    SwinBackbone(device=DEVICE),
    num_classes=4
).to(DEVICE)


# =========================
# PRINT TRAINABLE PARAMS (ONCE)
# =========================
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"🧠 Trainable parameters: {trainable:,}")
print(f"📦 Total parameters:     {total:,}")
print(f"🔒 Frozen parameters:    {total - trainable:,}")


# =========================
# OPTIMIZER / LOSS
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# =========================
# TRAINING
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")


# =========================
# EVALUATION
# =========================
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
preds = np.argmax(all_probs, axis=1)


# =========================
# ROC–AUC
# =========================
plt.figure()
for i in range(4):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    auc = roc_auc_score(all_labels == i, all_probs[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curves (Multimodal Fusion)")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_auc.png")
plt.close()


# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()


# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(all_labels, preds)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), f"{OUTPUT_DIR}/fusion_model.pt")

print("\n✅ Day 8B.1 COMPLETE (stabilized run)")
print(f"📁 Artifacts saved to: {OUTPUT_DIR}")