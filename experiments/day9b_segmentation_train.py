import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.data.cvc_seg_dataset import CVCClinicSegDataset
from src.models.multitask_convnext import MultiTaskConvNeXt
from src.losses.dice_loss import DiceLoss

# =============================
# CONFIG
# =============================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

DATA_PATH = "data/raw/CVC-ClinicDB/PNG"
SAVE_PATH = "outputs/day9/convnext_segmentation.pth"
os.makedirs("outputs/day9", exist_ok=True)

# =============================
# DATA
# =============================
dataset = CVCClinicSegDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Total samples:", len(dataset))

# =============================
# MODEL
# =============================
model = MultiTaskConvNeXt(num_classes=4).to(DEVICE)

# Freeze classifier head
for param in model.classifier.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

dice_loss = DiceLoss()

# =============================
# TRAIN LOOP
# =============================
print("\n🚀 Starting Segmentation Training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(loader)

    for imgs, masks in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        _, seg_preds = model(imgs)

        loss = dice_loss(seg_preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# =============================
# SAVE MODEL
# =============================
torch.save(model.state_dict(), SAVE_PATH)
print("\n✅ Segmentation training complete!")
print("Model saved to:", SAVE_PATH)