# experiments/day9c_multitask_train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.multitask_convnext import MultiTaskConvNeXt
from src.data.cvc_seg_dataset import CVCClinicSegDataset

# =============================
# CONFIG
# =============================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_SEG = 0.5  # Weight for segmentation loss

SAVE_PATH = "outputs/day9/multitask_convnext.pth"
os.makedirs("outputs/day9", exist_ok=True)

# =============================
# DATASET
# =============================
dataset = CVCClinicSegDataset("data/raw/CVC-ClinicDB/PNG")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Total samples:", len(dataset))

# =============================
# MODEL
# =============================
model = MultiTaskConvNeXt(num_classes=4).to(DEVICE)

# =============================
# LOSSES
# =============================
ce_loss = nn.CrossEntropyLoss()

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# =============================
# OPTIMIZER
# =============================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =============================
# TRAIN LOOP
# =============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    loop = tqdm(loader)
    
    for imgs, masks in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        logits, seg_out = model(imgs)
        
        # Dummy classification labels (optional)
        # If CVC dataset has class labels, replace here
        cls_labels = torch.zeros(imgs.size(0), dtype=torch.long).to(DEVICE)
        
        loss_cls = ce_loss(logits, cls_labels)
        loss_seg = dice_loss(seg_out, masks)
        
        loss = loss_cls + LAMBDA_SEG * loss_seg
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(loader):.4f}")

# =============================
# SAVE MODEL
# =============================
torch.save(model.state_dict(), SAVE_PATH)

print("\n✅ Multi-task training complete!")
print("Model saved to:", SAVE_PATH)