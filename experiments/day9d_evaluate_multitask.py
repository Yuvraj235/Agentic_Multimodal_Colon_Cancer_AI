import torch
import numpy as np
from tqdm import tqdm

from src.data.cvc_seg_dataset import CVCClinicSegDataset
from src.models.multitask_convnext import MultiTaskConvNeXt

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = "outputs/day9/multitask_convnext.pth"

# =========================
# Load Dataset
# =========================
dataset = CVCClinicSegDataset("data/raw/CVC-ClinicDB/PNG")
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

print("Total samples:", len(dataset))

# =========================
# Load Model
# =========================
model = MultiTaskConvNeXt(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Multitask model loaded")

# =========================
# Metrics
# =========================
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

dice_list = []
iou_list = []

with torch.no_grad():
    for imgs, masks in tqdm(loader):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        _, seg = model(imgs)

        seg = torch.sigmoid(seg)

        for i in range(seg.size(0)):
            d = dice_score(seg[i], masks[i])
            iou = iou_score(seg[i], masks[i])

            dice_list.append(d.item())
            iou_list.append(iou.item())

print("\n==============================")
print("📊 Multitask Segmentation Results")
print("==============================")
print(f"Mean Dice: {np.mean(dice_list):.4f}")
print(f"Mean IoU:  {np.mean(iou_list):.4f}")
print("==============================")