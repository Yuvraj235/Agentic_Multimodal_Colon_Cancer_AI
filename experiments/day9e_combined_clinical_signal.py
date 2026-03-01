"""
===============================================================
DAY 9E — CROSS-DOMAIN COMBINED CLINICAL SIGNAL VISUALIZATION
===============================================================
Uses multitask ConvNeXt (classification + segmentation)
Builds risk map = segmentation × classification confidence
Paper-grade layout.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn.functional import softmax, sigmoid
from tqdm import tqdm

from src.models.multitask_convnext import MultiTaskConvNeXt
from src.data.cvc_seg_dataset import CVCClinicSegDataset
from src.data.dataset import HyperKvasirDataset

# =========================
# CONFIG
# =========================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MULTITASK_MODEL_PATH = "outputs/day9/multitask_convnext.pth"

CVC_PATH = "data/raw/CVC-ClinicDB/PNG"
HK_CSV = "data/processed/clean_labels.csv"
HK_IMG = "data/processed/hyper_kvasir_clean"

OUT_DIR = "outputs/day9/combined_signal"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SAMPLES = 6
IMG_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Polyps",
    "Ulcerative Colitis",
    "Esophagitis"
]

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# =========================
# LOAD DATASETS
# =========================
cvc_dataset = CVCClinicSegDataset(CVC_PATH)
hk_dataset = HyperKvasirDataset(HK_CSV, HK_IMG, transform=transform)

print("CVC samples:", len(cvc_dataset))
print("HyperKvasir samples:", len(hk_dataset))

# =========================
# LOAD MODEL
# =========================
model = MultiTaskConvNeXt(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MULTITASK_MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Multitask model loaded")

# =========================
# METRICS
# =========================
def dice_score(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

# =========================
# VISUALIZATION FUNCTION
# =========================
def generate_panel(image, seg_prob, risk_map,
                   logits, domain, mask=None, save_path=None):

    probs = softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

    seg_np = seg_prob.squeeze().cpu().numpy()
    risk_np = risk_map.squeeze().cpu().numpy()

    image_np = image.permute(1,2,0).cpu().numpy()
    image_np = np.clip(image_np,0,1)

    # Normalize risk map
    risk_np = risk_np - risk_np.min()
    if risk_np.max() > 0:
        risk_np = risk_np / risk_np.max()

    # Heatmaps
    heat_seg = cv2.applyColorMap(np.uint8(255*seg_np), cv2.COLORMAP_JET)
    heat_seg = cv2.cvtColor(heat_seg, cv2.COLOR_BGR2RGB)

    heat_risk = cv2.applyColorMap(np.uint8(255*risk_np), cv2.COLORMAP_JET)
    heat_risk = cv2.cvtColor(heat_risk, cv2.COLOR_BGR2RGB)

    overlay_risk = cv2.addWeighted(
        (image_np*255).astype(np.uint8),
        0.6,
        heat_risk,
        0.4,
        0
    )

    fig = plt.figure(figsize=(18,10))

    # ---------------- ROW 1 ----------------
    plt.subplot(2,4,1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(heat_seg)
    plt.title("Segmentation Probability")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(heat_risk)
    plt.title("Combined Clinical Risk")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(overlay_risk)
    plt.title("Risk Overlay")
    plt.axis("off")

    # ---------------- ROW 2 ----------------
    plt.subplot(2,4,5)
    plt.text(0.1,0.5,f"Domain: {domain}",fontsize=12)
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.text(0.1,0.5,f"Prediction:\n{CLASS_NAMES[pred_class]}",fontsize=12)
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.text(0.1,0.5,f"Confidence:\n{confidence*100:.2f}%",fontsize=12)
    plt.axis("off")

    plt.subplot(2,4,8)
    if mask is not None:
        d = dice_score(seg_prob, mask.to(DEVICE))
        i = iou_score(seg_prob, mask.to(DEVICE))
        plt.text(0.1,0.6,f"Dice: {d:.3f}",fontsize=12)
        plt.text(0.1,0.4,f"IoU: {i:.3f}",fontsize=12)
    else:
        plt.text(0.1,0.5,"No GT Mask",fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close()

# =========================
# RUN FOR BOTH DOMAINS
# =========================
print("\nGenerating panels...\n")

# ---- CVC ----
for idx in range(NUM_SAMPLES):
    img, mask = cvc_dataset[idx]
    x = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, seg = model(x)

    seg_prob = sigmoid(seg)
    risk_map = seg_prob * softmax(logits,dim=1).max()

    save_path = os.path.join(OUT_DIR,f"CVC_{idx}.png")
    generate_panel(
        img,
        seg_prob,
        risk_map,
        logits,
        domain="CVC-ClinicDB",
        mask=mask,
        save_path=save_path
    )

# ---- Hyper-Kvasir ----
for idx in range(NUM_SAMPLES):
    img, label = hk_dataset[idx]
    x = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, seg = model(x)

    seg_prob = sigmoid(seg)
    risk_map = seg_prob * softmax(logits,dim=1).max()

    save_path = os.path.join(OUT_DIR,f"HK_{idx}.png")
    generate_panel(
        img,
        seg_prob,
        risk_map,
        logits,
        domain="Hyper-Kvasir",
        mask=None,
        save_path=save_path
    )

print("\n✅ Day 9E Combined Clinical Signal Complete!")
print("Saved to:", OUT_DIR)