import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import HyperKvasirDataset
from src.models.backbones.convnext import ConvNeXtBackbone
from src.models.backbones.efficientnet import EfficientNetBackbone
from src.models.backbones.swin import SwinBackbone
from src.models.fusion.fusion_model import MultimodalFusionModel

# -------------------------
# CONFIG
# -------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8

CSV_PATH = "data/processed/clean_labels.csv"
IMAGE_ROOT = "data/processed/hyper_kvasir_clean"
MODEL_PATH = "outputs/day8/fusion_model.pt"
OUTPUT_DIR = "outputs/day8/attention"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using device:", DEVICE)

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# DATA
# -------------------------
dataset = HyperKvasirDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------------
# MODEL
# -------------------------
model = MultimodalFusionModel(
    ConvNeXtBackbone(device=DEVICE),
    EfficientNetBackbone(device=DEVICE),
    SwinBackbone(device=DEVICE),
    num_classes=4
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# COLLECT ATTENTION
# -------------------------
all_attn = []
all_labels = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(DEVICE)
        _, attn = model(x)

        all_attn.append(attn.cpu().numpy())
        all_labels.append(y.numpy())

all_attn = np.concatenate(all_attn)
all_labels = np.concatenate(all_labels)

# -------------------------
# SAVE CSV
# -------------------------
df = pd.DataFrame(
    all_attn,
    columns=["ConvNeXt", "EfficientNet", "Swin"]
)
df["label"] = all_labels
df.to_csv(f"{OUTPUT_DIR}/attention_weights.csv", index=False)

print("✅ Attention weights saved")

# -------------------------
# PLOT MEAN ATTENTION
# -------------------------
mean_attn = df.groupby("label")[["ConvNeXt", "EfficientNet", "Swin"]].mean()

mean_attn.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Mean Attention Weight")
plt.title("Mean Modality Attention per Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mean_attention_per_class.png")
plt.close()

print("📊 Mean attention plot saved")