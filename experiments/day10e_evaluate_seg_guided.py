"""
DAY 10E – COMPLETE EVALUATION SUITE
Segmentation Guided vs Raw
Includes:
• Confusion Matrix
• ROC Curves
• Calibration Curve
• ECE
• McNemar Test
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar
import timm
from torchvision import transforms
import torch.nn.functional as F

from src.models.convnext_unet import ConvNeXtUNet
from src.data.dataset import HyperKvasirDataset


# ==================================================
# CONFIG
# ==================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16

CSV_PATH = "data/processed/clean_labels.csv"
IMAGE_ROOT = "data/processed/hyper_kvasir_clean"

CLS_MODEL_PATH = "outputs/day10/seg_guided_classifier_best.pth"
SEG_MODEL_PATH = "outputs/day10/convnext_unet_strong.pth"

SAVE_DIR = "outputs/day10/evaluation"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = [
    "Anatomical Landmarks",
    "Pathological Findings",
    "Quality Views",
    "Therapeutic"
]

print("Using device:", DEVICE)


# ==================================================
# TRANSFORM
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# ==================================================
# DATA
# ==================================================

dataset = HyperKvasirDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
    transform=transform
)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Validation samples:", len(val_dataset))


# ==================================================
# LOAD MODELS
# ==================================================

model = timm.create_model("convnext_small", pretrained=False, num_classes=4)
model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

seg_model = ConvNeXtUNet(pretrained=False)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
seg_model.to(DEVICE).eval()

print("✅ Models Loaded")


# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate(use_mask=True):

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if use_mask:
                masks = seg_model(images)
                images = images * masks

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ==================================================
# RUN BOTH
# ==================================================

print("\nEvaluating WITHOUT mask...")
y_true_raw, y_pred_raw, y_prob_raw = evaluate(use_mask=False)

print("\nEvaluating WITH mask...")
y_true_mask, y_pred_mask, y_prob_mask = evaluate(use_mask=True)


# ==================================================
# CONFUSION MATRIX
# ==================================================

def plot_confusion(y_true, y_pred, name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix_{name}.png", dpi=300)
    plt.close()

plot_confusion(y_true_raw, y_pred_raw, "unmasked")
plot_confusion(y_true_mask, y_pred_mask, "masked")


# ==================================================
# ROC CURVES
# ==================================================

y_bin = label_binarize(y_true_mask, classes=[0,1,2,3])

plt.figure(figsize=(8,6))

for i in range(4):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob_mask[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve (Masked)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/roc_curve_masked.png", dpi=300)
plt.close()


# ==================================================
# CALIBRATION CURVE + ECE
# ==================================================

confidences = np.max(y_prob_mask, axis=1)
predictions = y_pred_mask
accuracies = predictions == y_true_mask

prob_true, prob_pred = calibration_curve(accuracies, confidences, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'--')
plt.title("Calibration Curve")
plt.xlabel("Mean Confidence")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/calibration_curve.png", dpi=300)
plt.close()

ece = np.abs(prob_true - prob_pred).mean()


# ==================================================
# MCNEMAR TEST
# ==================================================

correct_raw = y_pred_raw == y_true_raw
correct_mask = y_pred_mask == y_true_mask

table = [
    [np.sum(correct_raw & correct_mask),
     np.sum(correct_raw & ~correct_mask)],
    [np.sum(~correct_raw & correct_mask),
     np.sum(~correct_raw & ~correct_mask)]
]

result = mcnemar(table, exact=True)


# ==================================================
# SAVE SUMMARY
# ==================================================

with open(f"{SAVE_DIR}/metrics_summary.txt","w") as f:

    f.write("=== Accuracy Comparison ===\n")
    f.write(f"Without Mask: {np.mean(correct_raw):.4f}\n")
    f.write(f"With Mask: {np.mean(correct_mask):.4f}\n\n")

    f.write("=== ECE ===\n")
    f.write(f"{ece:.4f}\n\n")

    f.write("=== McNemar Test ===\n")
    f.write(f"Statistic: {result.statistic}\n")
    f.write(f"P-value: {result.pvalue}\n")

print("\n✅ FULL Evaluation Complete")