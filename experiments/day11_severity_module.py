"""
DAY 11 – Lesion Severity Scoring Module
Segmentation-Based Clinical Severity Estimation
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
import os

from src.models.convnext_unet import ConvNeXtUNet

# ================= CONFIG =================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLS_MODEL_PATH = "outputs/day10/seg_guided_classifier_best.pth"
SEG_MODEL_PATH = "outputs/day10/convnext_unet_strong.pth"

IMAGE_PATH = None  # leave None to auto-pick

CLASS_NAMES = [
    "Anatomical Landmarks",
    "Pathological Findings",
    "Quality Views",
    "Therapeutic"
]

# ================= LOAD MODELS =================

classifier = timm.create_model(
    "convnext_small",
    pretrained=False,
    num_classes=4
)

classifier.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
classifier.to(DEVICE).eval()

seg_model = ConvNeXtUNet(pretrained=False)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
seg_model.to(DEVICE).eval()

print("✅ Models Loaded")

# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ================= AUTO PICK IMAGE =================

if IMAGE_PATH is None:
    import glob
    images = glob.glob("data/processed/hyper_kvasir_clean/**/*.jpg", recursive=True)
    IMAGE_PATH = images[np.random.randint(len(images))]

print("Using image:", IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ================= CLASSIFICATION =================

with torch.no_grad():
    output = classifier(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

predicted_class = predicted_class.item()
confidence = confidence.item()

print("Predicted:", CLASS_NAMES[predicted_class])
print("Confidence:", round(confidence,4))

# ================= SEGMENTATION =================

with torch.no_grad():
    raw_mask = seg_model(input_tensor)

mask = torch.sigmoid(raw_mask)
mask = (mask > 0.5).float()
mask = torch.nn.functional.interpolate(mask, size=(224,224))

lesion_area = mask.sum().item()
total_pixels = 224 * 224
area_ratio = lesion_area / total_pixels

# ================= SEVERITY LOGIC =================

severity = "Not Applicable"

if predicted_class == 1:  # Pathological Findings

    if area_ratio < 0.05:
        severity = "Mild"
    elif area_ratio < 0.15:
        severity = "Moderate"
    else:
        severity = "Severe"

print("\n==============================")
print("Lesion Area Ratio:", round(area_ratio*100,2), "%")
print("Severity:", severity)
print("==============================")

# ================= SAVE REPORT =================

os.makedirs("outputs/day11/severity", exist_ok=True)

report_path = "outputs/day11/severity/severity_report.txt"

with open(report_path, "w") as f:
    f.write("Predicted Class: " + CLASS_NAMES[predicted_class] + "\n")
    f.write("Confidence: " + str(round(confidence,4)) + "\n")
    f.write("Lesion Area (%): " + str(round(area_ratio*100,2)) + "\n")
    f.write("Severity: " + severity + "\n")

print("✅ Severity report saved")