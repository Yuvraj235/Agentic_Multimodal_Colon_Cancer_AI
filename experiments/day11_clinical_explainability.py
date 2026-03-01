"""
DAY 11 – CLINICAL GRADE EXPLAINABILITY
GradCAM++ + Integrated Gradients + Uncertainty + Severity
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import timm

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from captum.attr import IntegratedGradients

from src.models.convnext_unet import ConvNeXtUNet


# ================= CONFIG =================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLS_MODEL_PATH = "outputs/day10/seg_guided_classifier_best.pth"
SEG_MODEL_PATH = "outputs/day10/convnext_unet_strong.pth"

OUTPUT_DIR = "outputs/day11"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Anatomical Landmarks",
    "Pathological Findings",
    "Quality Views",
    "Therapeutic"
]


# ================= LOAD MODELS =================
print("Loading models...")

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


# ================= LOAD RANDOM IMAGE =================
import glob
image_paths = glob.glob("data/processed/hyper_kvasir_clean/**/*.jpg", recursive=True)
IMAGE_PATH = np.random.choice(image_paths)

print("Using image:", IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

rgb_np = np.array(image.resize((224,224))) / 255.0


# ================= SEGMENTATION MASK =================
with torch.no_grad():
    mask = seg_model(input_tensor)

mask = (mask > 0.5).float()  # thresholding

masked_input = input_tensor * mask


# ================= CLASSIFICATION =================
with torch.no_grad():
    outputs = classifier(masked_input)
    probs = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

print("Predicted:", CLASS_NAMES[predicted_class])
print("Confidence:", round(confidence,4))


# ================= GRADCAM++ =================
target_layers = [classifier.stages[-1].blocks[-1].norm]

cam = GradCAMPlusPlus(model=classifier, target_layers=target_layers)

targets = [ClassifierOutputTarget(predicted_class)]

grayscale_cam = cam(input_tensor=masked_input, targets=targets)[0]
cam_overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

plt.imsave(f"{OUTPUT_DIR}/gradcam_pp.png", cam_overlay)
print("✅ GradCAM++ saved")


# ================= INTEGRATED GRADIENTS =================
print("Computing Integrated Gradients...")

classifier_cpu = classifier.to("cpu")
masked_input_cpu = masked_input.detach().cpu().float()

ig = IntegratedGradients(classifier_cpu)

attributions, delta = ig.attribute(
    masked_input_cpu,
    target=predicted_class,
    n_steps=50,
    return_convergence_delta=True
)

attr_np = attributions.squeeze().detach().numpy()
attr_np = np.transpose(attr_np, (1,2,0))
attr_np = np.mean(np.abs(attr_np), axis=2)

attr_np = attr_np / (attr_np.max() + 1e-8)

ig_overlay = show_cam_on_image(rgb_np, attr_np, use_rgb=True)

plt.imsave(f"{OUTPUT_DIR}/integrated_gradients.png", ig_overlay)

classifier = classifier.to(DEVICE)

print("✅ Integrated Gradients saved")


# ================= UNCERTAINTY (MC DROPOUT) =================
print("Computing Uncertainty...")

classifier.train()

mc_preds = []

for _ in range(20):
    out = classifier(masked_input)
    mc_preds.append(F.softmax(out, dim=1).detach().cpu().numpy())

mc_preds = np.array(mc_preds)

mean_probs = mc_preds.mean(axis=0)
uncertainty = mc_preds.std(axis=0)

classifier.eval()

uncertainty_score = uncertainty[0][predicted_class]

print("Uncertainty Score:", round(float(uncertainty_score),4))


# ================= SEVERITY SCORING =================
lesion_ratio = mask.sum().item() / (224*224)

if predicted_class == 1:  # Pathological
    if lesion_ratio < 0.05:
        severity = "Mild"
    elif lesion_ratio < 0.15:
        severity = "Moderate"
    else:
        severity = "Severe"
else:
    severity = "Not Applicable"

print("\n==============================")
print("Lesion Area Ratio:", round(lesion_ratio*100,2),"%")
print("Severity:", severity)
print("==============================")


# ================= CLINICAL SUMMARY =================
print("\n==============================")
print("Clinical Interpretation")
print("==============================")

if predicted_class == 0:
    print("Normal anatomical region detected.")
elif predicted_class == 1:
    print("Abnormal pathology detected.")
elif predicted_class == 2:
    print("Low image quality.")
elif predicted_class == 3:
    print("Therapeutic intervention scene.")

if confidence < 0.7:
    print("⚠ Moderate confidence – recommend review.")

if uncertainty_score > 0.1:
    print("⚠ High model uncertainty – clinical verification advised.")

print("\n✅ Clinical Explainability Complete")