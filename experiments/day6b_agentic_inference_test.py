import torch
from torchvision import transforms
from torchvision.models import convnext_tiny
import torch.nn as nn

from src.data.dataset import HyperKvasirDataset
from src.agents.image_agent import ImagePerceptionAgent
from src.agents.pathology_agent import PathologyReasoningAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator import OrchestratorAgent
from src.models.pathology_head import PathologyHead


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

IMAGE_ROOT = "data/processed/hyper_kvasir_clean"
LABELS_CSV = "data/processed/clean_labels.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# DATASET
# --------------------------------------------------
dataset = HyperKvasirDataset(
    image_root=IMAGE_ROOT,
    labels_csv=LABELS_CSV,
    transform=transform
)

sample = dataset[0]
image_tensor = sample["image"].unsqueeze(0).to(DEVICE)


# --------------------------------------------------
# BACKBONE (ConvNeXt)
# --------------------------------------------------
backbone = convnext_tiny(weights="IMAGENET1K_V1")
backbone.classifier = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()


# --------------------------------------------------
# PATHOLOGY HEAD
# --------------------------------------------------
head = PathologyHead(num_classes=len(dataset.LABEL_MAP))
head.load_state_dict(
    torch.load("models/pathology_head_hyperkvasir.pth", map_location=DEVICE)
)
head = head.to(DEVICE)
head.eval()


# --------------------------------------------------
# AGENTS
# --------------------------------------------------
image_agent = ImagePerceptionAgent(model=backbone)

pathology_agent = PathologyReasoningAgent(
    head=head,
    label_map=dataset.LABEL_MAP,
    device=DEVICE
)

explanation_agent = ExplanationAgent(
    target_layer_name="features.7",
    device=DEVICE
)

orchestrator = OrchestratorAgent(
    image_agent=image_agent,
    pathology_agent=pathology_agent,
    explanation_agent=explanation_agent,
    device=DEVICE
)


# --------------------------------------------------
# RUN AGENTIC PIPELINE
# --------------------------------------------------
output = orchestrator.run(image_tensor)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
print("\n===== DAY 7B CLINICAL OUTPUT =====\n")

print("PERCEPTION:")
print(output["perception"])

print("\nREASONING:")
for k, v in output["reasoning"].items():
    print(f"{k}: {v}")

print("\nEXPLANATION:")
print(output["explanation"])

print("\nCLINICAL_TEXT:")
for k, v in output["clinical_text"].items():
    print(f"{k}: {v}")