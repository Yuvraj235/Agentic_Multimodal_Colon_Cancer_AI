import torch
from torchvision import transforms
from torchvision.models import convnext_tiny
import torch.nn as nn

from src.data.dataset import HyperKvasirDataset
from src.agents.image_agent import ImagePerceptionAgent
from src.agents.pathology_agent import PathologyReasoningAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.orchestrator import OrchestratorAgent


# -----------------------------
# Config
# -----------------------------
IMAGE_ROOT = "data/processed/hyper_kvasir_clean"
LABELS_CSV = "data/processed/clean_labels.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# Dataset
# -----------------------------
dataset = HyperKvasirDataset(
    image_root=IMAGE_ROOT,
    labels_csv=LABELS_CSV,
    transform=transform
)

sample = dataset[0]
image_tensor = sample["image"].unsqueeze(0)


# -----------------------------
# Backbone
# -----------------------------
backbone = convnext_tiny(weights="IMAGENET1K_V1")
backbone.classifier = nn.Identity()


# -----------------------------
# Agents
# -----------------------------
image_agent = ImagePerceptionAgent(backbone)
pathology_agent = PathologyReasoningAgent()
explanation_agent = ExplanationAgent()

orchestrator = OrchestratorAgent(
    image_agent=image_agent,
    pathology_agent=pathology_agent,
    explanation_agent=explanation_agent
)


# -----------------------------
# Run
# -----------------------------
output = orchestrator.run(image_tensor)

print("\n===== AGENTIC OUTPUT =====")
for k, v in output.items():
    print(f"\n{k.upper()}:")
    print(v)