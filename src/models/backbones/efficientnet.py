import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class EfficientNetBackbone(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.model = efficientnet_b3(
            weights=EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        self.model.classifier = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)
        self.model.eval()

        self.output_dim = 1536  # EfficientNet-B3 feature dim

    def forward(self, x):
        return self.model(x)