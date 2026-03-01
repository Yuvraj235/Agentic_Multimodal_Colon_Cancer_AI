import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtBackbone(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.model = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # Remove classifier
        self.model.classifier = nn.Identity()

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.model.to(device)
        self.pool.to(device)
        self.model.eval()

        self.output_dim = 768

    def forward(self, x):
        x = self.model.features(x)     # [B, 768, H, W]
        x = self.pool(x)               # [B, 768, 1, 1]
        x = torch.flatten(x, 1)        # ✅ [B, 768]
        return x