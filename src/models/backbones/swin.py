import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class SwinBackbone(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.model = swin_t(
            weights=Swin_T_Weights.IMAGENET1K_V1
        )

        self.model.head = nn.Identity()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)
        self.model.eval()

        self.output_dim = 768  # Swin-T feature dim

    def forward(self, x):
        return self.model(x)