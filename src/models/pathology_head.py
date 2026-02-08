import torch
import torch.nn as nn
import torch.nn.functional as F


class PathologyHead(nn.Module):
    """
    Trainable pathology classification head.

    Input:
        features from frozen vision backbone
        Shape: [B, C, H, W] OR [B, C]

    Output:
        logits over pathology classes
    """

    def __init__(self, in_dim=768, num_classes=4, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x can be:
        - [B, C, H, W] → pooled
        - [B, C]       → direct
        """

        # 🔥 CRITICAL FIX
        if x.dim() == 4:
            # Global Average Pooling
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = x.view(x.size(0), -1)

        elif x.dim() != 2:
            raise ValueError(f"Unexpected feature shape: {x.shape}")

        return self.classifier(x)