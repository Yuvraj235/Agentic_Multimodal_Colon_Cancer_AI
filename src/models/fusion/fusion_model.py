import torch
import torch.nn as nn

from src.models.fusion.attention_fusion import AttentionFusion
from src.models.projection_head import ProjectionHead


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        backbone1,
        backbone2,
        backbone3,
        embed_dim=512,
        num_classes=4
    ):
        super().__init__()

        # -------------------------
        # Backbones
        # -------------------------
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.backbone3 = backbone3

        # -------------------------
        # Projection heads (CRITICAL)
        # -------------------------
        self.proj1 = ProjectionHead(backbone1.output_dim, embed_dim)
        self.proj2 = ProjectionHead(backbone2.output_dim, embed_dim)
        self.proj3 = ProjectionHead(backbone3.output_dim, embed_dim)

        # -------------------------
        # Attention fusion
        # -------------------------
        self.fusion = AttentionFusion(
            embed_dim=embed_dim,
            num_modalities=3
        )

        # -------------------------
        # Classifier
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        # Backbone features
        f1 = self.backbone1(x)   # [B, D1]
        f2 = self.backbone2(x)   # [B, D2]
        f3 = self.backbone3(x)   # [B, D3]

        # Projection to common space
        f1 = self.proj1(f1)      # [B, 512]
        f2 = self.proj2(f2)      # [B, 512]
        f3 = self.proj3(f3)      # [B, 512]

        # Attention fusion
        fused, attn_weights = self.fusion([f1, f2, f3])

        # Classification
        logits = self.classifier(fused)

        return logits, attn_weights