# src/models/fusion/attention_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multimodal embeddings.

    Input:
        List of tensors [B, D] from different modalities

    Output:
        Fused tensor [B, D]
        Attention weights [B, num_modalities]
    """

    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Learnable attention scorer
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, features: list):
        """
        features: List of tensors, each shape [B, D]
        """

        if len(features) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(features)}"
            )

        # Stack -> [B, M, D]
        x = torch.stack(features, dim=1)

        # Compute attention scores per modality
        # [B, M, 1]
        scores = self.attention_fc(x)

        # Normalize across modalities
        # [B, M]
        weights = F.softmax(scores.squeeze(-1), dim=1)

        # Weighted sum
        # [B, D]
        fused = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return fused, weights