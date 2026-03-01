import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),   # ✅ FIX: LayerNorm instead of BatchNorm
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.projection(x)