import torch
import torch.nn as nn
import timm


class ConvNeXtMultiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # Binary head
        self.binary_head = nn.Linear(in_features, 2)

        # 4-class head
        self.multi_head = nn.Linear(in_features, 4)

    def forward(self, x):
        features = self.backbone(x)

        binary_logits = self.binary_head(features)
        multi_logits = self.multi_head(features)

        return binary_logits, multi_logits