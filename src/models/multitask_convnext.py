import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones.convnext import ConvNeXtBackbone


class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_classes=4, img_size=224):
        super().__init__()

        # Load backbone
        self.backbone = ConvNeXtBackbone(device="cpu")

        feature_dim = self.backbone.output_dim

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.img_size = img_size

    def forward(self, x):

        # ====================================
        # IMPORTANT: DO NOT use forward_features
        # ====================================

        # Get spatial feature map
        feat_map = self.backbone.model.features(x)

        # Global pooling for classification
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        logits = self.classifier(pooled)

        # Segmentation branch
        seg_out = self.seg_head(feat_map)

        seg_out = F.interpolate(
            seg_out,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False
        )

        return logits, seg_out