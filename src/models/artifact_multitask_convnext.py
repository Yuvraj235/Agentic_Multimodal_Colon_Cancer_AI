import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ArtifactAwareConvNeXt(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        # Classification head
        self.classifier = nn.Linear(in_features, num_classes)

        # Lesion segmentation head
        self.lesion_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )

        # Artifact segmentation head
        self.artifact_head = nn.Sequential(
            nn.Conv2d(768, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        features = self.backbone.features(x)

        pooled = features.mean(dim=[2,3])
        logits = self.classifier(pooled)

        lesion_map = F.interpolate(
            self.lesion_head(features),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        )

        artifact_map = F.interpolate(
            self.artifact_head(features),
            size=(224,224),
            mode="bilinear",
            align_corners=False
        )

        return logits, lesion_map, artifact_map