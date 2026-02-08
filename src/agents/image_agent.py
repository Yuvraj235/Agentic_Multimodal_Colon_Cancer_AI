import torch


class ImagePerceptionAgent:
    """
    Visual perception agent.
    Responsible ONLY for extracting visual features.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def perceive(self, image_tensor):
        """
        image_tensor: Tensor [B, C, H, W]

        Returns:
            dict with key 'features'
        """

        features = self.model(image_tensor)

        # ConvNeXt returns [B, 768, 1, 1]
        if features.dim() == 4:
            features = features.flatten(1)

        return {
            "features": features
        }