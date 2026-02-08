import torch
from src.utils.gradcam import GradCAM


class ExplanationAgent:
    """
    Agent responsible for visual explanations (Grad-CAM).
    """

    def __init__(self, target_layer_name=None, device="cpu"):
        self.target_layer_name = target_layer_name
        self.device = device

    def _resolve_target_layer(self, model):
        """
        Resolve layer string like 'features.7' to actual nn.Module
        """
        layer = model
        for attr in self.target_layer_name.split("."):
            layer = getattr(layer, attr)
        return layer

    def explain(self, image_tensor, model, class_idx):
        if self.target_layer_name is None:
            return {"method": "None", "overlay_path": None}

        model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        target_layer = self._resolve_target_layer(model)

        cam_generator = GradCAM(
            model=model,
            target_layer=target_layer
        )

        cam = cam_generator.generate(
            input_tensor=image_tensor,
            class_idx=class_idx
        )

        output_path = cam_generator.save_overlay(
            cam,
            image_tensor,
            filename="outputs/gradcam/gradcam_sample.png"
        )

        return {
            "method": "Grad-CAM",
            "overlay_path": output_path
        }