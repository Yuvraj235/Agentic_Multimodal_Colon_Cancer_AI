import torch
import torch.nn.functional as F
import numpy as np


class GradCAMPlusPlus:
    """
    Clinical-grade Grad-CAM++ implementation.
    Compatible with ConvNeXt / ResNet / CNN backbones.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx, confidence=None):
        """
        Generate Grad-CAM++ heatmap.

        Args:
            image_tensor (Tensor): shape (1, C, H, W)
            class_idx (int): target class index
            confidence (float | None): optional clinical confidence scaling

        Returns:
            np.ndarray: normalized CAM (Hc, Wc)
        """

        self.model.zero_grad()
        logits = self.model(image_tensor)

        # Backward target score
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients           # (B, C, H, W)
        acts = self.activations          # (B, C, H, W)

        # Grad-CAM++ math
        grads_pow2 = grads ** 2
        grads_pow3 = grads ** 3

        eps = 1e-8
        denom = 2 * grads_pow2 + torch.sum(
            acts * grads_pow3, dim=(2, 3), keepdim=True
        )
        denom = torch.where(denom != 0.0, denom, torch.ones_like(denom) * eps)

        alpha = grads_pow2 / denom
        positive_grads = torch.relu(grads)

        weights = torch.sum(alpha * positive_grads, dim=(2, 3))
        cam = torch.sum(weights[:, :, None, None] * acts, dim=1)
        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        # Normalize
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        # Optional clinical confidence scaling
        if confidence is not None:
            cam *= float(confidence)

        return cam


# --------------------------------------------------
# Clinical Postprocessing
# --------------------------------------------------
def postprocess_heatmap(cam, target_size=(224, 224), threshold=0.3):
    """
    Convert raw CAM to clinical-grade heatmap.

    - Upsample to image resolution
    - Suppress noise
    - Normalize again

    Args:
        cam (np.ndarray): (Hc, Wc)
        target_size (tuple): output resolution
        threshold (float): activation cutoff

    Returns:
        np.ndarray: (H, W)
    """

    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(
        cam_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    cam = cam_tensor.squeeze().numpy()

    cam[cam < threshold] = 0.0

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam