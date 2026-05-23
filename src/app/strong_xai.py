"""Stronger XAI methods bolted onto the existing GradCAM++ pipeline.

We add Integrated Gradients (Sundararajan 2017) at the input-pixel level — it
is mathematically more rigorous than GradCAM and complements the existing
heatmap.  Both can be shown to the clinician side by side; if they agree on
the salient region the explanation is highly trustworthy.

Method:
   IG(x) = (x - baseline) * ∫₀¹ ∂F/∂x(baseline + α·(x - baseline)) dα

We use a black-image baseline and approximate the integral with 32 steps.
"""
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F


def integrated_gradients(
    model,
    image_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tabular: torch.Tensor,
    target_class: int,
    n_steps: int = 32,
    baseline: str = "black",
) -> np.ndarray:
    """Compute integrated-gradients attribution for the IMAGE branch only.

    Returns a (224, 224) numpy array of attribution scores, normalised to
    [0, 1], that can be overlaid as a heatmap.
    """
    device = image_tensor.device
    model.eval()

    # Baseline: black image (all zeros after ImageNet normalisation, this is
    # roughly the mean ImageNet value but with much less information)
    if baseline == "black":
        bl = torch.zeros_like(image_tensor)
    elif baseline == "gray":
        bl = torch.zeros_like(image_tensor)  # zero tensor in normalised space
    else:
        bl = torch.zeros_like(image_tensor)

    # Path from baseline to input
    alphas = torch.linspace(0.0, 1.0, n_steps, device=device)
    path = torch.stack([bl + a * (image_tensor - bl) for a in alphas], dim=0).squeeze(1)
    # path shape: (n_steps, 3, 224, 224)
    path.requires_grad_(True)

    # Forward + grad in chunks to stay in memory
    chunk = 8
    total_grad = torch.zeros_like(image_tensor)
    for i in range(0, n_steps, chunk):
        chunk_imgs = path[i:i + chunk]
        bsz = chunk_imgs.shape[0]
        ids_b  = input_ids.expand(bsz, -1)
        mask_b = attention_mask.expand(bsz, -1)
        tab_b  = tabular.expand(bsz, -1)

        out = model(image=chunk_imgs, input_ids=ids_b,
                    attention_mask=mask_b, tabular=tab_b)
        logits = out["pathology"]
        scores = logits[:, target_class].sum()
        grads = torch.autograd.grad(scores, chunk_imgs, retain_graph=False)[0]
        total_grad = total_grad + grads.sum(dim=0, keepdim=True) / n_steps

    # IG attribution = (input - baseline) * mean grad
    ig = (image_tensor - bl) * total_grad
    # Collapse channels: take absolute value, sum across RGB
    ig_map = ig.detach().abs().sum(dim=1).squeeze(0).cpu().numpy()

    # Normalise to [0, 1]
    if ig_map.max() > 1e-8:
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
    return ig_map


def overlay_ig(rgb_image: np.ndarray, ig_map: np.ndarray,
               alpha: float = 0.45) -> np.ndarray:
    """Overlay the IG attribution map on the original image as a hot colourmap."""
    import cv2
    H, W = rgb_image.shape[:2]
    if ig_map.shape != (H, W):
        ig_map = cv2.resize(ig_map, (W, H), interpolation=cv2.INTER_LINEAR)

    heat = (ig_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)        # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    img = rgb_image.astype(np.float32)
    overlay = (1 - alpha) * img + alpha * heat.astype(np.float32)
    return overlay.clip(0, 255).astype(np.uint8)


def gradcam_ig_agreement(gradcam: np.ndarray, ig: np.ndarray) -> float:
    """Compute how much GradCAM and Integrated Gradients agree on the salient
    region.  Returns IoU of their top-25% pixels in [0, 1].

    High agreement = both methods independently identified the same region as
    important.  This is a strong signal that the explanation is genuine, not
    an artefact of one particular method.
    """
    import cv2
    # Resize to common size
    H, W = 224, 224
    g = cv2.resize(gradcam, (W, H)).astype(np.float32)
    i = cv2.resize(ig,      (W, H)).astype(np.float32)
    # Top-25% threshold
    g_thr = np.quantile(g, 0.75)
    i_thr = np.quantile(i, 0.75)
    g_mask = g >= g_thr
    i_mask = i >= i_thr
    inter = float(np.logical_and(g_mask, i_mask).sum())
    union = float(np.logical_or(g_mask, i_mask).sum())
    if union < 1:
        return 0.0
    return float(inter / union)
