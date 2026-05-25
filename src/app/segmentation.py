"""ColonAI — segmentation-decoder inference helper.

The UNet-style decoder was trained by scripts/train_segmentation_head.py and
saved to outputs/unified_multimodal_v2/seg_head.pth — but the live inference
path in app.py was never wired to actually USE it. The cross-check expects
out["seg_mask"] and was getting None every time, so the consistency check
was running with one signal missing.

This module fixes that. Single public function: predict_seg_mask().
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

_DEC_PATH = Path("outputs/unified_multimodal_v2/seg_head.pth")


# Mirror of scripts/train_segmentation_head.SegDecoder so we don't have to
# import from a training script (those import heavy training deps).
class SegDecoder(nn.Module):
    def __init__(self, in_dim=2048, mid=256, drop=0.1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, mid, 4, stride=2, padding=1),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Dropout2d(drop),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(mid, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )
    def forward(self, feats):
        x = self.up1(feats); x = self.up2(x); x = self.up3(x)
        return x  # (B, 1, 224, 224) logits


_decoder: Optional[SegDecoder] = None
_load_attempted = False


def _load_decoder(device) -> Optional[SegDecoder]:
    """Lazy-load the decoder once. Returns None if file is missing."""
    global _decoder, _load_attempted
    if _decoder is not None:
        return _decoder
    if _load_attempted:
        return None  # already tried, file missing
    _load_attempted = True
    if not _DEC_PATH.exists():
        return None
    try:
        from src.app.security import safe_torch_load
        state = safe_torch_load(str(_DEC_PATH), map_location=device, allow_unsafe=True)
        sd = state.get("decoder_state") if isinstance(state, dict) else state
        _decoder = SegDecoder().to(device)
        _decoder.load_state_dict(sd)
        _decoder.eval()
        return _decoder
    except Exception as e:
        import logging
        logging.getLogger("colonai.seg").warning(
            "seg decoder load failed: %s", e)
        return None


def predict_seg_mask(model, image_tensor, device) -> Optional[np.ndarray]:
    """Run the segmentation decoder on the model's ResNet50 features.

    Returns a 224x224 numpy float array in [0, 1] (sigmoid output), or
    None if the decoder isn't available.
    """
    dec = _load_decoder(device)
    if dec is None:
        return None
    try:
        with torch.no_grad():
            # Use the model's frozen ResNet50 backbone for the features
            feats = model.image_encoder.resnet_backbone(image_tensor.to(device))
            logits = dec(feats)                            # (1, 1, 224, 224)
            prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return prob
    except Exception as e:
        import logging
        logging.getLogger("colonai.seg").warning("seg inference failed: %s", e)
        return None
