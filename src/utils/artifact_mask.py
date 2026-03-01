import torch

def generate_black_box_mask(image_tensor, threshold=0.05):
    """
    Detect black regions in image.
    image_tensor: [B,3,H,W]
    returns: [B,1,H,W]
    """
    gray = image_tensor.mean(dim=1, keepdim=True)
    mask = (gray < threshold).float()
    return mask