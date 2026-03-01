import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice

def masked_dice_loss(pred, target, artifact_mask):
    valid_region = 1 - artifact_mask
    pred = pred * valid_region
    target = target * valid_region
    return dice_loss(pred, target)