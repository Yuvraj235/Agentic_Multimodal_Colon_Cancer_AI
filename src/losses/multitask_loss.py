import torch.nn as nn
from src.losses.dice_loss import DiceLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, seg_preds, cls_targets, seg_targets):

        loss_cls = self.ce(logits, cls_targets)
        loss_seg = self.dice(seg_preds, seg_targets)

        total_loss = self.alpha * loss_cls + self.beta * loss_seg

        return total_loss, loss_cls, loss_seg