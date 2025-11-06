
import torch
import torch.nn as nn

def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-7):
    # pred and target are tensors with values in [0,1], shape [B,1,H,W]
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, target):
        # logits: raw (not sigmoid). target in {0,1}
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, target, eps=self.eps)
        return 1.0 - dice

def iou_score(pred: torch.Tensor, target: torch.Tensor, thr=0.5, eps=1e-7):
    pred_bin = (pred > thr).float()
    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    target_flat = target.view(B, -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()