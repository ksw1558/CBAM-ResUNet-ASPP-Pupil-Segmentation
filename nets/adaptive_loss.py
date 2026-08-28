import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCombinedLoss(nn.Module):
    """Older adaptive loss used by ablation experiments."""

    def __init__(self, w_tversky=1.0, w_focal=1.0, w_edge=0.2,
                 alpha=0.75, beta=0.25, gamma=0.75):
        super().__init__()
        self.w_tversky = w_tversky
        self.w_focal = w_focal
        self.w_edge = w_edge
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def tversky_loss(self, inputs, target, alpha=0.75, beta=0.25, smooth=1e-5):
        probs = torch.softmax(inputs, dim=1)[:, 1]
        target = (target == 1).float()
        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()
        return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    def focal_tversky_loss(self, inputs, target, alpha=0.75, beta=0.25, gamma=0.75, smooth=1e-5):
        tversky = 1 - self.tversky_loss(inputs, target, alpha, beta, smooth)
        return torch.pow((1 - tversky), gamma)

    def edge_aware_loss(self, inputs, target):
        probs = torch.softmax(inputs, dim=1)[:, 1]
        diff_x = torch.abs(probs[:, :, 1:] - probs[:, :, :-1]).mean()
        diff_y = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]).mean()
        return diff_x + diff_y

    def forward(self, inputs, target):
        loss_t = self.tversky_loss(inputs, target, self.alpha, self.beta)
        loss_f = self.focal_tversky_loss(inputs, target, self.alpha, self.beta, self.gamma)
        loss_e = self.edge_aware_loss(inputs, target)
        return self.w_tversky * loss_t + self.w_focal * loss_f + self.w_edge * loss_e

