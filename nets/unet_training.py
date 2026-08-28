import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(net, init_type="normal", init_gain=0.02):
    """Initialize convolution and batch-norm layers for older training scripts."""
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if init_type == "normal":
                nn.init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            else:
                nn.init.orthogonal_(module.weight.data, gain=init_gain)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05,
                     warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    """Return a scheduler function compatible with the original experiment scripts."""
    if lr_decay_type == "cos":
        warmup_total_iters = min(max(int(warmup_iters_ratio * total_iters), 1), 3)
        no_aug_iter = min(max(int(no_aug_iter_ratio * total_iters), 1), 15)

        def scheduler(iters):
            if iters <= warmup_total_iters:
                warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
                return (lr - warmup_lr_start) * (iters / warmup_total_iters) ** 2 + warmup_lr_start
            if iters >= total_iters - no_aug_iter:
                return min_lr
            cos_iter = iters - warmup_total_iters
            cos_total = total_iters - warmup_total_iters - no_aug_iter
            return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * cos_iter / cos_total))
    else:
        decay_rate = (min_lr / lr) ** (1 / max(step_num - 1, 1))
        step_size = total_iters / step_num

        def scheduler(iters):
            return lr * decay_rate ** (iters // step_size)

    return scheduler


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class EdgeAwareLoss(nn.Module):
    def forward(self, inputs, target):
        probs = torch.softmax(inputs, dim=1)[:, 1]
        diff_x = torch.abs(probs[:, :, 1:] - probs[:, :, :-1]).mean()
        diff_y = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]).mean()
        return diff_x + diff_y


def Tversky_loss(inputs, target, alpha=0.7, beta=0.3, smooth=1e-5):
    probs = torch.softmax(inputs, dim=1)[:, 1]
    target = (target == 1).float()
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def CE_Loss(inputs, target, cls_weights, num_classes=2):
    weights = torch.as_tensor(cls_weights, device=inputs.device, dtype=inputs.dtype)
    return F.cross_entropy(inputs, target.long(), weight=weights)


def Focal_Loss(inputs, target, cls_weights, num_classes=2, alpha=0.5, gamma=2):
    logpt = F.log_softmax(inputs, dim=1)
    target_one_hot = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
    pt = (target_one_hot * logpt).sum(dim=1)
    loss = -alpha * (1 - torch.exp(pt)) ** gamma * pt
    return loss.mean()


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    probs = torch.softmax(inputs, dim=1)
    target_one_hot = torch.zeros_like(probs).scatter_(1, target.long().unsqueeze(1), 1)
    intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    score = (2 * intersection + smooth) / (union + smooth)
    return 1 - torch.mean(score)

