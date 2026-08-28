import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    """Focal Loss：降低简单样本权重，强化困难像素的学习。

    对瞳孔分割来说，大量背景像素很容易分类，真正困难的是边缘、反光、
    遮挡等区域。Focal Loss 会让模型更关注这些容易分错的像素。
    """

    def __init__(self, alpha=0.5, gamma=2, num_classes=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, target):
        # 先计算每个像素属于各类别的 log 概率。
        inputs = F.log_softmax(inputs, dim=1)
        # 将 [B,H,W] 的标签转成 one-hot，方便取出真实类别对应的概率。
        target_one_hot = torch.zeros_like(inputs).scatter_(1, target.unsqueeze(1), 1)

        # pt 是真实类别的 log 概率；预测越准，exp(pt) 越接近 1。
        pt = (target_one_hot * inputs).sum(dim=1)
        # alpha 用于平衡正负样本权重。
        at = self.alpha * target_one_hot.sum(dim=1) + (1 - self.alpha) * (1 - target_one_hot).sum(dim=1)

        # (1 - p)^gamma 是困难样本加权项：预测越错，权重越大。
        focal_weight = (1 - torch.exp(pt)) ** self.gamma
        loss = -at * focal_weight * pt
        return loss.mean()
