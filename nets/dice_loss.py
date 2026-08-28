import torch
import torch.nn as nn


class Dice_Loss(nn.Module):
    """Dice Loss：衡量预测区域和真实区域的重叠程度。

    在瞳孔分割中，瞳孔区域通常比背景小很多。Dice Loss 不直接按像素数量平均，
    而是关注预测掩码和真实掩码的交并关系，因此适合处理前景/背景不均衡问题。
    """

    def __init__(self, num_classes=2):
        super(Dice_Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, target, smooth=1e-5):
        # inputs 是模型输出的 logits，先通过 softmax 转成每个类别的概率。
        inputs = torch.softmax(inputs, dim=1)
        # target 原本是 [B,H,W] 的类别编号，这里转成 one-hot，形状与 inputs 一致。
        target_one_hot = torch.zeros_like(inputs).scatter_(1, target.unsqueeze(1), 1)

        # intersection 表示预测和真实标签的重叠区域，union 表示两者区域大小之和。
        intersection = (inputs * target_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

        # Dice 越接近 1 表示分割越准确；训练时最小化 1-Dice。
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
