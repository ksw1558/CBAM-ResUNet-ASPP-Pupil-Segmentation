import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """边界损失：约束预测边缘和真实边缘尽量一致。

    瞳孔最终还要做椭圆拟合、中心点和直径计算，所以边界质量很重要。
    这里使用 Sobel 算子分别提取预测掩码和真实标签的边缘，再计算两者差异。
    """

    def __init__(self, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.weight = weight

    def extract_edges(self, x):
        """使用 Sobel 算子提取边缘强度图。

        Args:
            x: 二值掩码或概率图，形状为 (B, 1, H, W)

        Returns:
            edge_map: 边缘强度图，形状为 (B, 1, H, W)
        """
        # Sobel 卷积只处理单通道；如果输入有多个通道，则取第一个通道。
        if x.shape[1] > 1:
            x = x[:, 0:1]

        # 横向 Sobel 核：检测左右方向的灰度变化。
        # device 和 dtype 跟随输入，兼容 GPU 和混合精度训练。
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            device=x.device,
            dtype=x.dtype
        ).view(1, 1, 3, 3)

        # 纵向 Sobel 核：检测上下方向的灰度变化。
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            device=x.device,
            dtype=x.dtype
        ).view(1, 1, 3, 3)

        # 分别计算水平和垂直方向的边缘响应。
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)

        # 合成边缘强度，数值越大表示该位置越可能是边界。
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        return edge_mag

    def forward(self, pred, target):
        """计算预测边缘和真实边缘之间的 L1 差异。

        Args:
            pred: 模型预测概率图，形状为 (B, C, H, W)，通常已经 softmax
            target: 真实标签，形状为 (B, H, W) 或 (B, 1, H, W)

        Returns:
            loss: 标量边界损失
        """
        # 二分类中第 1 类是瞳孔，因此只取瞳孔概率图来计算边界。
        if pred.shape[1] > 1:
            pred_pupil = pred[:, 1:2, :, :]
        else:
            pred_pupil = pred

        # 将不同格式的 target 统一成 (B,1,H,W) 的瞳孔二值图。
        if target.dim() == 3:
            target_pupil = (target == 1).float().unsqueeze(1)
        elif target.dim() == 4:
            if target.shape[1] == 1:
                target_pupil = target.float()
            elif target.shape[1] == 2:
                target_pupil = target[:, 1:2, :, :].float()
            else:
                target_pupil = target[:, 0:1, :, :].float()
        else:
            raise ValueError(f"Unsupported target dimension: {target.shape}")

        # 分别提取预测边缘和真实边缘。
        pred_edge = self.extract_edges(pred_pupil)
        target_edge = self.extract_edges(target_pupil)

        # 用 L1 距离约束两张边缘图接近。
        boundary_loss = F.l1_loss(pred_edge, target_edge, reduction='mean')

        return self.weight * boundary_loss
