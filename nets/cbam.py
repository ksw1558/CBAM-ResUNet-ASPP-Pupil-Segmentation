#CBAM卷积块注意力模块，是一个轻量级注意力组件，由通道注意力模块和空间注意力模块两部分串联组成。
import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    """通道注意力模块：判断哪些特征通道更重要。"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化都压缩到1x1，用来概括每个通道的整体响应。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享的两层1x1卷积相当于一个轻量MLP，先降维再升维，学习通道权重。
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化关注整体分布，最大池化关注显著响应；两者相加后得到通道注意力图。
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块：判断图像中哪些位置更重要。"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 输入是两个空间图：通道平均图和通道最大图；输出是 1 个空间权重图。
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道方向压缩，保留每个像素位置的平均响应和最强响应。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """CBAM = Channel Attention + Spatial Attention。

    在实验11中，CBAM 用于增强瞳孔相关特征，抑制眼睑、睫毛、反光等干扰区域。
    """

    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention()
        # 保存最近一次前向传播的空间注意力图，便于后续可视化或调试。
        self.spatial_attn_map = None

    def forward(self, x):
        # 先做通道注意力：让网络选择“哪些语义通道”更有用。
        x = x * self.ChannelGate(x)
        # 再做空间注意力：让网络选择“图像中的哪些位置”更有用。
        self.spatial_attn_map = self.SpatialGate(x)
        x = x * self.spatial_attn_map
        return x
