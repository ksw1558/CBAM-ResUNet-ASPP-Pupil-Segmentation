#ASPP空洞空间金字塔池化，包含1×1卷积、三组不同膨胀率的空洞卷积和全局池化共5个并行分支。
import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    """空洞空间金字塔池化模块。

    ASPP 使用不同 dilation 的卷积分支同时观察小范围和大范围上下文。
    对瞳孔分割来说，它可以帮助模型兼顾瞳孔边缘细节和眼部整体结构。
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 分支1：1x1卷积，保留原始局部信息并调整通道数。
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 分支2：3x3空洞卷积，膨胀率=6，感受野较小。
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 分支3：3x3空洞卷积，dilation=12，获得中等尺度上下文。
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 分支4：3x3空洞卷积，dilation=18，获得更大尺度上下文。
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # 分支5：全局图像池化，提供整张图的全局语义先验。
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合层：把 5 个分支拼接后的特征压回out_channels，作为后续解码器输入。
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 五个分支并行处理同一份输入特征，输出尺寸保持一致。
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        x5 = self.pool(x)
        # 全局池化分支原本是 1x1，需要上采样回其他分支的空间尺寸再拼接。
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)

        # 按通道维度拼接多尺度特征，再通过 project 完成融合。
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.project(out)
        return out
