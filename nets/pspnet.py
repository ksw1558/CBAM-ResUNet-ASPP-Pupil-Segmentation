import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.vgg import VGG16


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        branch_channels = max(out_channels // len(bins), 1)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.ReLU(inplace=True),
            )
            for bin_size in bins
        ])
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + branch_channels * len(bins), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [x]
        feats.extend(F.interpolate(stage(x), size=size, mode="bilinear", align_corners=True) for stage in self.stages)
        return self.project(torch.cat(feats, dim=1))


class PSPNet(nn.Module):
    """PSPNet style model for experiment 07."""

    def __init__(self, num_classes=2, pretrained=False, backbone="vgg", **kwargs):
        super().__init__()
        if backbone != "vgg":
            raise ValueError("Compatibility PSPNet currently supports backbone='vgg'.")
        self.vgg = VGG16(pretrained=pretrained)
        self.psp = PSPModule(512, 256)
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        *_, feat5 = self.vgg(x)
        out = self.head(self.psp(feat5))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = True

