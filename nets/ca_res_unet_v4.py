import torch
import torch.nn as nn

from nets.ca import CoordAtt
from nets.vgg import VGG16


class _CAUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            CoordAtt(out_size),
        )

    def forward(self, feat, up):
        return self.block(torch.cat([feat, self.up(up)], dim=1))


class CAResUnetV4(nn.Module):
    """Coordinate-attention U-Net variant for experiment 10."""

    def __init__(self, num_classes=2, pretrained=False, backbone="vgg", **kwargs):
        super().__init__()
        if backbone != "vgg":
            raise ValueError("Compatibility CAResUnetV4 currently supports backbone='vgg'.")
        self.vgg = VGG16(pretrained=pretrained)
        self.up_concat4 = _CAUp(512 + 512, 512)
        self.up_concat3 = _CAUp(256 + 512, 256)
        self.up_concat2 = _CAUp(128 + 256, 128)
        self.up_concat1 = _CAUp(64 + 128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        return self.final(up1)

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = True

