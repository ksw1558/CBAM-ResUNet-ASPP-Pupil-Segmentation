import torch
import torch.nn as nn

from nets.cbam import CBAM
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, use_cbam=False):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attn = CBAM(out_size) if use_cbam else nn.Identity()

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        return self.attn(self.conv(outputs))


class Unet(nn.Module):
    """Basic VGG U-Net kept for experiments 01-03 and fallback visualizers."""

    def __init__(self, num_classes=2, pretrained=False, backbone="vgg", use_cbam=False):
        super().__init__()
        if backbone != "vgg":
            raise ValueError("Compatibility Unet currently supports backbone='vgg'.")
        self.backbone = backbone
        self.vgg = VGG16(pretrained=pretrained)
        self.up_concat4 = unetUp(512 + 512, 512, use_cbam=use_cbam)
        self.up_concat3 = unetUp(256 + 512, 256, use_cbam=use_cbam)
        self.up_concat2 = unetUp(128 + 256, 128, use_cbam=use_cbam)
        self.up_concat1 = unetUp(64 + 128, 64, use_cbam=use_cbam)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(inputs)
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

