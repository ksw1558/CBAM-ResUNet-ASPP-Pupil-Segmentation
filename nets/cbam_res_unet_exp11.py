"""ACR-UNet final inference model.

This module defines the final model used by experiment 11:
ResNet50 encoder + ASPP multi-scale context + CBAM attention + U-Net decoder.
It is kept compatible with ``final_exp11_miou98_16_epoch070.pth``.
"""

import torch.nn as nn
import torchvision.models as models

from nets.aspp import ASPP
from nets.cbam import CBAM
from nets.cbam_res_unet import unetUp as unetUpCBAM
from nets.res_unet import unetUp as unetUpRes


class CBAMResUnetExp11(nn.Module):
    """Experiment 11 final ACR-UNet inference structure."""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        # Encoder: ResNet50 extracts hierarchical visual features.
        # The layer names match the saved final checkpoint.
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Bottleneck: ASPP enlarges the receptive field, then CBAM reweights
        # important channels and spatial regions.
        self.bottleneck_aspp = ASPP(2048, 512)
        self.bottleneck_cbam = CBAM(512)

        # Decoder: progressively upsamples deep semantic features and fuses
        # them with encoder features to recover pupil boundaries.
        self.up_concat4 = unetUpCBAM(1024 + 512, 512)
        self.up_concat3 = unetUpCBAM(512 + 512, 256)
        self.up_concat2 = unetUpRes(256 + 256, 128)
        self.up_concat1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Pixel-level classifier. In this project num_classes=2:
        # background and pupil.
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        feat0 = self.layer0(inputs)
        feat0 = self.maxpool(feat0)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        feat5 = self.bottleneck_aspp(feat4)
        feat5 = self.bottleneck_cbam(feat5)

        up4 = self.up_concat4(feat3, feat5)
        up3 = self.up_concat3(feat2, up4)
        up2 = self.up_concat2(feat1, up3)
        up1 = self.up_concat1(up2)

        return self.final(up1)

    def freeze_backbone(self):
        """Freeze the ResNet50 encoder during warm-up training."""
        for module in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the ResNet50 encoder for full fine-tuning."""
        for module in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = True
