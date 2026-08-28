import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from nets.aspp import ASPP


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ style segmentation model for experiment 06."""

    def __init__(self, num_classes=2, pretrained_backbone=False, output_stride=16, **kwargs):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained_backbone)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, 256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.layer0(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        low = self.low_proj(low)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=True)
        x = self.decoder(torch.cat([x, low], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

    def freeze_backbone(self):
        for module in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for module in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = True

