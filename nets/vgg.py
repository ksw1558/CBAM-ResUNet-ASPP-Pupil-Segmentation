import torch
import torch.nn as nn
from torchvision.models import vgg16


class VGG16(nn.Module):
    """VGG16 backbone used by the earlier U-Net style experiments."""

    def __init__(self, pretrained=False):
        super().__init__()
        model = vgg16(pretrained=pretrained)
        features = list(model.features.children())
        self.stage1 = nn.Sequential(*features[:4])
        self.stage2 = nn.Sequential(*features[4:9])
        self.stage3 = nn.Sequential(*features[9:16])
        self.stage4 = nn.Sequential(*features[16:23])
        self.stage5 = nn.Sequential(*features[23:30])

    def forward(self, x):
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        return feat1, feat2, feat3, feat4, feat5

