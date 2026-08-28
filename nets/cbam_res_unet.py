import torch
import torch.nn as nn
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from nets.cbam import CBAM
except ImportError:
    from cbam import CBAM


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res_cbam_block = ResCBAMBlock(in_size, out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        return self.res_cbam_block(outputs)


class CBAMResUnet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='vgg'):
        super(CBAMResUnet, self).__init__()
        if backbone == 'vgg':
            from nets.vgg import VGG16 as VGG16_func
            self.vgg = VGG16_func(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError('CBAMResUnet currently supports vgg backbone.')

        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            feat1, feat2, feat3, feat4, feat5 = self.vgg.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        return final

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = True


class CBAMResUnetV2(CBAMResUnet):
    """Compatibility alias for older experiment 09 scripts."""
    pass


class CBAMResUnetV3(CBAMResUnet):
    """Compatibility alias for experiment 09 training scripts."""
    pass
