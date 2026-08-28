import torch
import torch.nn as nn

from nets.aspp import ASPP
from nets.cbam import CBAM
from nets.ca import CoordAtt
from nets.cbam_res_unet import unetUp as unetUpCBAM
from nets.res_unet import unetUp as unetUpRes
from nets.vgg import VGG16


class AblationResUnet(nn.Module):
    """Configurable U-Net variant for ablation tools.

    Supported switches:
    - use_aspp: add ASPP at the bottleneck
    - use_cbam: add CBAM attention after bottleneck
    - use_ca: add coordinate attention after bottleneck
    """

    def __init__(self, num_classes=2, pretrained=False, use_aspp=True,
                 use_cbam=True, use_ca=False, **kwargs):
        super().__init__()
        self.vgg = VGG16(pretrained=pretrained)
        self.use_aspp = use_aspp
        self.use_cbam = use_cbam
        self.use_ca = use_ca
        self.aspp = ASPP(512, 512) if use_aspp else nn.Identity()
        self.cbam = CBAM(512) if use_cbam else nn.Identity()
        self.ca = CoordAtt(512) if use_ca else nn.Identity()
        self.up_concat4 = unetUpCBAM(512 + 512, 512)
        self.up_concat3 = unetUpCBAM(256 + 512, 256)
        self.up_concat2 = unetUpRes(128 + 256, 128)
        self.up_concat1 = unetUpRes(64 + 128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        feat5 = self.ca(self.cbam(self.aspp(feat5)))
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

