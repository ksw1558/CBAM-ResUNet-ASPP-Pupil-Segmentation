import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from nets.cbam import CBAM
from nets.aspp import ASPP
from nets.cbam_res_unet import ResCBAMBlock, unetUp as unetUpCBAM
from nets.res_unet import ResBlock, unetUp as unetUpRes
from nets.resnet50_dilated import ResNet50Dilated


class CBAMResUnetV5(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, backbone='resnet50'):
        super(CBAMResUnetV5, self).__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.backbone = ResNet50Dilated(pretrained=pretrained)
            
        elif backbone == 'vgg':
            from nets.vgg import VGG16 as VGG16_func
            self.vgg = VGG16_func(pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone.')

        out_filters = [64, 128, 256, 512]

        bottleneck_input = 2048 if backbone == 'resnet50' else 512
        self.bottleneck_aspp = ASPP(bottleneck_input, 512)
        self.bottleneck_cbam = CBAM(512)

        if backbone == 'resnet50':
            # 【完全重写decoder以适配dilated backbone】
            # feat3: 8x8, 1024ch | feat5: 16x16, 512ch
            
            # Step 1: 上采样feat3从8x8到16x16，然后concat with feat5
            self.up4_conv = nn.Sequential(
                nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            
            # Step 2: up4 (16x16) -> up3 (32x32)，与feat2 (16x16) concat
            self.up_concat3 = unetUpCBAM(512 + 512, 256)  # 输入1024ch
            
            # Step 3: up3 (32x32) -> up2 (64x64)，与feat1 (32x32) concat  
            self.up_concat2 = unetUpRes(256 + 256, 128)  # 输入512ch
            
            # Step 4: up2 -> up1 (256x256)
            self.up_concat1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),  # 64->128
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.UpsamplingBilinear2d(scale_factor=2),  # 128->256
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            in_filters = [192, 384, 768, 1024]
            self.up_concat4 = unetUpCBAM(in_filters[3], out_filters[3])
            self.up_concat3 = unetUpCBAM(in_filters[2], out_filters[2])
            self.up_concat2 = unetUpRes(in_filters[1], out_filters[1])
            self.up_concat1 = unetUpRes(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        if self.backbone_name == "resnet50":
            feat1, feat2, feat3, feat4, feat5 = self.backbone(inputs)
            # feat1: 32x32, 256ch
            # feat2: 16x16, 512ch
            # feat3: 8x8, 1024ch
            # feat5: 16x16, 512ch (after ASPP+CBAM)
        elif self.backbone_name == "vgg":
            feat1, feat2, feat3, feat4, feat5 = self.vgg.forward(inputs)

        feat5 = self.bottleneck_aspp(feat5)
        feat5 = self.bottleneck_cbam(feat5)

        if self.backbone_name == "resnet50":
            # Decoder for dilated backbone
            # Step 1: 上采样feat3 (8x8) 到 16x16，与feat5 concat
            feat3_up = F.interpolate(feat3, size=feat5.shape[2:], mode='bilinear', align_corners=True)
            concat_4 = torch.cat([feat3_up, feat5], dim=1)  # 16x16, 1536ch
            up4 = self.up4_conv(concat_4)  # 16x16, 512ch
            
            # Step 2: 上采样up4到32x32，与feat2 concat
            up3 = self.up_concat3(feat2, up4)  # 32x32, 256ch
            
            # Step 3: 上采样up3到64x64，与feat1 concat
            up2 = self.up_concat2(feat1, up3)  # 64x64, 128ch
            
            # Step 4: 上采样到256x256
            up1 = self.up_concat1(up2)  # 256x256, 64ch
        else:
            up4 = self.up_concat4(feat4, feat5)
            up3 = self.up_concat3(feat3, up4)
            up2 = self.up_concat2(feat2, up3)
            up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        return final

    def freeze_backbone(self):
        if self.backbone_name == "resnet50":
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.vgg.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone_name == "resnet50":
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for param in self.vgg.parameters():
                param.requires_grad = True
