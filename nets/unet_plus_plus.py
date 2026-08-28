"""
UNet++ (Nested U-Net): Dense skip pathways for medical image segmentation.
Reference: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", DLMIA 2018.
"""
import torch
import torch.nn as nn

from nets.vgg import VGG16


class VGGBlock(nn.Module):
    """VGG-style convolution block."""

    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """UNet++ with nested dense skip pathways."""

    def __init__(self, num_classes=2, pretrained=False, backbone="vgg", deep_supervision=False):
        super().__init__()
        if backbone != "vgg":
            raise ValueError("UNetPlusPlus currently supports backbone='vgg'.")
        
        self.deep_supervision = deep_supervision
        
        # Encoder (VGG16 backbone)
        self.vgg = VGG16(pretrained=pretrained)
        
        # Decoder blocks - Level 0 (standard U-Net path)
        self.conv_0_0 = VGGBlock(64, 64)
        self.conv_1_0 = VGGBlock(128, 128)
        self.conv_2_0 = VGGBlock(256, 256)
        self.conv_3_0 = VGGBlock(512, 512)
        self.conv_4_0 = VGGBlock(512, 512)
        
        # Decoder blocks - Level 1 (first nested level)
        self.conv_0_1 = VGGBlock(64 + 128, 64)
        self.conv_1_1 = VGGBlock(128 + 256, 128)
        self.conv_2_1 = VGGBlock(256 + 512, 256)
        self.conv_3_1 = VGGBlock(512 + 512, 512)
        
        # Decoder blocks - Level 2 (second nested level)
        self.conv_0_2 = VGGBlock(64 * 2 + 128, 64)
        self.conv_1_2 = VGGBlock(128 * 2 + 256, 128)
        self.conv_2_2 = VGGBlock(256 * 2 + 512, 256)
        
        # Decoder blocks - Level 3 (third nested level)
        self.conv_0_3 = VGGBlock(64 * 3 + 128, 64)
        self.conv_1_3 = VGGBlock(128 * 3 + 256, 128)
        
        # Decoder blocks - Level 4 (fourth nested level)
        self.conv_0_4 = VGGBlock(64 * 4 + 128, 64)
        
        # Upsampling
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Final classification layers
        self.final_0 = nn.Conv2d(64, num_classes, 1)
        self.final_1 = nn.Conv2d(64, num_classes, 1)
        self.final_2 = nn.Conv2d(64, num_classes, 1)
        self.final_3 = nn.Conv2d(64, num_classes, 1)
        self.final_4 = nn.Conv2d(64, num_classes, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Encoder
        feat1, feat2, feat3, feat4, feat5 = self.vgg(x)
        
        # Level 0
        x_0_0 = feat1
        x_1_0 = feat2
        x_2_0 = feat3
        x_3_0 = feat4
        x_4_0 = feat5
        
        # Level 1
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, self.up(x_1_0)], 1))
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.up(x_2_0)], 1))
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up(x_3_0)], 1))
        x_3_1 = self.conv_3_1(torch.cat([x_3_0, self.up(x_4_0)], 1))
        
        # Level 2
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, self.up(x_1_1)], 1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, self.up(x_2_1)], 1))
        x_2_2 = self.conv_2_2(torch.cat([x_2_0, x_2_1, self.up(x_3_1)], 1))
        
        # Level 3
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up(x_1_2)], 1))
        x_1_3 = self.conv_1_3(torch.cat([x_1_0, x_1_1, x_1_2, self.up(x_2_2)], 1))
        
        # Level 4
        x_0_4 = self.conv_0_4(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, self.up(x_1_3)], 1))
        
        # Output
        if self.deep_supervision:
            out_0 = self.final_0(x_0_0)
            out_1 = self.final_1(x_0_1)
            out_2 = self.final_2(x_0_2)
            out_3 = self.final_3(x_0_3)
            out_4 = self.final_4(x_0_4)
            return [out_0, out_1, out_2, out_3, out_4]
        else:
            return self.final_4(x_0_4)

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = True
