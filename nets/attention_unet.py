import torch
import torch.nn as nn

from nets.vgg import VGG16


class AttentionGate(nn.Module):
    """注意力门控机制：抑制背景区域，突出目标区域。"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # 门控信号（来自深层特征）
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 跳跃连接信号（来自浅层特征）
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 注意力系数生成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, use_attention=False, gate_channels=None, skip_channels=None):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 注意力门控（仅在跳跃连接上使用）
        if use_attention:
            # skip_channels 必须显式指定，表示 inputs1 (跳跃连接) 的通道数
            if skip_channels is None:
                raise ValueError("When use_attention=True, skip_channels must be specified!")
            
            # gate_channels 必须显式指定，表示 inputs2 的通道数
            if gate_channels is None:
                raise ValueError("When use_attention=True, gate_channels must be specified!")
            
            self.attention = AttentionGate(
                F_g=gate_channels, 
                F_l=skip_channels, 
                F_int=gate_channels // 2
            )
        else:
            self.attention = None
        self.use_attention = use_attention

    def forward(self, inputs1, inputs2):
        upsampled = self.up(inputs2)
        if self.use_attention and self.attention is not None:
            inputs1 = self.attention(upsampled, inputs1)
        outputs = torch.cat([inputs1, upsampled], 1)
        return self.conv(outputs)


class AttentionUnet(nn.Module):
    """Attention U-Net with VGG16 backbone."""

    def __init__(self, num_classes=2, pretrained=False, backbone="vgg"):
        super().__init__()
        if backbone != "vgg":
            raise ValueError("AttentionUnet currently supports backbone='vgg'.")
        self.backbone = backbone
        self.vgg = VGG16(pretrained=pretrained)
        
        # Decoder with attention gates on skip connections
        # VGG16 features: feat1=64, feat2=128, feat3=256, feat4=512, feat5=512
        
        # up_concat4: inputs1=feat4(512), inputs2=feat5(512), output=512
        # Attention: F_g=512 (feat5), F_l=512 (feat4), F_int=256
        self.up_concat4 = unetUp(512 + 512, 512, use_attention=True, gate_channels=512, skip_channels=512)
        
        # up_concat3: inputs1=feat3(256), inputs2=up4(512), output=256
        # Attention: F_g=512 (up4), F_l=256 (feat3), F_int=256
        self.up_concat3 = unetUp(256 + 512, 256, use_attention=True, gate_channels=512, skip_channels=256)
        
        # up_concat2: inputs1=feat2(128), inputs2=up3(256), output=128
        # Attention: F_g=256 (up3), F_l=128 (feat2), F_int=128
        self.up_concat2 = unetUp(128 + 256, 128, use_attention=True, gate_channels=256, skip_channels=128)
        
        # up_concat1: inputs1=feat1(64), inputs2=up2(128), output=64
        # Attention: F_g=128 (up2), F_l=64 (feat1), F_int=64
        self.up_concat1 = unetUp(64 + 128, 64, use_attention=True, gate_channels=128, skip_channels=64)
        
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


# Compatibility alias used by experiment 04 scripts.
# The architecture is the Attention U-Net with attention gates on skip paths.
AttentionGateUnet = AttentionUnet
