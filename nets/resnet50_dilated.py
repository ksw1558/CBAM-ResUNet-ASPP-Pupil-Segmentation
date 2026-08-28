import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Dilated(nn.Module):
    """
    ResNet50 with Dilated Convolutions (Output stride=16)
    
    标准DeepLabV3做法：
    - layer3: 完全不变（stride=32）
    - layer4: stride从2改为1，dilation=2（保持感受野）
    - 最终output stride = 16（而非原始的32）
    """
    
    def __init__(self, pretrained=True):
        super(ResNet50Dilated, self).__init__()
        
        resnet = models.resnet50(pretrained=pretrained)
        
        # Layer0: conv1 + bn1 + relu + maxpool (stride=4)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # Layer1: stride=8 (不修改)
        self.layer1 = resnet.layer1
        
        # Layer2: stride=16 (不修改)
        self.layer2 = resnet.layer2
        
        # Layer3: 完全不变 (stride=32)
        self.layer3 = resnet.layer3
        
        # Layer4: 【关键修改】stride从2改为1，dilation=2
        self.layer4 = self._create_dilated_layer4(resnet.layer4)
    
    def _create_dilated_layer4(self, original_layer4):
        """
        创建新的layer4：
        1. 第一个block: stride=1, dilation=2
        2. 后续block: stride=1, dilation=2
        """
        from torchvision.models.resnet import Bottleneck
        
        new_blocks = []
        
        for idx, block in enumerate(original_layer4):
            # 确定downsample
            downsample = None
            if block.downsample is not None:
                if idx == 0:
                    # 第一个block: downsample stride从2改为1
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            block.downsample[0].in_channels,
                            block.downsample[0].out_channels,
                            kernel_size=1,
                            stride=1,  # 从2改为1
                            bias=False
                        ),
                        nn.BatchNorm2d(block.downsample[0].out_channels)
                    )
                    # 复制权重
                    downsample[0].weight.data = block.downsample[0].weight.data.clone()
                    downsample[1].weight.data = block.downsample[1].weight.data.clone()
                    downsample[1].bias.data = block.downsample[1].bias.data.clone()
                    downsample[1].running_mean.data = block.downsample[1].running_mean.data.clone()
                    downsample[1].running_var.data = block.downsample[1].running_var.data.clone()
                else:
                    # 后续block: 没有downsample或保持不变
                    downsample = block.downsample
            
            # 创建新block
            planes = block.conv2.out_channels
            new_block = Bottleneck(
                inplanes=block.conv1.in_channels,
                planes=planes,
                stride=1,  # 全部设为1
                downsample=downsample
            )
            
            # 设置dilation=2
            new_block.conv2.dilation = (2, 2)
            new_block.conv2.padding = (2, 2)
            
            # 复制权重
            new_block.conv1.weight.data = block.conv1.weight.data.clone()
            new_block.bn1.weight.data = block.bn1.weight.data.clone()
            new_block.bn1.bias.data = block.bn1.bias.data.clone()
            new_block.bn1.running_mean.data = block.bn1.running_mean.data.clone()
            new_block.bn1.running_var.data = block.bn1.running_var.data.clone()
            
            new_block.conv2.weight.data = block.conv2.weight.data.clone()
            new_block.bn2.weight.data = block.bn2.weight.data.clone()
            new_block.bn2.bias.data = block.bn2.bias.data.clone()
            new_block.bn2.running_mean.data = block.bn2.running_mean.data.clone()
            new_block.bn2.running_var.data = block.bn2.running_var.data.clone()
            
            new_block.conv3.weight.data = block.conv3.weight.data.clone()
            new_block.bn3.weight.data = block.bn3.weight.data.clone()
            new_block.bn3.bias.data = block.bn3.bias.data.clone()
            new_block.bn3.running_mean.data = block.bn3.running_mean.data.clone()
            new_block.bn3.running_var.data = block.bn3.running_var.data.clone()
            
            new_blocks.append(new_block)
        
        return nn.Sequential(*new_blocks)
    
    def forward(self, x):
        feat1 = self.layer1(self.layer0(x))  # stride=8, size=32x32
        feat2 = self.layer2(feat1)           # stride=16, size=16x16
        feat3 = self.layer3(feat2)           # stride=32, size=8x8
        feat4 = self.layer4(feat3)           # stride=16 (modified), size=16x16
        feat5 = feat4
        
        return feat1, feat2, feat3, feat4, feat5
