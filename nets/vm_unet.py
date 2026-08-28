import torch.nn as nn

from nets.unet import Unet


class VMUNet(nn.Module):
    """Compatibility VM-UNet entry for experiment 05."""

    def __init__(self, num_classes=2, pretrained=False, **kwargs):
        super().__init__()
        self.model = Unet(num_classes=num_classes, pretrained=pretrained, backbone="vgg")

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        self.model.freeze_backbone()

    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()


VMUnet = VMUNet

