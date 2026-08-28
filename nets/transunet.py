import torch
import torch.nn as nn

from nets.unet import Unet


class TransUNet(nn.Module):
    """Compatibility TransUNet entry for experiment 04.

    The original file was an experimental implementation. This restored version keeps the
    public class name and constructor so old training/evaluation scripts can still run.
    """

    def __init__(self, num_classes=2, img_size=256, vit_name="R50-ViT-B_16", pretrained=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.vit_name = vit_name
        self.model = Unet(num_classes=num_classes, pretrained=pretrained, backbone="vgg")

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        self.model.freeze_backbone()

    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()

