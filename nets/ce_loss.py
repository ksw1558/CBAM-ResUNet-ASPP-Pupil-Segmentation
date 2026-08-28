import torch
import torch.nn as nn

class CE_Loss(nn.Module):
    def __init__(self, weight=None, num_classes=2):
        super(CE_Loss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, inputs, target):
        return self.criterion(inputs, target)
