
import torch.nn as nn
from torchvision.models import resnet18


class XRayResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # For torchvision >= 0.13 you can use weights parameter, here we keep it simple:
        self.backbone = resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

