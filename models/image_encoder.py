import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            self.out_channels = 512
        else:
            raise ValueError("Unsupported backbone")
        self.backbone = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.backbone(x)  # (B, C, H, W)
