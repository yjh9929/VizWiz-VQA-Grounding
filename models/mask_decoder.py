import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(x)
