import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(self.image_encoder.out_channels, self.image_encoder.out_channels),
            nn.ReLU(),
            nn.Linear(self.image_encoder.out_channels, self.image_encoder.out_channels)
        )
        self.decoder = UNetDecoder(in_channels=self.image_encoder.out_channels)

    def forward(self, image, text):
        image_feat = self.image_encoder(image)              # (B, C, H, W)
        text_feat = self.text_encoder(text)                 # (B, D)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        fused = image_feat * weight                         # (B, C, H, W)
        mask = self.decoder(fused)                          # (B, 1, H, W)
        return mask
