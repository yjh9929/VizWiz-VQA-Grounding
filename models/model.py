import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, self.image_encoder.out_channels),
            nn.ReLU(),
            nn.Linear(self.image_encoder.out_channels, self.image_encoder.out_channels),
        )

        self.decoder = UNetDecoder(in_channels=self.image_encoder.out_channels)

    def forward(self, image, text):
        bottleneck, enc3, enc2, enc1 = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        fused = bottleneck * weight  # (B, C, H, W)

        output = self.decoder(fused, enc3, enc2, enc1)
        return output
