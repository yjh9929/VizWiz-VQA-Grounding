import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.text_encoder = TextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, self.image_encoder.out_channels),
            nn.ReLU(),
            nn.Linear(self.image_encoder.out_channels, self.image_encoder.out_channels),
        )

        self.decoder = UNetDecoder(in_channels=self.image_encoder.out_channels)

    def forward(self, image, text):
        bottleneck = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (batch, 768, 1, 1)
        fused = bottleneck * weight

        output = self.decoder(fused)
        return output
