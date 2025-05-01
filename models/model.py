import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.text_encoder = TextEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )
        self.decoder = UNetDecoder(in_channels=1024, mid_channels=[1024, 512, 256, 128])

    def forward(self, image, text):
        enc_feat1, enc_feat2, enc_feat3, bottleneck = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, 768, 1, 1)
        fused = bottleneck * weight

        output = self.decoder(fused, enc_feat3, enc_feat2, enc_feat1)
        return output
