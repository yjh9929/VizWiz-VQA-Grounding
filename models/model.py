import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()  # model_name 기본값 "openai/clip-vit-base-patch32"
        self.text_encoder = TextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(512, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 1024),# 512 → 1024
        )
        self.decoder = UNetDecoder(in_channels=1024)  # 512 → 1024

    def forward(self, image, text):
        bottleneck = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (batch, 768, 1, 1)
        fused = bottleneck * weight

        output = self.decoder(fused)
        return output