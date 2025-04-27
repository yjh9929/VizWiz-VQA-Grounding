import torch.nn as nn
from models import TextEncoder, ImageEncoder, UNetDecoder

class GroundingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.text_encoder = TextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(512, 512),  # self.image_encoder.out_channels가 512니까
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.decoder = UNetDecoder(in_channels=512)

    def forward(self, image, text):
        enc_feat1, enc_feat2, enc_feat3, bottleneck = self.image_encoder(image)
        text_feat = self.text_encoder(text)

        weight = self.fusion(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        fused = bottleneck * weight                                 # (B, 512, H, W)

        # decoder에 fused와 enc_feat3, enc_feat2, enc_feat1을 넣어줘야 함!
        output = self.decoder(fused, enc_feat3, enc_feat2, enc_feat1)
        return output


