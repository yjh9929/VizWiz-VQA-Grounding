import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, in_channels=768, mid_channels=[1024, 512, 256, 128], out_channels=1):
        """
        U-Net 기반 디코더 (ImageEncoder의 출력 구조에 맞춤)
        in_channels: bottleneck 채널 수 (예: 768)
        mid_channels: 디코더 각 단계 출력 채널 [1024, 512, 256, 128]
        """
        super(UNetDecoder, self).__init__()

        # 업샘플 레이어
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),  # <-- 여기!
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        ])

        # 디코딩 블록 (concat 이후 conv)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels[0] + 1024, mid_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[0], mid_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[0]),
                nn.ReLU(inplace=True)
            ),
            # 2단계 디코더 블록 (1024 + 512 = 1536 → 512)
            nn.Sequential(
                nn.Conv2d(1536, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # 3단계 디코더 블록 (1024 + 256 = 1280 → 256)
            nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[3]),
                nn.ReLU(inplace=True)
            )
        ])

        # 출력층
        self.final_conv = nn.Conv2d(mid_channels[3], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        # 1단계: x(768) → 1024 업샘플 + enc_feat3(1024) concat
        x = self.upconvs[0](x)
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(enc_feat3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat3], dim=1)
        x = self.dec_blocks[0](x)

        # 2단계
        x = self.upconvs[1](x)
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(enc_feat2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat2], dim=1)
        x = self.dec_blocks[1](x)

        # 3단계
        x = self.upconvs[2](x)
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(enc_feat1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat1], dim=1)
        x = self.dec_blocks[2](x)

        # 4단계 (skip 없음)
        x = self.upconvs[3](x)
        x = self.dec_blocks[3](x)

        # 출력 마스크
        x = self.final_conv(x)
        mask = self.activation(x)
        return mask