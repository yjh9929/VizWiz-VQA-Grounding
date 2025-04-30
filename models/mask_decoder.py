import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDecoder(nn.Module):
    def __init__(self, in_channels=512, mid_channels=[256, 128, 64, 32], out_channels=1):
        """
        U-Net 기반 디코더를 구성합니다.
        in_channels: 인코더 최종 출력 채널 수 (예: 512)
        mid_channels: 업샘플 단계별 출력 채널 리스트 (고해상도부터 순차적으로 감소)
        out_channels: 출력 마스크 채널 (바이너리 마스크이므로 기본값 1)
        """
        super(UNetDecoder, self).__init__()
        # 업샘플 단계 (ConvTranspose2d) 레이어들을 정의
        # 각 단계에서 채널 수를 mid_channels에 맞게 감소시키면서 공간 해상도 2배 확대
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, mid_channels[0], kernel_size=2, stride=2),    # 예: 512 -> 256 채널 업샘플
            nn.ConvTranspose2d(mid_channels[0], mid_channels[1], kernel_size=2, stride=2), # 예: 256 -> 128 채널 업샘플
            nn.ConvTranspose2d(mid_channels[1], mid_channels[2], kernel_size=2, stride=2), # 예: 128 -> 64 채널 업샘플
            nn.ConvTranspose2d(mid_channels[2], mid_channels[3], kernel_size=2, stride=2)  # 예: 64 -> 32  채널 업샘플
        ])
        # 업샘플 후에 이어지는 합성곱 블록들을 정의 (Conv2d + BatchNorm2d + ReLU 반복)
        # 각 합성곱 블록은 업샘플 출력과 대응되는 인코더 특징맵이 concatenation된 입력을 처리
        self.dec_blocks = nn.ModuleList([
            # 첫 번째 디코더 블록: 업샘플 256채널 + 인코더 skip 256채널 = 입력 512채널 -> 출력 256채널
            nn.Sequential(
                nn.Conv2d(mid_channels[0] * 2, mid_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[0], mid_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[0]),
                nn.ReLU(inplace=True)
            ),
            # 두 번째 디코더 블록: 업샘플 128채널 + 인코더 skip 128채널 = 입력 256채널 -> 출력 128채널
            nn.Sequential(
                nn.Conv2d(mid_channels[1] * 2, mid_channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[1], mid_channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[1]),
                nn.ReLU(inplace=True)
            ),
            # 세 번째 디코더 블록: 업샘플 64채널 + 인코더 skip 64채널 = 입력 128채널 -> 출력 64채널
            nn.Sequential(
                nn.Conv2d(mid_channels[2] * 2, mid_channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[2], mid_channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[2]),
                nn.ReLU(inplace=True)
            ),
            # 네 번째 디코더 블록: 업샘플 32채널 (skip 연결 없음, 최종 단계) -> 출력 32채널
            nn.Sequential(
                nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels[3], mid_channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels[3]),
                nn.ReLU(inplace=True)
            )
        ])
        # 최종 출력 1x1 합성곱 층과 활성화 함수 (Sigmoid)
        self.final_conv = nn.Conv2d(mid_channels[3], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
    
    import torch.nn.functional as F

    def forward(self, x, enc_feat3, enc_feat2, enc_feat1):
        """
        x: 인코더의 최종 출력 특징 (예: 512채널 bottleneck 특징맵)
        enc_feat3: 인코더의 중간 특징 (x보다 한 단계 높은 해상도, 예: 256채널, 1/8 스케일)
        enc_feat2: 그 위 인코더 특징 (예: 128채널, 1/4 스케일)
        enc_feat1: 얕은 인코더 특징 (예: 64채널, 1/2 스케일)
        """
        # 1단계 업샘플 + skip 연결 + 합성곱
        x = self.upconvs[0](x)  # 512 -> 256채널, 해상도 2배
        if x.shape[2:] != enc_feat3.shape[2:]:
            enc_feat3 = F.interpolate(enc_feat3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat3], dim=1)
        x = self.dec_blocks[0](x)

        # 2단계 업샘플 + skip 연결 + 합성곱
        x = self.upconvs[1](x)  # 256 -> 128채널, 해상도 2배
        if x.shape[2:] != enc_feat2.shape[2:]:
            enc_feat2 = F.interpolate(enc_feat2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat2], dim=1)
        x = self.dec_blocks[1](x)

        # 3단계 업샘플 + skip 연결 + 합성곱
        x = self.upconvs[2](x)  # 128 -> 64채널, 해상도 2배
        if x.shape[2:] != enc_feat1.shape[2:]:
            enc_feat1 = F.interpolate(enc_feat1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, enc_feat1], dim=1)
        x = self.dec_blocks[2](x)

        # 4단계 업샘플 + 합성곱
        x = self.upconvs[3](x)  # 64 -> 32채널, 해상도 2배
        x = self.dec_blocks[3](x)

        # 최종 출력
        x = self.final_conv(x)
        mask = self.activation(x)
        return mask


# 사용 예시:
# encoder_outputs = [enc_feat1 (1/2), enc_feat2 (1/4), enc_feat3 (1/8), bottleneck_feature (1/16)]
# decoder = UNetDecoder(in_channels=512)
# mask_pred = decoder(bottleneck_feature, enc_feat3, enc_feat2, enc_feat1)
