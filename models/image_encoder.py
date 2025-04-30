import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        # ResNet18 백본 초기화 (미리 학습된 가중치 사용 가능)
        resnet = models.resnet18(pretrained=pretrained)
        # ResNet18의 초기 계층 (conv1, bn1, relu, maxpool)을 하나의 시퀀스로 묶음
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # ResNet18의 각 레이어 블록을 가져옴
        self.layer1 = resnet.layer1  # 출력: layer1 결과(feature map)
        self.layer2 = resnet.layer2  # 출력: layer2 결과(feature map)
        self.layer3 = resnet.layer3  # 출력: layer3 결과(feature map)
        self.layer4 = resnet.layer4  # 출력: layer4 결과(feature map, bottleneck)
        # avgpool과 fc 층은 사용하지 않으므로 생략

    def forward(self, x):
        # 입력 이미지에 대한 특징 추출
        x = self.initial(x)
        enc_feat1 = self.layer1(x)        # ResNet18 layer1 출력 (가장 얕은 특징 맵)
        enc_feat2 = self.layer2(enc_feat1)  # ResNet18 layer2 출력
        enc_feat3 = self.layer3(enc_feat2)  # ResNet18 layer3 출력
        bottleneck = self.layer4(enc_feat3) # ResNet18 layer4 출력 (최종 bottleneck 특징 맵)
        # 네 개의 특징 맵을 튜플로 반환
        return enc_feat1, enc_feat2, enc_feat3, bottleneck
