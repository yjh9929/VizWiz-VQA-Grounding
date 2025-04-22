from models.model import GroundingModel
from dataset import VizWizGroundingDataset
from metrics import compute_iou
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os

# 1. 모델 로드
from models.model import GroundingModel
model = GroundingModel()
model.load_state_dict(torch.load("outputs/model.pt"))
model.eval().cuda()

# 2. 데이터 준비
image_path = "data/vizwiz/val/VizWiz_val_00000001.jpg"
mask_path = "data/vizwiz/binary_masks_png/val/VizWiz_val_00000001.png"
text = "Can you tell me what this medicine is please?"

image = ToTensor()(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
true_mask = ToTensor()(Image.open(mask_path).convert("L")).unsqueeze(0).cuda()

# 3. 예측
with torch.no_grad():
    pred_mask = model(image, text)

# 4. IoU 계산
iou = compute_iou(pred_mask, true_mask)
print(f"IoU: {iou:.4f}")