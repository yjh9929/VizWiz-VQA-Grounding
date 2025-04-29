from models.model import GroundingModel
from metrics import compute_iou
from torchvision.transforms import ToTensor
from PIL import Image
import torch, json, os

from torchvision import transforms as T

# 경로 설정
val_json = "data/vizwiz/val_grounding.json"
image_dir = "data/vizwiz/val"
mask_dir = "data/vizwiz/binary_masks_png/val"

# 1. 모델 로드
model = GroundingModel()
model.load_state_dict(torch.load("outputs/clip-vit-large-patch14-336_epoch100.pt"))
model.eval().cuda()

# 2. val json 불러오기
with open(val_json, "r") as f:
    val_data = json.load(f)

# 3. IoU 계산
ious = []

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

for filename, meta in val_data.items():
    image_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png"))
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        continue

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
    true_mask = transform(Image.open(mask_path).convert("L")).unsqueeze(0).cuda()
    text = f"Q: {meta['question']} A: {meta.get('most_common_answer', '')}"

    with torch.no_grad():
        pred_mask = model(image, text)

    iou = compute_iou(pred_mask, true_mask)
    ious.append(iou)    

# 4. 평균 IoU 출력
print(f"Mean IoU over {len(ious)} samples: {sum(ious)/len(ious):.4f}")
