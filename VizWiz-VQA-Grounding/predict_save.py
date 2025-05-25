import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
from models import GroundingModel

# ===== 사용자 설정 =====
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
checkpoint_path = "outputs/clip-vit-L-p14-336-wo-crop_epoch3.pt"
save_path = "result/clip-vit-L-p14-336-wo-crop_epoch3/VizWiz_test_00000006_pred_mask.png"
image_size = (336, 336)
device = torch.device("cpu")
# ===== 모델 로드 =====
model = GroundingModel().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ===== 이미지 전처리 =====
transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor()
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
text_input = [f"Q: {question}"]

# ===== 추론 =====
with torch.no_grad():
    output = model(input_tensor, text_input)  # raw logits
    output = torch.sigmoid(output)            # apply sigmoid
    output = torch.nn.functional.interpolate(output, size=image_size, mode="bilinear", align_corners=False)
    binary_mask = (output > 0.5).float()[0, 0]  # binary (H, W)

# ===== 저장 =====
os.makedirs(os.path.dirname(save_path), exist_ok=True)
mask_img = TF.to_pil_image(binary_mask)  # convert tensor to PIL image
mask_img.save(save_path)
print(f"✅ Saved binary mask to: {save_path}")