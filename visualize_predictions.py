import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os
from models import TextEncoder, ImageEncoder, GroundingModel

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)
model.load_state_dict(torch.load("outputs/cross_model_final_epoch100.pt", map_location=device))
model.eval()

# 테스트 데이터
image_path = "data/vizwiz/test/VizWiz_test_00000006.jpg"
question = "What does it say on here?"
text_input = [f"Q: {question}"]

# 전처리
transform = T.Compose([
    T.Resize((336, 336)),
    T.ToTensor()
])

image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# 예측
with torch.no_grad():
    mask = model(image, text_input)
    mask = torch.nn.functional.interpolate(mask, size=(336, 336), mode="bilinear")[0, 0].cpu()

# 시각화 및 저장
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path))
plt.title("Original Image")

# sigmoid + threshold
plt.subplot(1, 2, 2)
plt.imshow(Image.open(image_path))
plt.imshow(mask, alpha=0.5, cmap="jet", interpolation='bilinear')
plt.title("Predicted Mask")

# 저장
os.makedirs("result/cross_model_final_epoch100", exist_ok=True)
save_path = "result/cross_model_final_epoch100/VizWiz_test_00000006_masked.png" # 이거 바꿔가면서 실험해봐요
plt.savefig(save_path)
plt.close()

print(f"✅ Saved to: {save_path}")