import torch
from torch.cuda.amp import autocast, GradScaler  # ✅ 추가
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import os

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import TextEncoder, ImageEncoder, GroundingModel

# 출력 디렉토리 생성 (없으면)
os.makedirs("outputs", exist_ok=True)

# config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# dataset
train_set = VizWizGroundingDataset(
    json_path=config["dataset"]["train_json"],
    image_root=config["dataset"]["train_image_root"],
    mask_root=config["dataset"]["train_mask_root"],
    image_size=tuple(config["image_size"])
)
val_set = VizWizGroundingDataset(
    json_path=config["dataset"]["val_json"],
    image_root=config["dataset"]["val_image_root"],
    mask_root=config["dataset"]["val_mask_root"],
    image_size=tuple(config["image_size"])
)
train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],  # ✅ 24 → 8~12 정도로 줄이자
    pin_memory=True,
    prefetch_factor=2  # 4 → 2로 줄이면 부담 덜함
)
val_loader = DataLoader(
    val_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],  # ✅ 24 → 8~12 정도로 줄이자
    pin_memory=True,
    prefetch_factor=2  # 4 → 2로 줄이면 부담 덜함
)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)

# optimizer / loss
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# resume checkpoint
resume_path = config.get("resume_checkpoint", None)
start_epoch = 0


#10 Epoch마다 저장장
    # start_epoch = int(resume_path.split("epoch")[1].split(".")[0])  # 파일명에서 에폭 추출하는 방식 (선택)
    # 그 다음 range(start_epoch, config["num_epochs"])로 바꿔서 학습 재시작 가능하게 만들기
    # config 파일 예시: resume_checkpoint: outputs/checkpoint_epoch10.pt 등등

if resume_path and os.path.exists(resume_path):
    model.load_state_dict(torch.load(resume_path))
    print(f"✅ Resumed model from {resume_path}")
    # 자동으로 epoch 번호 추정
    try:
        start_epoch = int(resume_path.split("epoch")[1].split(".")[0])
    except Exception:
        start_epoch = 0  # 실패 시 0부터 시작

# training loop
for epoch in range(start_epoch, config["num_epochs"]):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Training)")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        pred_masks = model(images, texts)
        pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')

        loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        with autocast():
            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')
            loss = loss_fn(pred_masks, masks)

        scaler.scale(loss).backward()  # ✅ AMP 대응
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Average Training Loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Validation)")
    with torch.no_grad():
        for batch in val_loop:
            batch = to_device(batch, device)
            images = batch["image"]
            masks = batch["mask"]
            texts = batch["text"]

            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')

            loss = loss_fn(pred_masks, masks)
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"[Epoch {epoch+1}] Average Validation Loss: {avg_val_loss:.4f}")

    # 10 epoch마다 체크포인트 저장
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"outputs/checkpoint_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")



# 최종save
torch.save(model.state_dict(), f"outputs/model_final_epoch_0501_{config['num_epochs']}.pt")

print(f" Final model saved")







'''
# GPU 병렬 사용
torch.save(model.module.state_dict(), "outputs/clip-vit-large-patch14-336_epoch100.pt")
'''