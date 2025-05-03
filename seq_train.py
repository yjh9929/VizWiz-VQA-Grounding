import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import GroundingModel

# === Config 로딩 ===
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# === Dataset 설정 ===
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
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                          num_workers=config["num_workers"], pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], pin_memory=True, prefetch_factor=2)

# === Model + Optimizer 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# === 이어서 학습을 위한 설정 ===
start_epoch = 100
total_epochs = 300

# 체크포인트 불러오기
checkpoint_path = f"outputs/baic_checkpoint_epoch{start_epoch}.pt"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded model checkpoint from {checkpoint_path}")
else:
    print(f"❌ Checkpoint {checkpoint_path} not found. Starting from scratch.")
    start_epoch = 0

# === Training Loop ===
for epoch in range(start_epoch, total_epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Training)")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')
            loss = loss_fn(pred_masks, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Avg Training Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} (Validation)")
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
    print(f"[Epoch {epoch+1}] Avg Validation Loss: {avg_val_loss:.4f}")

    # === Checkpoint 저장 ===
    if (epoch + 1) % 10 == 0:
        ckpt_path = f"outputs/baic_checkpoint_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Checkpoint saved at {ckpt_path}")

# === 최종 저장 ===
torch.save(model.state_dict(), f"outputs/baic_model_final_epoch{total_epochs}.pt")
print("✅ Final model saved.")