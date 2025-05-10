import torch
from torch.cuda.amp import autocast, GradScaler  # ✅ 추가
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import TextEncoder, ImageEncoder, GroundingModel

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

# training loop
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Training)")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        with autocast():
            pred_masks = model(images, texts)
            pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')
            loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        scaler.scale(loss).backward()

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
        checkpoint_path = f"outputs/cross_checkpoint_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint saved at {checkpoint_path}")



# 최종save
torch.save(model.state_dict(), f"outputs/cross_model_final_epoch{config['num_epochs']}.pt")


'''
# GPU 병렬 사용
torch.save(model.module.state_dict(), "outputs/clip-vit-large-patch14-336_epoch100.pt")
'''