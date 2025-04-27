import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import os

from dataset import VizWizGroundingDataset
from utils import to_device, compute_iou
from models import TextEncoder, ImageEncoder, GroundingModel

os.makedirs("outputs", exist_ok=True)

# config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# dataset
train_set = VizWizGroundingDataset(
    json_path=config["dataset"]["train_json"],
    image_root=config["dataset"]["image_root"],
    mask_root=config["dataset"]["mask_root"],
    image_size=tuple(config["image_size"])
)
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)

'''
# GPU 병렬 사용
model = GroundingModel()
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model = model.to("cuda")
'''

# optimizer / loss
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCELoss()

# training loop
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    for batch in loop:
        batch = to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"]
        texts = batch["text"]

        pred_masks = model(images, texts)
        pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')

        loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f}")

    #10 epoch마다 checkpoint 저장
    if (epoch + 1) % 10 == 0:
        os.makedirs("outputs", exist_ok=True)
        checkpoint_path = f"outputs/checkpoint_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")






os.makedirs("outputs", exist_ok=True)
# save
torch.save(model.state_dict(), "outputs/model_haesol_1.pt")
'''
# GPU 병렬 사용
torch.save(model.module.state_dict(), "outputs/model.pt")
'''
