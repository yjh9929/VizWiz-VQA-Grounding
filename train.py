import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingModel().to(device)

# optimizer / loss
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.BCELoss()

# loss history
train_loss_history = []
val_loss_history = []

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

        pred_masks = model(images, texts)
        pred_masks = nn.functional.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear')

        loss = loss_fn(pred_masks, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)
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
    val_loss_history.append(avg_val_loss)
    print(f"[Epoch {epoch+1}] Average Validation Loss: {avg_val_loss:.4f}")

# save model
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/model_0430_01.pt")
print("✅ Model saved to: outputs/model_0428_01.pt")

# visualize loss curve
plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('outputs/loss_curve.png')
plt.close()
print("📈 Loss curve saved to outputs/loss_curve.png")
