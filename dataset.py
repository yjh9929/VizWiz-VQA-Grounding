from torch.utils.data import Dataset
from PIL import Image
import os, json
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import random

class VizWizGroundingDataset(Dataset):
    def __init__(self, json_path, image_root, mask_root=None, image_size=(336, 336), is_test=False):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.mask_root = mask_root
        self.is_test = is_test
        self.image_size = image_size
        self.entries = list(self.data.items())

        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, meta = self.entries[idx]

        image = Image.open(os.path.join(self.image_root, filename)).convert("RGB")
        question = meta["question"]
        answer = meta.get("most_common_answer", "")
        text = f"Q: {question} A: {answer}" if answer else f"Q: {question}"

        if self.mask_root:
            mask_path = os.path.join(self.mask_root, filename.replace(".jpg", ".png"))
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", self.image_size)

        # Resize 이미지/마스크 (동일하게)
        image = self.resize(image)
        mask = self.resize(mask)

        # 동일한 증강 적용 (rotation + flip)
        if not self.is_test:
            # 회전
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = image.rotate(angle)
                mask = mask.rotate(angle)

            # 좌우 반전
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # 텐서 변환
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()  # Binary mask 처리

        return {
            "image": image,
            "text": text,
            "mask": mask,
            "filename": filename
        }