from torch.utils.data import Dataset
from PIL import Image
import os, json
import torchvision.transforms as T
import torch
import random

class VizWizGroundingDataset(Dataset):
    def __init__(self, json_path, image_root, mask_root=None, image_size=(224, 224), is_test=False):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.mask_root = mask_root
        self.is_test = is_test
        self.image_size = image_size
        self.entries = list(self.data.items())

        # train/val 구분해서 transform 설정
        if not self.is_test:
            self.image_tf = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
            self.mask_tf = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
        else:
            self.image_tf = T.Compose([
                T.Resize(image_size),
                T.ToTensor()
            ])
            self.mask_tf = T.Compose([
                T.Resize(image_size),
                T.ToTensor()
            ])

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

        # training일 때만 rotation 적용
        if not self.is_test:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = image.rotate(angle)
                mask = mask.rotate(angle)

        image = self.image_tf(image)
        mask = self.mask_tf(mask)

        return {"image": image, "text": text, "mask": mask, "filename": filename}
