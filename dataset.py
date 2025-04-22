from torch.utils.data import Dataset
from PIL import Image
import os, json
import torchvision.transforms as T
import torch

class VizWizGroundingDataset(Dataset):
    def __init__(self, json_path, image_root, mask_root=None, image_size=(224, 224), is_test=False):
        self.data = json.load(open(json_path))
        self.image_root = image_root
        self.mask_root = mask_root
        self.is_test = is_test
        self.image_size = image_size
        self.entries = list(self.data.items())

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
        image = self.image_tf(Image.open(os.path.join(self.image_root, filename)).convert("RGB"))
        question = meta["question"]
        answer = meta.get("most_common_answer", "")
        text = f"Q: {question} A: {answer}" if answer else f"Q: {question}"

        if not self.is_test and self.mask_root:
            mask_path = os.path.join(self.mask_root, filename.replace(".jpg", ".png"))
            mask = self.mask_tf(Image.open(mask_path).convert("L"))
        else:
            mask = torch.zeros(1, *self.image_size)

        return {"image": image, "text": text, "mask": mask, "filename": filename}