import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size  # 768

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # (B, L, D)
