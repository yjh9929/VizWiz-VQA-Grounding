import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size

    def forward(self, text_list):
        tokens = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        return outputs.pooler_output  # (B, D)