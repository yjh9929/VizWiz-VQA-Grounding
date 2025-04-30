import torch.nn as nn
from transformers import CLIPModel

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super(ImageEncoder, self).__init__()
        clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_encoder = clip_model.vision_model
        self.out_channels = self.vision_encoder.config.hidden_size  # 768

    def forward(self, x):
        outputs = self.vision_encoder(x)
        last_hidden_state = outputs.last_hidden_state  # (B, seq_len, hidden_dim)

        # (B, seq_len, D) -> (B, D, H, W)로 변환
        batch_size, seq_len, hidden_dim = last_hidden_state.shape
        patch_size = int((seq_len - 1) ** 0.5)  # cls token 제외
        feature = last_hidden_state[:, 1:, :].transpose(1, 2)  # (B, D, seq_len)
        feature = feature.view(batch_size, hidden_dim, patch_size, patch_size)  # (B, D, H, W)
        return feature
