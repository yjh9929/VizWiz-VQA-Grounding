import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPModel
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super(ImageEncoder, self).__init__()
        clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_encoder = clip_model.vision_model
        self.out_channels = self.vision_encoder.config.hidden_size  # 1024 or 768

    def forward(self, pixel_values):
        # input shape: (B, 3, H, W)
        outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: len=13 (1 + 12 layers)

        # hidden_states[0] = patch embedding after projection
        # hidden_states[1]~[12] = transformer layers 1~12 output

        # 적절한 레이어에서 skip features 선택 (뒤에서부터 깊은 순서)
        enc_feat3 = self._to_feature_map(hidden_states[9])   # 1/8 scale
        enc_feat2 = self._to_feature_map(hidden_states[6])   # 1/4 scale
        enc_feat1 = self._to_feature_map(hidden_states[3])   # 1/2 scale
        bottleneck = self._to_feature_map(hidden_states[12]) # 1/16 scale

        return bottleneck, enc_feat3, enc_feat2, enc_feat1

    def _to_feature_map(self, hidden):
        # hidden: (B, seq_len, D) → (B, D, H, W)
        B, seq_len, D = hidden.shape
        spatial_tokens = hidden[:, 1:, :]  # remove CLS token
        H = W = int((seq_len - 1) ** 0.5)
        feature_map = spatial_tokens.transpose(1, 2).contiguous().view(B, D, H, W)
        return feature_map
       
