import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPModel
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", pretrained=True):
        super(ImageEncoder, self).__init__()
        if pretrained:
            clip_model = CLIPModel.from_pretrained(model_name)
        else:
            clip_model = CLIPModel.from_config(model_name)
        self.vision_encoder = clip_model.vision_model
        self.out_channels = self.vision_encoder.config.hidden_size

    def forward(self, x):
        outputs = self.vision_encoder(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # 예: 중간 레이어 3개 + 마지막 bottleneck
        enc_feat1 = hidden_states[4]  # (B, seq, D)
        enc_feat2 = hidden_states[7]
        enc_feat3 = hidden_states[9]
        bottleneck = outputs.last_hidden_state

        def reshape_feat(feat):
            feat = feat[:, 1:, :].transpose(1, 2)  # (B, D, seq_len)
            patch_size = int((feat.shape[-1]) ** 0.5)
            return feat.view(feat.shape[0], feat.shape[1], patch_size, patch_size)

        return tuple(map(reshape_feat, [enc_feat1, enc_feat2, enc_feat3])) + (reshape_feat(bottleneck),)
