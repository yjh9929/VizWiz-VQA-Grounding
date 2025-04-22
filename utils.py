import torch

def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_iou(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * gt).sum()
    union = ((pred_bin + gt) >= 1).float().sum()
    return (intersection / union).item() if union > 0 else 0.0