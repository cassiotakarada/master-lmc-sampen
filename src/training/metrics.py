import torch
from torch import Tensor


def accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().sum().item()
    return correct / max(1, labels.numel())
