import torch
from torch import Tensor


def accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().sum()
    return float((correct / labels.numel()).item())

