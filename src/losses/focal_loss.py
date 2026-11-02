"""Multiclass focal loss operating on raw logits."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, alpha: Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.long().view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("alpha must contain one weight per output class")
            loss = loss * self.alpha.to(logits.device)[targets.long()]
        return loss.mean()
