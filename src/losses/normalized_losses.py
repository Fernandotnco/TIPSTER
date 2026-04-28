import torch
import torch.nn as nn
import torch.nn.functional as F

class ZScoreMSE(nn.Module):
    """
    MSE loss where pred and label are each z-score normalized
    per sample before computing the error.

    Works on tensors shaped [B, D] or [..., D].
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _zscore(self, x):
        """
        Per-sample z-score normalization.
        x: tensor [..., D]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - mean) / std

    def forward(self, pred, target):
        # z-score each sample independently
        pred_norm = self._zscore(pred)
        target_norm = self._zscore(target)

        # standard MSE
        return F.mse_loss(pred_norm, target_norm)
