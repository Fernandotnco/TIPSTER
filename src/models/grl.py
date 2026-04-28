import torch
from torch import nn

class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_: float):
        ctx.lambda_ = float(lambda_)
        # forward is identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse + scale gradients
        return -ctx.lambda_ * grad_output, None


def grl(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GRLFn.apply(x, lambda_)


class GRL(nn.Module):
    """Gradient Reversal Layer."""
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grl(x, self.lambda_)