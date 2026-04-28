import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips

# ========== Basic loss building blocks ==========

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps**2))

def ssim_loss(x, y):
    # expects [0,1] range
    return 1 - ssim(x, y, data_range=1.0, size_average=True)

def gradient_loss(x, y):
    # simple Sobel kernels
    Sx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], 
                      dtype=torch.float32, device=x.device).view(1,1,3,3)
    Sy = Sx.transpose(-1,-2)
    gx = F.conv2d(x, Sx.expand(x.shape[1],1,3,3), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, Sy.expand(x.shape[1],1,3,3), padding=1, groups=x.shape[1])
    gxh = F.conv2d(y, Sx.expand(y.shape[1],1,3,3), padding=1, groups=y.shape[1])
    gyh = F.conv2d(y, Sy.expand(y.shape[1],1,3,3), padding=1, groups=y.shape[1])
    return (gx - gxh).abs().mean() + (gy - gyh).abs().mean()


# ========== Macro Loss ==========

class AELoss(nn.Module):
    def __init__(self, 
                 w_mse=0.2, w_l1=0.2, w_charb=0.2,
                 w_ssim=0.2, w_lpips=0.15, w_grad=0.05,
                 lpips_net='vgg'):
        super().__init__()
        self.w_mse   = w_mse
        self.w_l1    = w_l1
        self.w_charb = w_charb
        self.w_ssim  = w_ssim
        self.w_lpips = w_lpips
        self.w_grad  = w_grad

        self.mse   = nn.MSELoss()
        self.l1    = nn.L1Loss()
        self.charb = CharbonnierLoss()
        self.lpips = lpips.LPIPS(net=lpips_net).eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        losses = {}

        # Base pixel losses
        losses["mse"]   = self.mse(pred, target)
        losses["l1"]    = self.l1(pred, target)
        losses["charb"] = self.charb(pred, target)

        # Structural losses
        losses["ssim"]  = ssim_loss(pred, target)
        losses["grad"]  = gradient_loss(pred, target)

        # LPIPS (expects [-1,1])
        pred_n = pred*2 - 1
        tgt_n  = target*2 - 1
        losses["lpips"] = self.lpips(pred_n, tgt_n).mean()

        total = (
            self.w_mse   * losses["mse"] +
            self.w_l1    * losses["l1"] +
            self.w_charb * losses["charb"] +
            self.w_ssim  * losses["ssim"] +
            self.w_lpips * losses["lpips"] +
            self.w_grad  * losses["grad"]
        )
        return total, losses
