import torch
import torch.nn as nn
import torch.nn.functional as F

class NormLoss(nn.Module):

    def __init__(self, eps=1e-4):
        super().__init__()

        self.eps = eps

    def forward(self, x, z):
        
        x = x.view(x.size(0), -1) 
        z = z.view(z.size(0), -1) 

        x_norm = x.norm(dim=-1)
        z_norm = z.norm(dim=-1)

        return (x_norm/(z_norm+self.eps)).mean()
    
class MaxNormLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, r, z):

        r = r.view(r.size(0), -1)
        z = z.view(z.size(0), -1)

        r_norm = r.norm(dim=-1)
        z_norm = z.norm(dim=-1)

        loss = F.relu(r_norm / (z_norm + self.eps) - 1) ** 2

        return loss.mean()
    
class GateNormLoss(nn.Module):
    def __init__(self, eps=1e-8, lambda_sparse=0.1, lambda_bin=1):
        super().__init__()
        self.eps = eps
        self.lambda_bin=lambda_bin
        self.lambda_sparse=lambda_sparse

    def forward(self, r, z=None):

        loss_sparse = r.mean() 
        loss_bin = (r * (1.0 - r)).mean()
        loss = (self.lambda_sparse * loss_sparse) + (self.lambda_bin * loss_bin)
        return loss
    

class NormMSE(nn.Module):

    def __init__(self, eps=1e-4):
        super().__init__()

        self.eps = eps

    def forward(self, x, z):
        
        x = x / (x.norm(dim=-1, keepdim=True) + self.eps)
        z = z / (z.norm(dim=-1, keepdim=True) + self.eps)

        return F.mse_loss(x, z)

