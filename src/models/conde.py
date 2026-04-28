import torch
import torch.nn as nn
import torch.nn.functional as F
from .affine_coupling_net import AffineCouplingNet


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.f(x)


class ResidualHead(nn.Module):
    def __init__(self, in_dim, out_dim, return_gate=False):
        super().__init__()
        # Typically out_dim == in_dim for clean decomposition
        self.z_norm = nn.LayerNorm(in_dim)
        self.f = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )
        self.return_gate = return_gate

    def forward(self, z):
        gate = self.f(self.z_norm(z))
        if self.return_gate:
            return gate*z, gate
        return gate*z
    

class PreMLP(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, hidden_dim=None):
        super().__init__()
        assert depth >= 2, "depth must be >= 1"
        if hidden_dim is None:
            hidden_dim=out_dim

        layers = []


        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())


        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class ConDe(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_semantic_heads,
        head_class=ProjectionHead,
        residual_class=ResidualHead,
        invertible_num_layers = 0,
        invertible_hidden_dims = None,
        pre_mlp_depth = 0,
        pre_mlp_hidden_dims=None,
        decoder_amounts = 0,
        decoder_class = None,
        lora_decoder = False,
        encoder=None,
        normalize=False,
        train_encoder=True,
        invert_residual=False,
    ):
        """
        Args:
            latent_dim: dim of Z
            n_semantic_heads: number of semantic heads
            head_dim: dimension of semantic heads (default = latent_dim)
            residual_dim: dimension of residual vector R (default = latent_dim)
            head_class: class used to build semantic heads
            residual_class: class used to build residual head
        """
        super().__init__()

        self.encoder=encoder
        self.train_encoder=train_encoder

        # dimensions
        self.latent_dim = latent_dim

        self.residual_head = residual_class(latent_dim, latent_dim, return_gate = True)

        if n_semantic_heads > 0:
            self.semantic_heads = nn.ModuleList([
                head_class(self.latent_dim, latent_dim)
                for _ in range(n_semantic_heads)
            ])
        else:
            self.semantic_heads=None

        self.invertible=False
        if invertible_num_layers > 0:
            self.invertible=True
            
            if invertible_hidden_dims is None:
                invertible_hidden_dims =  self.latent_dim//2

            self.invertible_net = AffineCouplingNet(embed_dim = self.latent_dim, hidden_dim= invertible_hidden_dims,  num_layers = invertible_num_layers)

        self.pre = False
        if pre_mlp_depth > 0:
            self.pre=True
            
            if pre_mlp_hidden_dims is None:
                pre_mlp_hidden_dims =  self.latent_dim//2

            self.pre_mlp = PreMLP(self.latent_dim, latent_dim, pre_mlp_depth, pre_mlp_hidden_dims)
        
        self.normalize=normalize

    def forward(self, Z):
        """
        Args:
            Z: latent embedding
        Z -> {R, Z_sem, S_list, S_sum}
        """

        if self.encoder is not None:
            
            Z = self.encoder(Z)
            if not self.train_encoder:
                Z=Z.detach()
        if self.normalize:
            mean = Z.mean(dim=0, keepdim=True)
            std = Z.std(dim=0, keepdim=True) + 1e-8
            Z = (Z - mean) / std

        if self.pre:
            Z = self.pre_mlp(Z)

        if self.invertible:
            Z = self.invertible_net(Z)
        
        R, gate = self.residual_head(Z)   # [B, residual_dim]
        Z_sem = Z - R 
        if self.semantic_heads is not None:
            S_list = [head(Z_sem) for head in self.semantic_heads]  # each [B, head_dim]
        else:
            S_list = [Z_sem]
        R_detach, gate_detach = self.residual_head(Z.detach())
        

        ret_dict = {
            "R": R,
            "R_detach": R_detach,
            "Z": Z,
            "Z_sem": Z_sem,
            "S_list": S_list,
            "R_gate": gate
        }

        return ret_dict

    def reconstruct(self, Z=None, R=None, S_list=None):

        if Z is not None:
            if self.invertible:
                Z = self.invertible_net.inverse(Z)
            return Z

        assert R is not None and S_list is not None
        
        Z_hat = R + sum(S_list)
        if self.invertible:
            Z_hat = self.invertible_net.inverse(Z_hat)
        return Z_hat

class InvertedConDe(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_semantic_heads,
        head_class=ResidualHead,
        residual_class=ProjectionHead,
        encoder=None,
        train_encoder=True,
    ):
        """
        Args:
            latent_dim: dim of Z
            n_semantic_heads: number of semantic heads
            head_dim: dimension of semantic heads (default = latent_dim)
            residual_dim: dimension of residual vector R (default = latent_dim)
            head_class: class used to build semantic heads
            residual_class: class used to build residual head
        """
        super().__init__()

        self.encoder=encoder
        self.train_encoder=train_encoder

        # dimensions
        self.latent_dim = latent_dim

        self.residual_head = residual_class(latent_dim, latent_dim)

        if n_semantic_heads > 0:
            self.semantic_heads = nn.ModuleList([
                head_class(self.latent_dim, latent_dim)
                for _ in range(n_semantic_heads)
            ])
        else:
            raise "There must be at least one semantic head."


    def forward(self, Z):
        """
        Args:
            Z: latent embedding
        Z -> {R, Z_sem, S_list, S_sum}
        """

        if self.encoder is not None:
            
            Z = self.encoder(Z)
            if not self.train_encoder:
                Z=Z.detach()
        

        S_list = [head(Z) for head in self.semantic_heads]  # each [B, head_dim]

        Z_non_sem = Z - sum(S_list) 
        R = self.residual_head(Z_non_sem)

        R_detach = self.residual_head(Z_non_sem.detach())
        

        ret_dict = {
            "R": R,
            "R_detach": R_detach,
            "Z": Z,
            "Z_sem": sum(S_list),
            "Z_non_sem": Z_non_sem,
            "S_list": S_list,
            "R_gate": R
        }

        return ret_dict

    def reconstruct(self, Z=None, R=None, S_list=None):

        if Z is not None:
            return Z

        assert R is not None and S_list is not None
        
        Z_hat = R + sum(S_list)

        return Z_hat