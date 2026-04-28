import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import GRL


class ProbeProjectionHead(nn.Module):
    """
    Same interface/behavior as ConDe's ProjectionHead.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class ContrastiveProbe(nn.Module):
    """
    Dedicated probe heads (C_probe): given a latent Z (e.g., g(R)),
    returns a list of projections, one per semantic factor, matching ConDe's S_list format.

    Usage:
        C_probe = ContrastiveProbe(latent_dim, n_semantic_heads)
        probe_S_list = C_probe(probe_z_hat)  # -> List[Tensor], length = n_semantic_heads

    This is meant to replace:
        [head(probe_z_hat) for head in self.contrastive_probe]
    with:
        self.contrastive_probe(probe_z_hat)
    """
    def __init__(
        self,
        latent_dim: int,
        n_semantic_heads: int,
        head_class=ProbeProjectionHead,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_semantic_heads = n_semantic_heads
        self.out_dim = out_dim if out_dim is not None else latent_dim

        self.heads = nn.ModuleList([
            head_class(self.latent_dim, self.out_dim)
            for _ in range(n_semantic_heads)
        ])

    def forward(self, Z: torch.Tensor) -> list[torch.Tensor]:
        # Return a python list to match ConDe's S_list behavior exactly.
        return [h(Z) for h in self.heads]

    def __len__(self) -> int:
        return len(self.heads)

    def __getitem__(self, idx: int) -> nn.Module:
        # lets you still do contrastive_probe[k] if you want
        return self.heads[idx]
    
class ClassifierProbe(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dim = None):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.f(x)
    

class ProbeMLP(nn.Module):
    """
    MLP mapping from R -> Z_hat. Depth >= 2.
    """
    def __init__(self, latent_dim: int, depth: int = 4, hidden_dim: int | None = None):
        super().__init__()
        assert depth >= 2, "ProbeMLP depth must be >= 2"
        if hidden_dim is None:
            hidden_dim = latent_dim

        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, latent_dim)]
        self.f = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class ProbeNet(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        depth: int = 2,
        hidden_dim: int | None = None,
        use_layernorm: bool = True,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()

        self.grl_lambda = grl_lambda
        self.grl = GRL(lambda_=grl_lambda)

        self.g = ProbeMLP(latent_dim, depth=depth, hidden_dim=hidden_dim)



    def forward(self, R: torch.Tensor, use_grl: bool = True) -> torch.Tensor:
        x=R        
        if use_grl:
            x = self.grl(x)
        x=self.ln(R)
        return self.g(x)