import torch
import torch.nn as nn

class RBFKernelBase(nn.Module):
    def __init__(self, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0), batch=None):
        """
        Base class providing:
            • automatic sigma selection
            • RBF multi-sigma kernel computation
        Child classes implement their own: __call__(x, y)
        """
        if batch is not None:
            self.sigmas = self._auto_sigmas(batch)
        else:
            self.sigmas = sigmas

    def _auto_sigmas(self, batch):
        """
        Median distance heuristic + geometric ladder, identical to your version.
        Only affects the sigma bandwidths; losses defined in child classes.
        """
        with torch.no_grad():
            x = batch
            if x.size(0) > 512:
                x = x[:512]

            diffs = x[:, None] - x[None, :]
            dist = (diffs.pow(2).sum(-1)).sqrt()
            m = dist.median().item()

        return [m/8, m/4, m/2, m, 2*m, 4*m]

    def rbf_kernel(self, x, y):
        """
        Multi-scale RBF kernel sum:  K = Σ exp(-||x-y||² / 2σ²)
        """
        x2 = (x**2).sum(1, keepdim=True)
        y2 = (y**2).sum(1, keepdim=True)
        dist = x2 - 2 * (x @ y.t()) + y2.t()

        K = 0.0
        for s in self.sigmas:
            K += torch.exp(-dist / (2 * s * s))
        return K



class MMD_RBF(RBFKernelBase):
    def __call__(self, x, y):
        """
        Biased MMD estimator:
            MMD = E[Kxx] + E[Kyy] - 2 E[Kxy]
        """
        Kxx = self.rbf_kernel(x, x)
        Kyy = self.rbf_kernel(y, y)
        Kxy = self.rbf_kernel(x, y)

        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()



class HSIC_RBF(RBFKernelBase):
    def __call__(self, x, y):
        n = x.size(0)

        # Gram matrices
        K = self.rbf_kernel(x, x)
        L = self.rbf_kernel(y, y)

        # Remove diagonals
        K.fill_diagonal_(0)
        L.fill_diagonal_(0)

        # Precompute sums
        K_sum = K.sum()
        L_sum = L.sum()

        K_row_sum = K.sum(dim=1)
        L_row_sum = L.sum(dim=1)

        # Term 1: sum_{i != j} K_ij L_ij
        term1 = (K * L).sum()

        # Term 2: sum_i sum_{j,k} K_ij L_ik
        term2 = (K_row_sum * L_row_sum).sum()

        # Unbiased HSIC
        hsic = (
            term1
            - 2 * term2 / (n - 2)
            + K_sum * L_sum / ((n - 1) * (n - 2))
        )

        return hsic / (n * (n - 3))

class HSIC_RBF_Multi(nn.Module):
    """
    Stable HSIC for training:
      - standardizes x,y per-dim
      - uses multi-sigma RBF mixture (median heuristic * ladder)
      - uses centered (biased) HSIC: tr(Kc Lc)/(n-1)^2
    """
    def __init__(
        self,
        ladder=(1/8, 1/4, 1/2, 1.0, 2.0, 4.0),
        max_sigma_samples=512,
        eps=1e-4,
        standardize=True,
        detach_sigma=True,
    ):
        super().__init__()
        self.ladder = ladder
        self.max_sigma_samples = max_sigma_samples
        self.eps = eps
        self.standardize = standardize
        self.detach_sigma = detach_sigma

    def _pairwise_sq_dists(self, x: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=1, keepdim=True)
        d2 = x2 + x2.t() - 2.0 * (x @ x.t())
        return d2.clamp_min(0.0)

    def _median_sigma(self, x: torch.Tensor) -> torch.Tensor:
        # median of pairwise distances (excluding diagonal)
        n = x.size(0)
        d2 = self._pairwise_sq_dists(x)
        mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
        med = d2[mask].median().clamp_min(self.eps)
        # match common convention: sigma^2 = 0.5 * median(d^2)
        return torch.sqrt(0.5 * med).clamp_min(self.eps)

    def _rbf_mix(self, x: torch.Tensor, sigma0: torch.Tensor) -> torch.Tensor:
        d2 = self._pairwise_sq_dists(x)
        K = 0.0
        for m in self.ladder:
            s = sigma0 * float(m)
            gamma = 1.0 / (2.0 * (s * s).clamp_min(self.eps))
            K = K + torch.exp(-gamma * d2)
        return K / len(self.ladder)

    def _center(self, K: torch.Tensor) -> torch.Tensor:
        n = K.size(0)
        one_n = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            if x.dim() == 1: x = x.unsqueeze(-1)
            if y.dim() == 1: y = y.unsqueeze(-1)
            n = x.size(0)
            if n < 4:
                return x.new_zeros(())

            if self.standardize:
                x = (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
                y = (y - y.mean(0, keepdim=True)) / y.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)

            # sigma estimation on subset (speed + stability)
            xs = x
            ys = y
            if n > self.max_sigma_samples:
                xs = x[:self.max_sigma_samples]
                ys = y[:self.max_sigma_samples]

            sigma_x = self._median_sigma(xs.detach() if self.detach_sigma else xs)
            sigma_y = self._median_sigma(ys.detach() if self.detach_sigma else ys)
            sigma0 = torch.sqrt(sigma_x * sigma_y).clamp_min(self.eps)

            K = self._rbf_mix(x, sigma0)
            L = self._rbf_mix(y, sigma0)

            Kc = self._center(K)
            Lc = self._center(L)

            # biased HSIC (stable)
            hsic = (Kc * Lc).sum() / ((n - 1) ** 2)
            return hsic