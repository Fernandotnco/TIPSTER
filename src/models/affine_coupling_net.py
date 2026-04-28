import torch
import torch.nn as nn

class FixedPermutation(nn.Module):
    def __init__(self, dim, perm=None):
        super().__init__()
        if perm is None:
            perm = torch.randperm(dim)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv", inv)

    def forward(self, x):
        return x[:, self.perm]

    def inverse(self, y):
        return y[:, self.inv]


class AffineCouplingLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, mask, scale_clamp=1.0, use_gate=True):
        super().__init__()
        self.register_buffer("mask", mask)          # shape (D,)
        self.scale_clamp = float(scale_clamp)

        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        self.register_buffer("idx", idx)
        d_in = idx.numel()

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (embed_dim - d_in)),
        )

        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))  # exact identity at init

        # init last layer small (not zero) for nicer optimization once gate opens
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)

    def _st(self, x_active):
        out = self.net(x_active)
        if self.use_gate:
            out = out * self.gate
        s, t = out.chunk(2, dim=-1)

        # safer scale: keep it closer to 1
        s = torch.tanh(s) * self.scale_clamp
        return s, t

    def forward(self, x):
        # split into active and passive parts
        x_active = x[:, self.idx]                     # conditioner input
        s, t = self._st(x_active)

        y = x.clone()
        passive = (1 - self.mask).bool()
        y[:, passive] = x[:, passive] * torch.exp(s) + t
        return y

    def inverse(self, y):
        y_active = y[:, self.idx]
        s, t = self._st(y_active)

        x = y.clone()
        passive = (1 - self.mask).bool()
        x[:, passive] = (y[:, passive] - t) * torch.exp(-s)
        return x


class AffineCouplingNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, permute=True):
        super().__init__()
        assert embed_dim % 2 == 0

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if permute:
                self.blocks.append(FixedPermutation(embed_dim))

            if i % 2 == 0:
                mask = torch.cat([torch.ones(embed_dim // 2), torch.zeros(embed_dim // 2)])
            else:
                mask = torch.cat([torch.zeros(embed_dim // 2), torch.ones(embed_dim // 2)])

            self.blocks.append(
                AffineCouplingLayer(embed_dim, hidden_dim, mask=mask, scale_clamp=1.0, use_gate=True)
            )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def inverse(self, y):
        for b in reversed(self.blocks):
            y = b.inverse(y)
        return y
