import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMNormPatcher(nn.Module):
    """
    Wraps a decoder and injects FiLM conditioning after its normalization layers via hooks.
    Supports BatchNorm(1/2/3)d, GroupNorm, LayerNorm.
    Identity init: gamma=1, beta=0 for all experts.
    """

    def __init__(
        self,
        decoder: nn.Module,
        n_experts: int,
        apply_to=(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm),
    ):
        super().__init__()
        assert decoder is not None
        assert n_experts > 0

        self.decoder = decoder
        self.n_experts = n_experts
        self.apply_to = apply_to

        self._film_tables = nn.ModuleDict()
        self._handles = []
        self._current_expert_id = None

        self._install_hooks()

    def _norm_channels(self, m: nn.Module):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return int(m.num_features)
        if isinstance(m, nn.GroupNorm):
            return int(m.num_channels)
        if isinstance(m, nn.LayerNorm):
            ns = m.normalized_shape
            if isinstance(ns, int):
                return int(ns)
            if isinstance(ns, (tuple, list)):
                return int(ns[-1])
        return None

    def _make_table(self, C: int):
        tab = nn.Embedding(self.n_experts, 2 * C)
        nn.init.zeros_(tab.weight)  # => gamma_delta=0, beta=0
        return tab

    def _install_hooks(self):
        for name, m in self.decoder.named_modules():
            if not isinstance(m, self.apply_to):
                continue

            C = self._norm_channels(m)
            if C is None or C <= 0:
                continue

            key = name.replace(".", "__")
            if key in self._film_tables:
                continue

            self._film_tables[key] = self._make_table(C)

            def _hook_fn(mod, inp, out, _key=key, _C=C):
                inst_id = self._current_expert_id
                if inst_id is None:
                    return out

                gb = self._film_tables[_key](inst_id)  # [B, 2C]
                gamma, beta = gb[:, :_C], gb[:, _C:]
                gamma = 1.0 + gamma  # identity init

                # channel-first: [B, C, ...]
                if out.dim() >= 2 and out.shape[1] == _C:
                    shape = [out.shape[0], _C] + [1] * (out.dim() - 2)
                    return out * gamma.view(*shape) + beta.view(*shape)

                # channel-last: [B, ..., C]
                if out.dim() >= 2 and out.shape[-1] == _C:
                    shape = [out.shape[0]] + [1] * (out.dim() - 2) + [_C]
                    return out * gamma.view(*shape) + beta.view(*shape)

                return out

            self._handles.append(m.register_forward_hook(_hook_fn))

    def forward(self, z_hat, inst_id: torch.Tensor = None, **decoder_kwargs):
        """
        z_hat: input expected by decoder
        inst_id: [B] long selecting the FiLM expert
        """
        if inst_id is not None:
            self._current_expert_id = inst_id
        try:
            return self.decoder(z_hat, **decoder_kwargs)
        finally:
            self._current_expert_id = None


class DecoderFiLM(nn.Module):
    """
    High-level wrapper:
      - holds frozen decoder
      - applies FiLM via norm hooks
    """
    def __init__(self, decoder: nn.Module, n_experts: int):
        super().__init__()
        self.film = FiLMNormPatcher(decoder, n_experts=n_experts)

    def forward(self, z_hat, instrument_id: torch.Tensor, **decoder_kwargs):
        return self.film(z_hat, inst_id=instrument_id, **decoder_kwargs)