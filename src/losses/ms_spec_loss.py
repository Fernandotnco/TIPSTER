import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpectrogramLoss(nn.Module):
    def __init__(self, scales=None, eps=1e-5, device='cuda'):
        super().__init__()
        if scales is None:
            scales = [
                (1024, 256, 1024),
                (2048, 512, 2048),
                (512, 128, 512),
            ]
        self.scales = scales
        self.eps = eps
        self.window = torch.hann_window
        self.device = device

    def _get_two_channel_spec(self, spec):
        return torch.stack([spec.real, spec.imag], dim=1)

    def _safe_mag(self, spec):
        # safer than spec.abs() for backward near zero
        real = spec.real
        imag = spec.imag
        return torch.sqrt(real * real + imag * imag + self.eps)

    def forward(self, pred, target):
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        total_l1_loss = 0.0

        pred = pred.float()
        target = target.float()

        with torch.amp.autocast(device_type='cuda', enabled=False):
            if pred.shape[-1] > target.shape[-1]:
                pred = pred[:, :target.shape[-1]]
            elif pred.shape[-1] < target.shape[-1]:
                target = target[:, :pred.shape[-1]]

            for n_fft, hop, win in self.scales:
                window = self.window(win, device=pred.device)

                S_pred = torch.stft(
                    pred,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    window=window,
                    return_complex=True
                )

                S_tgt = torch.stft(
                    target,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    window=window,
                    return_complex=True
                )

                pred_spec = self._get_two_channel_spec(S_pred)
                tgt_spec = self._get_two_channel_spec(S_tgt)
                l1_loss = F.l1_loss(pred_spec, tgt_spec)

                S_pred_mag = self._safe_mag(S_pred)
                S_tgt_mag = self._safe_mag(S_tgt)

                diff = S_pred_mag - S_tgt_mag

                diff_flat = diff.flatten(1)
                tgt_flat = S_tgt_mag.flatten(1)

                sc_num = torch.sqrt((diff_flat * diff_flat).sum(dim=1) + self.eps)
                sc_den = torch.sqrt((tgt_flat * tgt_flat).sum(dim=1) + self.eps)
                sc_loss = (sc_num / sc_den).mean()

                log_pred = torch.log(S_pred_mag.clamp_min(self.eps))
                log_tgt = torch.log(S_tgt_mag.clamp_min(self.eps))
                mag_loss = F.l1_loss(log_pred, log_tgt)

                total_sc_loss += sc_loss
                total_mag_loss += mag_loss
                total_l1_loss += l1_loss

            total_sc_loss /= len(self.scales)
            total_mag_loss /= len(self.scales)
            total_l1_loss /= len(self.scales)

        return total_sc_loss + total_mag_loss + total_l1_loss
    



class SpecLoss(nn.Module):
    def __init__(
        self,
        mag_weight: float = 1.0,
        l1_weight: float = 1.0,
        eps: float = 1e-4,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mag_weight = mag_weight
        self.l1_weight = l1_weight
        self.eps = eps
        self.reduction = reduction

    def _safe_mag(self, spec):
        # safer than spec.abs() for backward near zero
        real = spec[:, 0]
        imag = spec[:, 1]
        return torch.sqrt(real.square() + imag.square()+ self.eps)

    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_spec:   (..., freq, time) or (B, C, F, T)
            target_spec: same shape as pred_spec

        Assumes these are already magnitude spectrograms (non-negative).
        """

        if pred_spec.shape != target_spec.shape:
            raise ValueError(
                f"Shape mismatch: pred_spec {pred_spec.shape} vs target_spec {target_spec.shape}"
            )

        l1_loss = F.l1_loss(pred_spec, target_spec)

        S_pred_mag = self._safe_mag(pred_spec)
        S_tgt_mag = self._safe_mag(target_spec)

        diff = S_pred_mag - S_tgt_mag

        diff_flat = diff.flatten(1)
        tgt_flat = S_tgt_mag.flatten(1)

        sc_num = torch.sqrt((diff_flat * diff_flat).sum(dim=1) + self.eps)
        sc_den = torch.sqrt((tgt_flat * tgt_flat).sum(dim=1) + self.eps)
        sc_loss = (sc_num / sc_den).mean()

        log_pred = torch.log(S_pred_mag.clamp_min(self.eps))
        log_tgt = torch.log(S_tgt_mag.clamp_min(self.eps))
        mag_loss = F.l1_loss(log_pred, log_tgt)

        return sc_loss + mag_loss*self.mag_weight + l1_loss * self.l1_weight