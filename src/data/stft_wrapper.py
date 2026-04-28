import torch
import torchaudio
import matplotlib.pyplot as plt


class STFTWrapper:
    def __init__(self,
                 n_fft=1024,
                 hop_length=512,
                 win_length=1024,
                 window_fn=torch.hann_window,
                 device='cpu'):

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window_fn(win_length).to(device)
        self.min_db=-80

    def normalize_complex_spectrogram(self, spec, eps=1e-8):
        '''real = spec[:, 0]
        imag = spec[:, 1]

        mag   = torch.sqrt(real**2 + imag**2 + eps)         # >= 0
        phase = torch.atan2(imag, real)                     # (-pi, pi]


        mag_db = 20 * torch.log10(mag + 1e-8)
        mag_db = torch.clamp(mag_db, self.min_db, 0.0)

        mag_norm = (mag_db - self.min_db) / (-self.min_db)

        # Rebuild complex with mag_norm as radius
        real_new = mag_norm * torch.cos(phase)
        imag_new = mag_norm * torch.sin(phase)

        return torch.stack([real_new, imag_new], dim=1)'''
        return spec


    def denormalize_complex_spectrogram(self, spec, eps=1e-8):

        ''' real = spec[:, 0]
        imag = spec[:, 1]

        mag_norm = torch.sqrt(real**2 + imag**2 + eps)      # ideally [0,1]
        phase    = torch.atan2(imag, real)


        mag_db = mag_norm * (-self.min_db) + self.min_db  

        # Back to linear magnitude
        mag = 10 ** (mag_db / 20.0)

        real_orig = mag * torch.cos(phase)
        imag_orig = mag * torch.sin(phase)

        return torch.stack([real_orig, imag_orig], dim=1)'''
        return spec

    def __call__(self, wav: torch.Tensor):
        """
        wav: (channels, samples)
        Returns: (channels, 2, freq, time)
                 second dim = [magnitude, phase]
        """
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )  # shape: (channels, freq, time)

        spec_real = spec.real        # float32
        spec_imag = spec.imag        # float32
        spec  = torch.stack([spec_real, spec_imag], dim=1)


        return self.normalize_complex_spectrogram(spec)

    def inverse(self, spec: torch.Tensor, length=None):
        """
        spec_mag_phase: (channels, 2, freq, time)
        Returns waveform: (channels, samples)
        """
        spec = self.denormalize_complex_spectrogram(spec)
        real = spec[:, 0]
        imag = spec[:, 1]

        complex_spec = torch.complex(real, imag)

        wav = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )
        return wav


