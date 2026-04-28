import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


def resample_audio(audio, base_sr, target_sr):
    if target_sr != base_sr:
        audio = librosa.resample(audio, orig_sr=base_sr, target_sr=target_sr)
    return audio



def normalize_audio_loudness(y, target_dB=-25):
    rms = np.sqrt(np.mean(y**2))
    scalar = 10**(target_dB / 20) / (rms + 1e-6)
    return y * scalar

