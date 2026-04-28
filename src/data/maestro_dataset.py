import h5py
from torch.utils.data import Dataset
import torch
import json
import random
import numpy as np

from . import audio_processing as ap
from .AudioAugmentation import AudioAugmentation

class MaestroDataset(Dataset):
    def __init__(self, split, sr=22050, base_sr=22050, normalize_audio=True,
                 data_config_file=None, device='cpu', no_augments_p=0.04, trim_audio_to=None):
        """
        A simplified dataset returning a single augmented audio tensor per sample.
        Args:
            split (str): One of ['train', 'test', 'sampled'].
            sr (int): Target sampling rate.
            base_sr (int): Original dataset sample rate.
            normalize_audio (bool): Apply loudness normalization.
            data_config_file (str): Path to JSON with augmentation config.
            device (str): 'cpu' or 'cuda'.
            no_augments_p (float): Probability of skipping augmentation.
        """
        assert split in ['train', 'test', 'sampled']
        self.split = split
        self.sr = sr
        self.base_sr = base_sr
        self.normalize_audio = normalize_audio
        self.no_augments_p = no_augments_p
        self.device = device

        self.h5_file = h5py.File('/storage/datasets/fernando.tonucci/maestro_processed.h5', 'r')
        self.dataset = self.h5_file['audios']

        with open('/storage/datasets/fernando.tonucci/maestro_splits.json', 'r') as f:
            self.data_dict = json.load(f)[split]
            
        self.trim_audio_to=trim_audio_to



        self.aug = AudioAugmentation(config_file=data_config_file, sample_rate=sr, device=device)

    def __len__(self):
        return len(self.data_dict)

    def process_audio(self, audio):
        audio = ap.resample_audio(audio, self.base_sr, self.sr)
        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)
        if random.random() > self.no_augments_p:
            audio, _ = self.aug(audio.astype('float32'))
        return audio

    def __getitem__(self, idx):
        perf_name = self.data_dict[idx]
        audio = self.process_audio(self.dataset[perf_name][:])
        if self.trim_audio_to:
          target_len = int(self.trim_audio_to * self.sr)
          orig_len = len(audio)
  

          start = np.random.randint(0, orig_len - target_len)
          audio = audio[start : start + target_len]  
                 
        return torch.from_numpy(audio).float()