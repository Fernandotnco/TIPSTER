import sys
import os
sys.path.append(os.path.abspath("/mnt/users/fernando.tonucci/conde/synesis"))

from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchaudio


from . import audio_processing as ap

def load_track(
    path: Union[str, Path],
    item_format: str,
    itemization: bool,
    sample_rate: int,
    item_len_sec=None,
):
    """Load an audio track (or features for it) from a file.

    Args:
        path: The path to the audio file.
        item_format: Format of the items to return: ["raw", "feature"].
        itemization: For datasets with variable-length items, whether to return them
                  as a list of equal-length items (True) or as a single item.
        item_len_sec: The length of the items in seconds.
        sample_rate: The sample rate to resample the audio to.
    """
    
    # Problem after downloading dataset, swapping " for _ on the file name, which was causing an error.
    # If your dataset is exactly the original, remove these lines below.
    # ====================================
    def normalize_path(path: str) -> str:
        return path.replace('"', '_')
    path = normalize_path(path)
    # ====================================
    
    waveform, original_sample_rate = torchaudio.load(path, normalize=True)
    if waveform.size(0) != 1:  # make mono if stereo (or more)
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=sample_rate,
        )
        waveform = resampler(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if itemization:
        if item_len_sec is None:
            # Return the entire waveform if item_len_sec is not provided
            return waveform
        # we need to split the track into fixed-length segments of
        # item_len_sec and return them as a list
        item_len_samples = int(item_len_sec * sample_rate)
        waveform_len = waveform.size(1)
        num_items = waveform_len // item_len_samples
        remainder_item = waveform_len % item_len_samples

        # pad with zeros so that we can split without remainder
        waveform = torch.cat(
            [waveform, torch.zeros(1, item_len_samples - remainder_item)],
            dim=1,
        )

        # split into subitems of size item_len_samples
        waveform = waveform.view(num_items + 1, 1, item_len_samples)

    return waveform



class ChordsDataset(Dataset):
    def __init__(
        self,
        split: str = "train",               # "train" or "test"
        test_size: float = 0.10,
        seed: int = 42,
        root: Union[str, Path] = "/storage/datasets/raphael.mendes/final_syntheory",
        base_sr: int = 22050,
        sr: int = 22050,
        item_format: str = "raw",
        normalize_audio: bool = True,
        trim_audio_to: Optional[float] = None,
        cache_raw_audio_fp16: bool = True,
    ):
        self.root = Path(root)
        self.dataset_dir = self.root / "chords"

        self.base_sr = base_sr
        self.sr = sr
        self.item_format = item_format
        self.normalize_audio = normalize_audio
        self.trim_audio_to = trim_audio_to
        self.cache_raw_audio_fp16 = cache_raw_audio_fp16

        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")

        full_df = pd.read_csv(self.dataset_dir / "info.csv").reset_index(drop=True)

        required_cols = [
            "synth_file_path",
            "chord_type",
            "midi_program_num",
        ]
        for col in required_cols:
            if col not in full_df.columns:
                raise ValueError(f"Missing column: {col}")

        train_idx, test_idx = train_test_split(
            np.arange(len(full_df)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

        selected_idx = train_idx if split == "train" else test_idx
        self.df = full_df.iloc[selected_idx].reset_index(drop=True)

        self.paths = [
            str(self.dataset_dir / p)
            for p in self.df["synth_file_path"]
        ]

        self.instrument_labels = torch.tensor(
            self.df["midi_program_num"].values,
            dtype=torch.long,
        )

        self._preload_data_to_ram()

        self.program_to_indices = [0]*128

    def _load_from_disk(self, path):
        return load_track(
            path=path,
            item_format=self.item_format,
            itemization=False,
            sample_rate=self.base_sr,
        )

    def _cache_item(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    
        audio = audio.squeeze(0)  # [1, N] -> [N]
    
        if self.cache_raw_audio_fp16:
            audio = audio.astype(np.float16)
    
        return audio

        

        return audio

    def _restore_cached_item(self, audio):
        return audio.astype(np.float32)

    def _preload_data_to_ram(self):
        def load_and_cache(path):
            return self._cache_item(self._load_from_disk(path))

        with ThreadPoolExecutor() as executor:
            self.ram_cache = list(
                tqdm(
                    executor.map(load_and_cache, self.paths),
                    total=len(self.paths),
                    desc="Preloading chords to RAM",
                )
            )

    def process_audio(self, audio):
        audio = ap.resample_audio(audio, self.base_sr, self.sr)

        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)

        if self.trim_audio_to is not None:
            target_len = int(self.trim_audio_to * self.sr)
            orig_len = len(audio)

            if orig_len > target_len:
                start = np.random.randint(0, orig_len - target_len + 1)
                audio = audio[start:start + target_len]

            elif orig_len < target_len:
                pad_amount = target_len - orig_len
                audio = np.pad(audio, (0, pad_amount), mode="constant")

        return audio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio = self._restore_cached_item(self.ram_cache[idx])
        audio = self.process_audio(audio)

        audio = torch.from_numpy(audio).float()

        return audio, self.instrument_labels[idx]