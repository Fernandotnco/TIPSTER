import json
import random
from typing import List, Optional, Dict, Any, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import audio_processing as ap
from .AudioAugmentation import AudioAugmentation


class MultiH5AudioDataset(Dataset):
    """
    Multi-dataset wrapper over multiple H5 audio files, each with an 'audios' dataset/group.

    You pass:
      - h5_paths: list of .h5 files
      - split_json_paths: list of split jsons (same length as h5_paths), each containing:
            {"train": [...], "test": [...], "sampled": [...]}
        where each item is a key into that file's 'audios'.
    Returns:
      - torch.FloatTensor audio (T,)
    """

    def __init__(
        self,
        split: str,
        h5_paths: List[str],
        split_json_paths: List[str],
        sr: int = 22050,
        base_sr: int = 22050,
        normalize_audio: bool = True,
        data_config_file: Optional[str] = None,
        device: str = "cpu",
        no_augments_p: float = 0.04,
        trim_audio_to: Optional[float] = None,
    ):
        assert split in ["train", "test", "sampled"]
        assert len(h5_paths) == len(split_json_paths) and len(h5_paths) > 0

        self.split = split
        self.sr = sr
        self.base_sr = base_sr
        self.normalize_audio = normalize_audio
        self.no_augments_p = no_augments_p
        self.device = device
        self.trim_audio_to = trim_audio_to

        self.h5_paths = list(h5_paths)
        self.split_json_paths = list(split_json_paths)

        # Keep files lazily opened per worker/process (HDF5 is not fork-safe).
        self._h5_files: List[Optional[h5py.File]] = [None] * len(self.h5_paths)
        self._audios: List[Optional[Any]] = [None] * len(self.h5_paths)

        # Load split lists and build a global index: (dataset_id, key)
        self.entries: List[Tuple[int, str]] = []
        self.per_dataset_sizes: List[int] = []

        for di, sj in enumerate(self.split_json_paths):
            with open(sj, "r") as f:
                dd = json.load(f)
            keys = dd[split]
            self.per_dataset_sizes.append(len(keys))
            for k in keys:
                self.entries.append((di, k))

        self.aug = AudioAugmentation(config_file=data_config_file, sample_rate=sr, device=device)

    def __len__(self) -> int:
        return len(self.entries)

    def _ensure_open(self, dataset_id: int):
        """Open the H5 file for this worker/process if not opened yet."""
        if self._h5_files[dataset_id] is None:
            self._h5_files[dataset_id] = h5py.File(self.h5_paths[dataset_id], "r")
            self._audios[dataset_id] = self._h5_files[dataset_id]["audios"]

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = ap.resample_audio(audio, self.base_sr, self.sr)
        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)
        if random.random() > self.no_augments_p:
            audio, _ = self.aug(audio.astype("float32"))
        return audio

    def __getitem__(self, idx: int) -> torch.Tensor:
        dataset_id, key = self.entries[idx]
        self._ensure_open(dataset_id)

        audio_np = self._audios[dataset_id][key][:]
        audio_np = self.process_audio(audio_np)

        if self.trim_audio_to is not None:
            target_len = int(self.trim_audio_to * self.sr)
            orig_len = int(audio_np.shape[-1])

            if orig_len > target_len:
                start = np.random.randint(0, orig_len - target_len + 1)
                audio_np = audio_np[start : start + target_len]
            elif orig_len < target_len:
                # If you prefer strict trimming only, you can raise instead.
                pad = target_len - orig_len
                audio_np = np.pad(audio_np, (0, pad), mode="constant")

        return torch.from_numpy(audio_np).float()

    def __del__(self):
        # Best-effort cleanup
        for f in getattr(self, "_h5_files", []):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
