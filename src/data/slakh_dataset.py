from . import audio_processing as ap

import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import math

import json


class SlakhDataset(Dataset):
    def __init__(self, split, sr=22050, ram_sr=16000, base_sr=22050, normalize_audio=True, device="cpu", h5_path=None,  label_map_path=None, dtype_in_ram=np.float16, audio_len = 5, trim_audio_to = 2.5):

        self.store_configurations( sr=sr, ram_sr=ram_sr, base_sr=base_sr, normalize_audio=normalize_audio, audio_len=audio_len, trim_audio_to=trim_audio_to, )

        self.device = device
        self.dtype_in_ram = dtype_in_ram

        self.h5_path = (h5_path if h5_path is not None else f"/storage/datasets/fernando.tonucci/slakh_{split}_chunks_5s.h5")
        self.label_map_path = (label_map_path if label_map_path is not None else f"/storage/datasets/fernando.tonucci/slakh_labels.json")

        print("Loading metadata and caching data in RAM...")

        # RAM storage
        self.audio_ram = None
        self.program_names = None
        self.program_to_indices = {}

        self._load_into_ram()


    def store_configurations(self, sr, ram_sr, base_sr, normalize_audio, audio_len, trim_audio_to):

        self.sr = sr                  # final output sr for training
        self.ram_sr = ram_sr          # cached sr in RAM
        self.base_sr = base_sr        # original sr in H5
        self.normalize_audio = normalize_audio
        self.audio_len=audio_len
        self.trim_audio_to=trim_audio_to
        if self.trim_audio_to is not None:
            self.target_len = int(self.trim_audio_to * self.sr)

    def _decode_program_name(self, x):
        if isinstance(x, bytes):
            return x.decode("utf-8")
        if isinstance(x, np.bytes_):
            return x.decode("utf-8")
        return str(x).strip().lower()

    def _load_into_ram(self):
        with h5py.File(self.h5_path, "r") as f:
            audio_ds = f["audio"]
            program_ds = f["program_name"]

            size = len(audio_ds)
            target_len = int(self.audio_len*self.ram_sr)
            self.audio_ram = np.empty((size, target_len), dtype=self.dtype_in_ram)
            self.program_names = []
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                self.program_to_indices = json.load(f)

            l = 0

            for i in tqdm(range(size)):
                audio = np.asarray(audio_ds[i]).astype(np.float32, copy=False)
                program_name = self._decode_program_name(program_ds[i])

                # Cache audio as 16k fp16
                audio_16k = ap.resample_audio(audio, self.base_sr, self.ram_sr)
                self.audio_ram[i] = audio_16k.astype(self.dtype_in_ram, copy=False)

                self.program_names.append(program_name)


    def __len__(self):
        return len(self.program_names)

    def process_audio(self, audio):
        """
        audio arrives here as a numpy array stored in RAM at 16k fp16.
        We convert it back to float32 and resample to training sr.
        """
        audio = audio.astype(np.float32, copy=False)

        if self.ram_sr != self.sr:
            audio = ap.resample_audio(audio, self.ram_sr, self.sr)

        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)


        return audio
    
    def trim_audio(self, audio, start = None):
        if self.trim_audio_to is not None:           
            if len(audio) > self.target_len:
                if start is None:
                    start = np.random.randint(0, len(audio) - self.target_len)
                audio = audio[start : start + self.target_len]
        return audio, start
    
    def is_silent(self, x, threshold=1e-4):
        return x.abs().max() < threshold

    def _get_audio_from_ram(self, idx):
        audio = self.audio_ram[idx]
        audio = self.process_audio(audio)
        return torch.from_numpy(audio).float()
    
    def encode_label(self, program_name):
        return self.program_to_indices[program_name]

    def __getitem__(self, idx):

        program_name = self.program_names[idx]
        label = self.encode_label(program_name)
        A = self._get_audio_from_ram(idx)

        trim=True
        while trim:
            A, _ = self.trim_audio(A)
            trim = self.is_silent(A)

        return A, label

