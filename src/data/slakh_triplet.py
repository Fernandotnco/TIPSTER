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


def program_to_category(name):

    '''
    Groups program names into appropriate instrument classes to avoid choosing negative pairs of similar instruments.
    '''

    # --- Pianos ---
    if "piano" in name or 'harpsichord' in name or 'clavinet' in name:
        return "pianos"

    if "organ" in name:
        return "organs"

    # --- Synths ---
    if (
        name.startswith("lead ")
        or name.startswith("pad ")
        or "synth" in name
    ):
        return "synths"

    # --- Guitars (including bass) ---
    if (
        "guitar" in name
        or "bass" in name
    ):
        if 'bassoon' not in name and 'contrabass' not in name:
            return "guitars"
        

    # --- Strings ---
    if name in {
        "violin", "viola", "cello", "pizzicato strings",
        "string ensemble 1", "string ensemble 2",
        "tremolo strings", "orchestral harp", "contrabass"
    }:
        return "strings"

    # --- Brass ---
    if name in {
        "trumpet", "muted trumpet", "trombone",
        "tuba", "french horn", "brass section"
    }:
        return "brass"

    # --- Woodwinds ---
    if (
        "sax" in name
        or name in {
            "bassoon", "clarinet", "english horn", "flute",
            "harmonica", "oboe", "ocarina", "pan flute",
            "piccolo", "recorder", "shakuhachi", "whistle",
            "blown bottle"
        }
    ):
        return "woodwinds"

    # --- Tuned percussion ---
    if name in {
        "celesta", "dulcimer", "glockenspiel", "marimba",
        "music box", "timpani", "tubular bells",
        "vibraphone", "xylophone"
    }:
        return "tuned percussions"

    # --- Misc ---
    return "misc"

class SlakhTripletDataset(Dataset):
    def __init__(
        self,
        split,
        sr=22050,
        ram_sr=16000,
        base_sr=22050,
        normalize_audio=True,
        device="cpu",
        h5_path=None,
        dtype_in_ram=np.float16,
        audio_len = 5,
        trim_audio_to = 2.5,
        return_labels = False,
        weak_negatives = False,
    ):
        self.store_configurations(
            sr=sr,
            ram_sr=ram_sr,
            base_sr=base_sr,
            normalize_audio=normalize_audio,
            audio_len=audio_len,
            trim_audio_to=trim_audio_to,
            return_labels=return_labels,
            weak_negatives=weak_negatives
        )

        self.device = device
        self.dtype_in_ram = dtype_in_ram

        self.h5_path = (
            h5_path
            if h5_path is not None
            else f"/storage/datasets/fernando.tonucci/slakh_{split}_chunks_5s.h5"
        )


        print("Loading metadata and caching data in RAM...")

        # RAM storage
        self.audio_ram = None
        self.program_names = None
        self.program_to_indices = defaultdict(list)
        self.unique_programs = None

        self._load_into_ram()

        self.size = len(self.program_names)

        with open("/storage/datasets/fernando.tonucci/slakh_labels.json", "r", encoding="utf-8") as f:
            self.program_to_labels = json.load(f)

    def store_configurations(
        self,
        sr,
        ram_sr,
        base_sr,
        normalize_audio,
        audio_len,
        trim_audio_to,
        return_labels,
        weak_negatives
    ):
        self.sr = sr                  # final output sr for training
        self.ram_sr = ram_sr          # cached sr in RAM
        self.base_sr = base_sr        # original sr in H5
        self.normalize_audio = normalize_audio
        self.audio_len=audio_len
        self.return_labels=return_labels
        self.trim_audio_to=trim_audio_to
        if self.trim_audio_to is not None:
            self.target_len = int(self.trim_audio_to * self.sr)
        self.weak_negatives=weak_negatives

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

            for i in tqdm(range(size)):
                audio = np.asarray(audio_ds[i]).astype(np.float32, copy=False)
                program_name = self._decode_program_name(program_ds[i])

                # Cache audio as 16k fp16
                audio_16k = ap.resample_audio(audio, self.base_sr, self.ram_sr)
                self.audio_ram[i] = audio_16k.astype(self.dtype_in_ram, copy=False)

                self.program_names.append(program_name)
                self.program_to_indices[program_name].append(i)


        self.unique_programs = list(self.program_to_indices.keys())

        
        self.program_weights = [math.sqrt(len(self.program_to_indices[p]))for p in self.unique_programs]

        if len(self.unique_programs) < 2:
            raise ValueError("Dataset needs at least 2 different instruments for triplet sampling.")

        # Sanity warning
        singletons = [k for k, v in self.program_to_indices.items() if len(v) < 2]
        if singletons:
            print(
                f"[Warning] {len(singletons)} instruments have only 1 sample. "
                "For those, B may equal A."
            )

        self.instrument_groups = {}
        for name in self.unique_programs:
            self.instrument_groups[name] = program_to_category(name)

    def __len__(self):
        return len(self.program_names)

    def _sample_positive_index(self, anchor_idx, program_name):
        candidates = self.program_to_indices[program_name]

        if len(candidates) == 1:
            return anchor_idx

        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = random.choice(candidates)
        return pos_idx

    def _sample_negative_index(self, anchor_program_name):
        anchor_group = self.instrument_groups[anchor_program_name]
        while True:
            negative_program = random.choices(
                self.unique_programs,
                weights=self.program_weights,
                k=1
            )[0]
            negative_group = self.instrument_groups[negative_program]

            if self.weak_negatives:
                if anchor_group != negative_group:
                    break
            else:
                if anchor_program_name != negative_program:
                    break

        neg_idx = random.choice(self.program_to_indices[negative_program])
        return neg_idx

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

    def __getitem__(self, idx):
        anchor_program = self.program_names[idx]

        idx_a = idx
        idx_b = self._sample_positive_index(idx_a, anchor_program)
        idx_c = self._sample_negative_index(anchor_program)

        A = self._get_audio_from_ram(idx_a)
        B = self._get_audio_from_ram(idx_b)
        C = self._get_audio_from_ram(idx_c)

        trim=True
        while trim:
            A, _ = self.trim_audio(A)
            B, _ = self.trim_audio(B)
            C, _ = self.trim_audio(C)
            trim = self.is_silent(A) and  self.is_silent(B) and self.is_silent(C)

        A_emb = B_emb = C_emb = torch.zeros(1)

        pos_idx = torch.tensor(0, dtype=torch.long).unsqueeze(0)

        if self.return_labels:
            labels = {"instrument": torch.tensor(self.program_to_labels[anchor_program], dtype=torch.long)}
            return A, B, C, A_emb, B_emb, C_emb, pos_idx, labels

        return A, B, C, A_emb, B_emb, C_emb, pos_idx
