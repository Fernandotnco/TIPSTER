from . import audio_processing as ap

import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

class ASAPTripletDataset(Dataset):
    """
    Returns:
        A, B, C          : audio tensors
        A_emb, B_emb, C_emb
        pos_idx          : tensor([0])
        labels (optional): { "instrument": instrument_id }
    """

    VALID_INSTRUMENTS = ["piano", "vibraphone", "xylophone", "harp"]
    #VALID_INSTRUMENTS = ["piano", "xylophone"]

    def __init__(
        self,
        split,
        emb_model,
        sr=22050,
        base_sr=22050,
        normalize_audio=True,
        trim_audio_to=None,
        return_labels=False,
        device="cuda",
        batch_size=16,
        pair_performances = False
    ):
        assert split in ["train", "test", "sampled"]

        self.sr = sr
        self.base_sr = base_sr
        self.normalize_audio = normalize_audio
        self.trim_audio_to = trim_audio_to
        if self.trim_audio_to is not None:
            self.target_len = int(self.trim_audio_to * self.sr)
        self.return_labels = return_labels
        self.device = device

        self.pair_performances=pair_performances

        

        # --------------------------------------------------
        # Load ASAP
        # --------------------------------------------------

        print("[INFO] Loading dataset")
        self.h5_path = "/storage/datasets/fernando.tonucci/asap_pvxh22.h5"
        self.data={}

        with open("/storage/datasets/fernando.tonucci/asap_pvxh22_splits.json") as f:
            split_dict = json.load(f)[split]

        # Flatten samples
        self.samples = [
            perf for perfs in split_dict.values() for perf in perfs
        ]

        with h5py.File(self.h5_path, 'r') as h5f:
            grp = h5f['asap']
            for name in tqdm(self.samples):
                self.data[name] = grp[name][:].astype(np.float16)

        # --------------------------------------------------
        # Group by instrument
        # --------------------------------------------------
        self.by_instrument = {inst: [] for inst in self.VALID_INSTRUMENTS}

        for name in self.samples:
            inst = name.split("_")[0].lower()
            if inst in self.by_instrument:
                self.by_instrument[inst].append(name)

        # Remove instruments with <2 samples
        self.by_instrument = {
            k: v for k, v in self.by_instrument.items() if len(v) >= 2
        }

        self.instruments = list(self.by_instrument.keys())

        print("[INFO] Computing Embeddings")
        # --------------------------------------------------
        # Precompute embeddings (frozen)
        # --------------------------------------------------
        if emb_model is not None:
            self.embeddings = {}
            self.emb_model = emb_model.eval()

            with torch.no_grad():
                for inst, perfs in self.by_instrument.items():
                    print(f"Processing {inst} samples")
            
                    for i in tqdm(range(0, len(perfs), batch_size)):
                        batch_names = perfs[i : i + batch_size]
            
                        # Load audios
                        audios = [
                            self._load_audio(p) for p in batch_names
                        ]

                        audio_batch = torch.stack(audios).to(device)
            
                        # Encode
                        embs = self.emb_model.encode(audio_batch).cpu()
            
                        # Store
                        for name, emb in zip(batch_names, embs):
                            self.embeddings[name] = emb

        print(f"[INFO] finished loading dataset with {len(self)} samples")

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _load_audio(self, perf_name):
        audio = self.data[perf_name].astype(np.float32)
        audio = ap.resample_audio(audio, self.base_sr, self.sr)

        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)

        return torch.from_numpy(audio).float()
    
    def trim_audio(self, audio, start = None):
        if self.trim_audio_to is not None:           
            if len(audio) > self.target_len:
                if start is None:
                    start = np.random.randint(0, len(audio) - self.target_len)
                audio = audio[start : start + self.target_len]
        return audio, start

    # --------------------------------------------------
    # PyTorch API
    # --------------------------------------------------
    def __len__(self):
        return sum(len(v) for v in self.by_instrument.values())
    
    def is_silent(self, x, threshold=1e-4):
        return x.abs().max() < threshold

    def __getitem__(self, idx):
        # Anchor instrument
        inst = self.instruments[idx % len(self.instruments)]
        inst_id = self.VALID_INSTRUMENTS.index(inst)

        # Positive pair (same instrument)
        A_name, B_name = random.sample(self.by_instrument[inst], 2)

        # Negative instrument
        if not self.pair_performances:
            neg_inst = random.choice([i for i in self.instruments if i != inst])
            C_name = random.choice(self.by_instrument[neg_inst])
        else:
            neg_inst = random.choice([i for i in self.instruments if i != inst])
            anchor_name = '_'.join(A_name.split('_')[1:])
            C_name = neg_inst+'_'+anchor_name

        
        A = self._load_audio(A_name)
        B = self._load_audio(B_name)
        C = self._load_audio(C_name)

        trim=True
        while trim:
            A, start = self.trim_audio(A)
            B, _ = self.trim_audio(B)
            C_start = start if self.pair_performances else None
            C, _ = self.trim_audio(C, start=C_start)
            trim = self.is_silent(A) and  self.is_silent(B) and self.is_silent(C)

        

        #A = B = C = torch.zeros(1)

        # Embeddings
        '''A_emb = self.embeddings[A_name]
        B_emb = self.embeddings[B_name]
        C_emb = self.embeddings[C_name]'''
        A_emb = B_emb = C_emb = torch.zeros(1)


        if self.pair_performances:
            pos_idx = torch.tensor([0, 1], dtype=torch.long)
        else:
            pos_idx = torch.tensor(0, dtype=torch.long).unsqueeze(0)

        if self.return_labels:
            labels = {"instrument": torch.tensor(inst_id, dtype=torch.long), "song": torch.tensor(1, dtype=torch.long)}
            return A, B, C, A_emb, B_emb, C_emb, pos_idx, labels

        return A, B, C, A_emb, B_emb, C_emb, pos_idx
