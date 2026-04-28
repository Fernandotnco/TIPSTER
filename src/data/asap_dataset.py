import h5py
from torch.utils.data import Dataset
import torch
import json
import random
import numpy as np

from . import audio_processing as ap
from .AudioAugmentation import AudioAugmentation

from tqdm import tqdm


VALID_INSTRUMENTS = ["piano", "vibraphone", "xylophone", "harp"]


class ASAPDataset(Dataset):
    """PyTorch Dataset for loading ASAP data fully into RAM."""

    def __init__(
        self,
        split,
        sr=22050,
        base_sr=22050,
        normalize_audio=True,
        data_config_file=None,
        device="cpu",
        no_augments_p=0.04,
        trim_audio_to=None,
        valid_instruments=None,
        return_labels=False,
    ):
        assert split in ["train", "test", "sampled"]

        self.store_configurations(
            split=split,
            sr=sr,
            base_sr=base_sr,
            normalize_audio=normalize_audio,
            no_augments_p=no_augments_p,
            valid_instruments=valid_instruments,
            return_labels=return_labels,
        )

        self.device = device
        self.trim_audio_to = trim_audio_to

        self.aug = AudioAugmentation(
            config_file=data_config_file,
            sample_rate=sr,
            device=device,
        )

        self.load_data_split(split)
        self.load_all_data_into_ram()

    def store_configurations(
        self,
        split,
        sr,
        base_sr,
        normalize_audio,
        no_augments_p,
        valid_instruments,
        return_labels,
    ):
        self.split = split
        self.sr = sr
        self.base_sr = base_sr
        self.normalize_audio = normalize_audio
        self.no_augments_p = no_augments_p
        self.return_labels = return_labels

        if valid_instruments is None:
            valid_instruments = VALID_INSTRUMENTS

        self.valid_instruments = valid_instruments
        self.instrument_to_label = {
            instrument: idx for idx, instrument in enumerate(self.valid_instruments)
        }

        self.program_to_indices = {}
        for i, n in enumerate(valid_instruments):
            self.program_to_indices[n]=i

    def load_data_split(self, split):
        """Load sample names for the requested split."""
        file_path = "/storage/datasets/fernando.tonucci/asap_pvxh22_splits.json"

        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)

        all_samples = [
            sample
            for piece_samples in loaded_dict[self.split].values()
            for sample in piece_samples
        ]

        self.samples = []
        for sample in tqdm(all_samples):
            instrument = sample.split("_")[0]
            if instrument in self.instrument_to_label:
                self.samples.append(sample)

    def load_all_data_into_ram(self):
        """Load all selected audio arrays into RAM and store labels."""
        h5_path = "/storage/datasets/fernando.tonucci/asap_pvxh22.h5"

        self.audio_data = []
        self.labels = []

        print("Loading dataset")

        with h5py.File(h5_path, "r") as h5_file:
            dataset = h5_file["asap"]

            for sample_name in self.samples:
                instrument = sample_name.split("_")[0]
                label = self.instrument_to_label[instrument]

                # Force full load into RAM
                audio = dataset[sample_name][:]

                self.audio_data.append(audio)
                self.labels.append(label)

    def process_audio(self, audio):
        """Resample, normalize, and augment audio."""
        audio = ap.resample_audio(audio, self.base_sr, self.sr)

        if self.normalize_audio:
            audio = ap.normalize_audio_loudness(audio)

        if random.random() > self.no_augments_p:
            audio, _ = self.aug(audio.astype("float32"))

        return audio

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio = self.audio_data[index]
        label = self.labels[index]

        audio = self.process_audio(audio)

        if self.trim_audio_to is not None:
            target_len = int(self.trim_audio_to * self.sr)
            orig_len = len(audio)

            if orig_len > target_len:
                start = np.random.randint(0, orig_len - target_len + 1)
                audio = audio[start:start + target_len]
            elif orig_len < target_len:
                pad_amount = target_len - orig_len
                audio = np.pad(audio, (0, pad_amount), mode="constant")

        audio = torch.from_numpy(audio).float()

        if self.return_labels:
            return audio, label

        return audio