import json
from audiomentations import *
from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy as np
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import torchaudio
import random
import time
import os

import torch
import torchaudio
import torchaudio.functional as F
import os
import random

class Reverb:
    def __init__(self, ir_dir="/mnt/users/fernando.tonucci/contrastive-learning/Impulse Responses/", 
                 sample_rate=48000, device="cpu", apply_prob=0.5):
        """
        Reverb augmentation using impulse response (IR) files.

        Args:
            ir_dir (str): Directory containing IR files.
            sample_rate (int): Sample rate of the audio.
            device (str): Device to run the augmentation ('cpu' or 'cuda').
            apply_prob (float): Probability of applying reverb.
        """
        self.sample_rate = sample_rate
        self.device = device
        self.apply_prob = apply_prob

        # Load impulse response files
        self.ir_files = os.listdir(ir_dir)
        self.ir_dir = ir_dir

        # Preload IRs into memory and move them to the correct device
        self.ir_waveforms = []
        for ir_path in self.ir_files:
            ir_waveform, ir_sr = torchaudio.load(os.path.join(ir_dir, ir_path))
            if ir_sr != sample_rate:
                ir_waveform = torchaudio.transforms.Resample(ir_sr, sample_rate)(ir_waveform)
            ir_waveform = ir_waveform / torch.max(torch.abs(ir_waveform))  # Normalize IR
            self.ir_waveforms.append(ir_waveform.to(self.device))

    def __call__(self, batch, trim_duration=5.0):
        """
        Apply reverb to a batch of audio samples with a probability.

        Args:
            batch (torch.Tensor): Shape (batch_size, num_channels, num_samples).
            trim_duration (float): Duration to trim the output to (in seconds).

        Returns:
            torch.Tensor: Reverb-augmented batch.
            list: Indices of chosen IR files (-1 if no reverb applied).
        """
        batch_size, num_channels, num_samples = batch.shape
        num_trim_samples = int(trim_duration * self.sample_rate)

        # Decide which samples get reverb
        apply_reverb_mask = torch.rand(batch_size) < self.apply_prob  # Boolean mask

        # Initialize output tensor (copy of batch)
        reverb_batch = batch.clone()
        chosen_indices = [-1] * batch_size  # Default to -1 (no reverb applied)

        if apply_reverb_mask.any():  # Only process if at least one sample needs reverb
            selected_ir_indices = torch.randint(0, len(self.ir_waveforms), (apply_reverb_mask.sum(),))

            # Stack selected IRs for batch processing
            selected_irs = torch.stack([self.ir_waveforms[i] for i in selected_ir_indices])

            # Apply convolution to the selected samples
            reverb_audio = F.convolve(batch[apply_reverb_mask], selected_irs)

            # Trim or pad the output
            reverb_audio = reverb_audio[:, :, :num_trim_samples]
            reverb_audio = torch.nn.functional.pad(reverb_audio, (0, max(0, num_trim_samples - reverb_audio.shape[-1])))

            # Insert processed audio into batch
            reverb_batch[apply_reverb_mask] = reverb_audio
            chosen_indices = [-1 if not apply else idx for apply, idx in zip(apply_reverb_mask.tolist(), selected_ir_indices.tolist())]

        return reverb_batch, chosen_indices


class MildCompressor(BaseWaveformTransform):
    def __init__(self, min_threshold_db=-30.0, max_threshold_db=-10.0, min_ratio=1.5, max_ratio=4.0, p=0.5):
        super().__init__(p)
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, samples, sample_rate):
        threshold_db = np.random.uniform(self.min_threshold_db, self.max_threshold_db)
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)

        eps = 1e-8
        amplitude = np.abs(samples)
        db_samples = 20 * np.log10(amplitude + eps)
        over_threshold = db_samples > threshold_db
        db_samples[over_threshold] = threshold_db + (db_samples[over_threshold] - threshold_db) / ratio
        linear = 10 ** (db_samples / 20.0)
        return (np.sign(samples) * linear).astype('float32')


class TapeSaturation(BaseWaveformTransform):
    def __init__(self, min_drive=1.0, max_drive=3.0, p=0.5):
        super().__init__(p)
        self.min_drive = min_drive
        self.max_drive = max_drive

    def apply(self, samples, sample_rate):
        drive = np.random.uniform(self.min_drive, self.max_drive)
        saturated = np.tanh(samples * drive)
        max_val = np.max(np.abs(saturated))
        if max_val > 0:
            saturated = saturated / max_val
        return saturated.astype('float32')


class AudioAugmentation:
    """Class for applying audio augmentations using predefined parameters or a configuration file."""

    def __init__(self, config_file=None, sample_rate = 48000, device = 'cpu'):
        """Initialize the AudioAugmentation class.

        Args:
            config_file (str, optional): Path to a JSON file containing custom augmentation parameters. Defaults to None.
        """
        # Default parameters by category
        self.params = {

            # Distortion augmentations
            "distortion": {
                "tanh_distortion": {
                    "min_distortion": 0.15,
                    "max_distortion": 0.5,
                    "p": 1.0
                },
                "mild_compressor": {
                    "min_threshold_db": -30.0,
                    "max_threshold_db": -10.0,
                    "min_ratio": 1.5,
                    "max_ratio": 4.0,
                    "p": 1.0
                },
                "tape_saturation": {
                    "min_drive": 1.0,
                    "max_drive": 3.0,
                    "p": 1.0
                },
                "one_of_p": 0.62
            },

            # Environmental augmentations
            "environmental": {
                "air_absorption": {
                    "min_temperature": 10,
                    "max_temperature": 20,
                    "min_humidity": 30,
                    "max_humidity": 90,
                    "min_distance": 10,
                    "max_distance": 100,
                    "p": 0.1
                },
                "one_of_p": 1
            },

            # Frequency and EQ augmentation
            "frequency_eq": {
                "seven_band_eq": {
                    "min_gain_db": -8.0,
                    "max_gain_db": 12.0,
                    "p": 0.1
                },
                "one_of_p": 1
            },

            # Gain and loudness augmentations
            "gain_loudness": {
                "gain": {
                    "min_gain_db": -8,
                    "max_gain_db": 12,
                    "p": 0.8
                },
                "gain_transition": {
                    "min_gain_db": -8,
                    "max_gain_db": 12,
                    "min_duration": 0.1,
                    "max_duration": 0.9,
                    'duration_unit': 'fraction',
                    "p": 1.0
                },
                "normalize": {
                    "p": 0.6
                },
                "one_of_p": 0.8
            },

            # Filter settings
            "filter": {
                "low_pass_filter": {
                    "min_cutoff_freq": 7000,
                    "max_cutoff_freq": 16000,
                    "p": 0.4
                },
                "one_of_p": 1
            },

            # Noise Settings
            "noise": {
                "add_gaussian_snr":{
                    "min_snr_db": 0.01,
                    "max_snr_db":0.015,
                    "p":0
                }
            }
        }

        # Override with JSON file if provided
        if config_file:
            self.load_config(config_file)

        # Instantiate the augmentation pipeline
        self.augmentations = Compose([
            OneOf([
                TanhDistortion(**self.params["distortion"]["tanh_distortion"]),
                MildCompressor(**self.params["distortion"]["mild_compressor"]),
                TapeSaturation(**self.params["distortion"]["tape_saturation"]),
            ], p=self.params["distortion"]["one_of_p"]),
            AirAbsorption(**self.params["environmental"]["air_absorption"]),
            SevenBandParametricEQ(**self.params["frequency_eq"]["seven_band_eq"]),
            OneOf([
                Gain(**self.params["gain_loudness"]["gain"]),
                GainTransition(**self.params["gain_loudness"]["gain_transition"]),
                Normalize(**self.params["gain_loudness"]["normalize"]),
            ], p=self.params["gain_loudness"]["one_of_p"]),
            LowPassFilter(**self.params["filter"]["low_pass_filter"])
            
        ])
        self.add_noise = AddGaussianSNR(**self.params["noise"]["add_gaussian_snr"])

        self.augmentation_names = [
            "TanhDistortion", "MildCompressor", "TapeSaturation",
            "AirAbsorption", "SevenBandParametricEQ",
            "LowPassFilter"
        ]
        self.augmentation_index = {name: i for i, name in enumerate(self.augmentation_names)}

        '''n_steps = np.random.uniform(
            self.params["pitch"]["pitch_shift"]["min_semitones"],
            self.params["pitch"]["pitch_shift"]["max_semitones"]
        )'''
        self.sample_rate = sample_rate

        '''self.pitch_shift_transform = T.PitchShift(
            sample_rate=sample_rate,
            n_steps=n_steps,
            bins_per_octave=12
        ).to(device)'''

    def load_config(self, config_file):
        """Load custom augmentation parameters from a JSON configuration file.

        Args:
            config_file (str): Path to the JSON file containing custom parameters.
        """
        with open(config_file, "r") as file:
            custom_params = json.load(file)
            for category, values in custom_params.items():
                if category in self.params:
                    self.params[category].update(values)

    '''@staticmethod
    def room_simulator_with_fixed_duration(samples, sample_rate):
        """Apply room simulation to audio samples while maintaining original duration.

        Args:
            samples (np.ndarray): Audio samples.
            sample_rate (int): Sampling rate of the audio samples.

        Returns:
            np.ndarray: Augmented audio samples with original duration.
        """
        original_num_samples = len(samples)
        room_sim_transform = RoomSimulator(
            min_size_x=3.0, max_size_x=10.0,
            min_size_y=3.0, max_size_y=10.0,
            min_size_z=2.0, max_size_z=4.0,
            min_absorption_value=0.1, max_absorption_value=0.5,
            p=1.0
        )
        samples = room_sim_transform(samples, sample_rate)
        return samples[:original_num_samples]'''
    
    '''def pitch_shift(self, audio):
        """
        Apply pitch shift to an audio tensor using Torchaudio.

        Args:
            audio (torch.Tensor or np.ndarray): The audio tensor or array to augment. Shape: (num_channels, num_samples).
            sample_rate (int): The sample rate of the audio.

        Returns:
            np.ndarray: Pitch-shifted audio samples.
        """
        # Ensure the input is a torch tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        # Randomly choose pitch shift steps within the range
        

        # Apply the pre-initialized pitch shift transform
        shifted_audio = self.pitch_shift_transform(audio.unsqueeze(0))  # Add batch dimension
        return shifted_audio.squeeze()'''

    def __call__(self, samples):
        """Apply the augmentation pipeline to the audio samples and return a one-hot encoded vector.

        Args:
            samples (np.ndarray): Audio samples to augment.
            sample_rate (int): Sampling rate of the audio samples.

        Returns:
            tuple: (augmented_samples, one_hot_vector)
                - augmented_samples (np.ndarray): The augmented audio samples.
                - one_hot_vector (np.ndarray): One-hot encoded vector of applied augmentations.
        """
        # Initialize a zeroed one-hot vector
        one_hot_vector = np.zeros(len(self.augmentation_names), dtype=np.float32)

        # Define a wrapper function to record which augmentation was applied
        def record_augmentation(augmentation_name, transform, **kwargs):
            nonlocal samples
            if transform is not None:  # Apply the transformation if it exists
                samples = transform(samples=samples, sample_rate=self.sample_rate, **kwargs)
                if augmentation_name in self.augmentation_index:
                    one_hot_vector[self.augmentation_index[augmentation_name]] = 1

        # Iterate through the augmentations pipeline
        for step in self.augmentations.transforms:
            if isinstance(step, OneOf):

                #Manually calculating OneOf p so we can return which was chosen.
                if random.random() < step.p:
                    chosen = step.transforms[torch.multinomial(torch.tensor([t.p for t in step.transforms]), num_samples=1).item()]
                    record_augmentation(chosen.__class__.__name__, chosen)
            else:
                #Manually calculating p so we can return which was chosen.
                if random.random() < step.p:
                    record_augmentation(step.__class__.__name__, step)

        samples = self.add_noise(samples, sample_rate=self.sample_rate)
        

        return samples, one_hot_vector