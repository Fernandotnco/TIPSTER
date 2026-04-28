import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import models
import losses
import conde_trainer
import data



CONFIG = {
    "runtime": {
        "device": "cuda",
        "use_amp": False,
        "cudnn_benchmark": True,
        "detect_anomaly": False,
    },

    "training": {
        "epochs": 81,
        "validate_every": 10,
        "save_every": 10,
    },

    "paths": {
        "save_dir": "../res/slakh_classifier",
    },


    "model": {
        "backbone_cls": models.ClassifierNet,
        "backbone_kwargs": {
            "in_channels": 2,
            "out_channels": 2,
            "embed_dim": 512,
            "channels_list": [32, 64, 128, 256, 512],
            "res_blocks": [2, 2, 3, 2],
            "input_shape": [513, 224],
            "dropout_p": 0,
            "padding_on_stem": (0, 1),
            "norm_on_stem": False,
            "stem_kernel_size": 3,
            "stride_on_stem": 2,
            "num_classes": None,
            "norm": models.GNWrapper,
        },
        "spec_cls": data.STFTWrapper,
        "spec_kwargs": {
            "device": "cuda",
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
        },
        "model_cls": models.InstrumentClassifier,
        "model_kwargs": {
            "loss_aggregator": None,
            "padded_spec_width": 224,
            "device": "cpu",
        },
    },

    "conde": {
        "cls": models.ConDe,
        "kwargs": {
            "latent_dim": 2048,
            "n_semantic_heads": 2,
            "invertible_num_layers": 0,
            "pre_mlp_depth": 0,
            "normalize": False,
        },
    },

    "data": {
        "train_dataset_cls": data.SlakhDataset,
        "train_dataset_kwargs": {
            "split": "train",
            "trim_audio_to": 2.5,
        },
        "val_dataset_cls": data.SlakhDataset,
        "val_dataset_kwargs": {
            "split": "test",
            "trim_audio_to": 2.5,
        },
        "train_loader_kwargs": {
            "batch_size": 100,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
        },
        "val_loader_kwargs": {
            "batch_size": 32,
            "shuffle": True,
            "drop_last": True,
        },
    },


}