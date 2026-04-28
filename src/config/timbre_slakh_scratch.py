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
        "epochs": 201,
        "decoder_start": 0,
        "validate_every": 5,
        "save_every": 1,
    },

    "paths": {
        "save_dir": "../res/slakh_scratch",
    },

    "semantic_info": {
        "instrument": {"head_index": 0}
    },

    "ae": {
        "backbone_cls": models.EncoderDecoder,
        "backbone_kwargs": {
            "in_channels": 2,
            "out_channels": 2,
            "embed_dim": 2048,
            "channels_list": [32, 64, 128, 256, 512],
            "res_blocks": [3, 3, 17, 5],
            "input_shape": [513, 224],
            "dropout_p": 0,
            "padding_on_stem": (0, 1),
            "norm_on_stem": False,
            "stem_kernel_size": 3,
            "stride_on_stem": 2,
            "final_kernel_size": 3,
            "norm": models.GNWrapper,
        },
        "spec_cls": data.STFTWrapper,
        "spec_kwargs": {
            "device": "cuda",
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
        },
        "ae_cls": models.MusicAE,
        "ae_kwargs": {
            "loss_aggregator": None,
            "padded_spec_width": 224,
            "device": "cpu",
        },
        "freeze_ae": True,
    },

    "conde": {
        "cls": models.ConDe,
        "kwargs": {
            "latent_dim": 2048,
            "n_semantic_heads": 1,
            "invertible_num_layers": 0,
            "pre_mlp_depth": 0,
            "normalize": False,
        },
    },

    "data": {
        "train_dataset_cls": data.SlakhTripletDataset,
        "train_dataset_kwargs": {
            "split": "train",

        },
        "val_dataset_cls": data.SlakhTripletDataset,
        "val_dataset_kwargs": {
            "split": "test",

        },
        "train_loader_kwargs": {
            "batch_size": 50,
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

    "losses": {
        "contrastive_loss_fn": losses.SmoothTripletLoss,
        "contrastive_loss_kwargs": {'margin': 0.35},

        "residual_loss_fn": losses.CombinedLoss,
        "residual_loss_kwargs": {
            "losses": [losses.HSIC_RBF_Multi(), losses.CrossCorrelationLoss()]
        },

        "coverage_loss_fn": losses.NormMSE,
        "coverage_loss_kwargs": {},

        "decomp_loss_fn": nn.MSELoss,
        "decomp_loss_kwargs": {},

        "recon_loss_fn": losses.MultiScaleSpectrogramLoss,
        "recon_loss_kwargs": {},

        "residual_reg_fn": losses.GateNormLoss,
        "residual_reg_kwargs": {
            "lambda_bin": 0,
            "lambda_sparse": 1,
        },

    },


    "trainer": {
        "cls": conde_trainer.ConDeTrainer,
        "kwargs": {
            "encode": True,
            "train_encoder": True,
            "train_decoder": True,
            "use_upgrad": False,

            "lambda_decomp": 3,
            "lambda_contrast": 1,
            "lambda_coverage": 2,
            "lambda_residual": 5,
            "lambda_residual_loss": 0,
            "lambda_recon": 0.15,

            "grad_clip_norm": 3,
            "coverage_mode": "inverted_detach",
            "decomp_mode": "inverted_detach",
            "require_labels": False,
            "mode": "audio",
            "use_amp": False,
        },
    },
}