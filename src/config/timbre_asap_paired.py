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
        "load_method": "conde"
    },

    "training": {
        "epochs": 81,
        "decoder_start": 20,
        "validate_every": 10,
        "save_every": 10,
    },

    "paths": {
        "ckpt": "/mnt/users/fernando.tonucci/res/paired_mask_train2_continue/last_ckpt.pt",
        "save_dir": "../res/paired_mask_train2_continue2",
    },

    "semantic_info": {
        "instrument": {"head_index": 0},
        "song": {"head_index": 1},
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
            "n_semantic_heads": 2,
            "invertible_num_layers": 0,
            "pre_mlp_depth": 0,
            "normalize": False,
        },
    },

    "data": {
        "train_dataset_cls": data.ASAPTripletDataset,
        "train_dataset_kwargs": {
            "split": "train",
            "emb_model": None,
            "pair_performances": True,
            "trim_audio_to": 2.5,
        },
        "val_dataset_cls": data.ASAPTripletDataset,
        "val_dataset_kwargs": {
            "split": "test",
            "emb_model": None,
            "pair_performances": True,
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

    "losses": {
        "contrastive_loss_fn": losses.SmoothTripletLoss,
        "contrastive_loss_kwargs": {},

        "residual_loss_fn": losses.CombinedLoss,
        "residual_loss_kwargs": {
            "losses": [losses.HSIC_RBF_Multi(), losses.CrossCorrelationLoss()]
        },

        "coverage_loss_fn": nn.MSELoss,
        "coverage_loss_kwargs": {},

        "decomp_loss_fn": losses.CombinedLoss,
        "decomp_loss_kwargs": {
            "losses": [nn.MSELoss(), nn.L1Loss()]
        },

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

            "lambda_decomp": 5,
            "lambda_contrast": 3,
            "lambda_coverage": 1,
            "lambda_residual": 2,
            "lambda_residual_loss": 0.1,
            "lambda_recon": 0.2,

            "grad_clip_norm": 3,
            "coverage_mode": "inverted_detach",
            "decomp_mode": "inverted_detach",
            "require_labels": False,
            "mode": "audio",
            "use_amp": False,
        },
    },
}