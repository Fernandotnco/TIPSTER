import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from losses import ExplainedVariance, MRRMetric, DecoderGap
from utils import MetricAggregator
import math
import os
from scipy.io import wavfile

import torchjd
from torchjd.aggregation import UPGrad


from IPython.display import Audio, display

import torch

def check_bad_gradients(model):
    """
    Call AFTER backward().

    If any gradient contains NaN or Inf, prints detailed debugging info.
    Otherwise prints nothing.

    Returns
    -------
    bool
        True if gradients are clean, False if a bad gradient was found.
    """

    bad_found = False
    total_sq_norm = 0.0

    for name, p in model.named_parameters():

        if p.grad is None:
            continue

        g = p.grad

        # accumulate norm (only if finite)
        if torch.isfinite(g).all():
            total_sq_norm += g.norm().item() ** 2

        if not torch.isfinite(g).all():

            if not bad_found:
                print("\n===== NON-FINITE GRADIENT DETECTED =====\n")

            bad_found = True

            nan_count = torch.isnan(g).sum().item()
            inf_count = torch.isinf(g).sum().item()

            finite = g[torch.isfinite(g)]

            print(f"Parameter: {name}")
            print(f"  shape   : {tuple(g.shape)}")
            print(f"  dtype   : {g.dtype}")
            print(f"  device  : {g.device}")
            print(f"  NaN     : {nan_count}")
            print(f"  Inf     : {inf_count}")

            if finite.numel() > 0:
                print(f"  grad_min: {finite.min().item():.6g}")
                print(f"  grad_max: {finite.max().item():.6g}")
                print(f"  grad_mean: {finite.mean().item():.6g}")
                print(f"  grad_norm: {finite.norm().item():.6g}")
            else:
                print("  grad stats: no finite values")

            if p is not None:
                param_finite = p[torch.isfinite(p)]
                if param_finite.numel() > 0:
                    print(f"  param_norm: {param_finite.norm().item():.6g}")
                    print(f"  param_absmax: {param_finite.abs().max().item():.6g}")

            print("")

    if bad_found:
        print("Total grad norm (finite parts only):", total_sq_norm ** 0.5)
        print("\n=======================================\n")
        return False

    return True



class ConDeTrainer:
    """
    Generic trainer for ConDe with arbitrary semantic factors.

    Expected batch:
        x1, x2, x3, z1, z2, z3, pos_idx

    semantic_info: dict mapping:
        {
            "semantic_name": {"head_index": int}
        }
    """

    def __init__(
        self,
        conde_model,
        contrastive_loss_fn,
        residual_loss_fn,
        encode=False,
        coverage_loss_fn = None,
        decomp_loss_fn = None,
        recon_loss_fn = None,
        probe_loss_fn = None,
        residual_reg_fn=None,
        probe = None,
        classifier_probe = None,
        train_encoder = False,
        train_decoder = True,
        decoder=None,
        ae=None,
        device="cuda",
        semantic_info=None,
        lambda_contrast=1.0,
        lambda_decomp=1.0,
        lambda_residual=1.0,
        lambda_coverage=1.0,
        lambda_recon=1.0,
        lambda_residual_loss=1.0,
        lambda_probe=1,
        lr = 1e-4,
        total_steps = 10000,
        grad_clip_norm=5,
        use_upgrad=False,
        decomp_mode = 'inverted', # inverted, inverted_detach, regular or regular_detach
        coverage_mode = 'inverted',  # inverted, z_sem, regular, inverted_detach
        normalize = False,
        require_labels = False,
        mode='image', # 'audio' or 'image'
        save_dir=None,
        use_amp=True,
        mmd_loss = True,
        recon_mode='sum',
        finetune = False,
    ):
        self.model = conde_model.to(device)
        self.decoder = decoder.to(device) if decoder is not None else None
        self.ae = ae.to(device) if ae is not None else None
        self.encode = encode or train_encoder
        self.device = device

        self.semantic_info = semantic_info
        self.contrastive_loss_fn = contrastive_loss_fn
        self.residual_loss_fn = residual_loss_fn
        self.coverage_loss_fn = coverage_loss_fn
        self.decomp_loss_fn = decomp_loss_fn if decomp_loss_fn is not None else F.mse_loss 
        self.recon_loss_fn=recon_loss_fn
        self.residual_reg=residual_reg_fn
        self.probe_loss_fn = probe_loss_fn if probe_loss_fn is not None else decomp_loss_fn

        self.probe=probe
        self.classifier_probe=classifier_probe
        if self.classifier_probe is not None:
            self.classifier_loss_fn = nn.CrossEntropyLoss()

        self.finetune=finetune

        self.train_encoder=train_encoder

        self.lambda_contrast = lambda_contrast
        self.lambda_decomp = lambda_decomp
        self.lambda_residual = lambda_residual
        self.lambda_coverage=lambda_coverage
        self.lambda_recon=lambda_recon
        self.lambda_residual_loss=lambda_residual_loss
        self.lambda_probe=lambda_probe
        self.train_decoder = train_decoder and self.recon_loss_fn is not None


        param_groups = []
        conde_params = []

        if self.finetune:
            if self.train_encoder:
                # freeze entire encoder first
                for p in self.model.encoder.parameters():
                    p.requires_grad = False

                # unfreeze only the last encoder stage
                encoder_stage = self.model.encoder.conv[4]   # last DownBlock
                for p in encoder_stage.parameters():
                    p.requires_grad = True

                # optional: also unfreeze fc, since it directly shapes Z
                for p in self.model.encoder.fc.parameters():
                    p.requires_grad = True

                param_groups.append({
                    "params": list(encoder_stage.parameters()) + list(self.model.encoder.fc.parameters()),
                    "lr": lr / 5,
                    "weight_decay": 1e-5,
                })
            else:
                for p in self.model.encoder.parameters():
                    p.requires_grad = False

            # -----------------
            # Decoder
            # -----------------
            if self.decoder is not None:
                # freeze entire decoder first
                for p in self.decoder.parameters():
                    p.requires_grad = False

                if self.train_decoder:
                    # unfreeze only first decoder stage near bottleneck
                    decoder_stage = self.decoder.net[0]   # first UpBlock
                    for p in decoder_stage.parameters():
                        p.requires_grad = True

                    # optional: also unfreeze decoder fc
                    for p in self.decoder.fc.parameters():
                        p.requires_grad = True

                    param_groups.append({
                        "params": list(decoder_stage.parameters()) + list(self.decoder.fc.parameters()),
                        "lr": lr,
                        "weight_decay": 1e-5,
                    })
        
        else:

            for name, p in self.model.named_parameters():
                if not name.startswith("encoder."):
                    
                    conde_params.append(p)

            param_groups.append({"params": conde_params, "lr": lr, "weight_decay": 1e-6})

            if self.train_encoder:
                for p in self.model.parameters():
                    p.requires_grad = True
                param_groups.append({"params": list(self.model.encoder.parameters()), "lr": lr/5, "weight_decay": 1e-5})

            # Decoder: train only if recon loss is used
            if self.decoder is not None:
                for p in self.decoder.parameters():
                    p.requires_grad = self.train_decoder

            if self.train_decoder and self.decoder is not None:
                param_groups.append({"params": list(self.decoder.parameters()), "lr": lr/10, "weight_decay": 0})
                

        if self.probe is not None:
            params = []
            params += list(self.probe.parameters())
            if self.classifier_probe is not None:
                params += list(self.classifier_probe.parameters())
            param_groups.append({"params": params, "lr": lr, "weight_decay": 0})

        self.optimizer = torch.optim.Adam(
            param_groups,
            eps = 1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                self.optimizer, 
                                max_lr=lr,                     # Peak LR
                                total_steps = total_steps,
                                pct_start=0.06
                            ) 
        self.lr = lr
    
        self.total_steps=total_steps
        self.steps = 0

        self.grad_clip_norm=grad_clip_norm

        self.loss_agg = MetricAggregator()

        self.val_loss_agg = MetricAggregator()
        self.metric_agg = MetricAggregator()


        self.ev_metric  = ExplainedVariance()
        self.mrr_metric = MRRMetric()
        self.gap_metric = DecoderGap(similarity=False)

        possible_decomp = ['inverted', 'inverted_detach', 'regular', 'regular_detach']
        assert decomp_mode in possible_decomp , f"Invalid decomp_mode, please choose between one of {possible_decomp} "

        possible_coverage =  ['inverted', 'regular', 'z_sem', 'inverted_detach']
        assert coverage_mode in possible_coverage , f"Invalid decomp_mode, please choose between one of {possible_coverage} "

        self.decomp_mode = decomp_mode
        self.coverage_mode = coverage_mode

        self.normalize = normalize
        self.require_labels = require_labels

        self.mode=mode

        self.save_dir=save_dir

        self.use_upgrad=False
        if use_upgrad:
            
            self.jd_agg = UPGrad()
            self.use_upgrad=True

        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(device, enabled=use_amp)

        if self.use_upgrad and self.use_amp:
            raise NotImplementedError("If using Upgrad, please disable use_amp.")
        
        self.mmd_loss=mmd_loss
        self.recon_mode=recon_mode
        

    def load_ckpt_states(self, opt_ckpt, current_step):


        self.optimizer.load_state_dict(opt_ckpt)


        # --- Rebuild scheduler (OneCycleLR) ---
        if hasattr(self, "scheduler") and self.scheduler is not None:
  
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                self.optimizer, 
                                max_lr=self.lr,                     # Peak LR
                                total_steps = self.total_steps,
                                pct_start=0.06,
                                last_epoch=current_step - 1,
                            ) 

        self.steps = current_step

        

    def build_pos_neg(self, s1, s2, s3, pos_idx, head_idx):
        """
        s1, s2, s3 = embeddings from semantic head k on x1, x2, x3
        pos_idx[:, k] = Boolean that selects which is positive for x1
        """
        mask = pos_idx[:, head_idx].bool().unsqueeze(1).to(self.device)
        pos = torch.where(mask, s3, s2)
        neg = torch.where(mask, s2, s3)
        return pos, neg

    def _zscore_norm(self, z, eps=1e-8):
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True) + eps
    
        z_norm = (z - mean) / std
        return z_norm, mean, std

    def _zscore_unnorm(self, z_norm, mean, std):
        return z_norm * std + mean

    def decode(self, z, mean=None, std=None):
        if mean is not None and std is not None:
            z = self._zscore_unnorm(z, mean, std)
        z_hat = self.model.reconstruct(Z=z)

        rec = self.decoder(z_hat)
        return rec
    
    def on_epoch_end(self):
        
        self.loss_agg.print()
        self.loss_agg.reset()



    def extract_preds(self, batch):
        """
        Unpacks batch, moves tensors to device, optionally recomputes z from x if
        self.train_encoder is True, then runs ConDe forward for 3 views and returns
        a convenient dict.
        """
        labels = None
        if self.require_labels:
            x1, x2, x3, z1, z2, z3, pos_idx, labels = batch
        else:
            x1, x2, x3, z1, z2, z3, pos_idx = batch

        x1, x2, x3 = [t.to(self.device) for t in (x1, x2, x3)]
        z1, z2, z3 = [t.to(self.device) for t in (z1, z2, z3)]
        pos_idx = pos_idx.to(self.device)

        # If training encoder, recompute embeddings from X so gradients flow
        if self.encode:

            x_cat = torch.cat([x1, x2, x3], dim=0)
            spec_cat = self.ae.pre_process_audio(x_cat) 
            out_cat = self.model(spec_cat)                 
            B = x1.shape[0]

            spec1 = spec_cat[:B]
            spec2 = spec_cat[B:2*B]
            spec3 = spec_cat[2*B:]

            out1, out2, out3 = {}, {}, {}
            for k, v in out_cat.items():
                if isinstance(v, torch.Tensor):
                    # simple case: just split along batch dim
                    out1[k] = v[:B]
                    out2[k] = v[B:2*B]
                    out3[k] = v[2*B:]
                elif isinstance(v, (list, tuple)):
                    # list/tuple of tensors, e.g. S_list
                    out1[k] = [t[:B]       for t in v]
                    out2[k] = [t[B:2*B]    for t in v]
                    out3[k] = [t[2*B:]     for t in v]
        else:
            # Forward through ConDe for each view
            out1 = self.model(z1)  # anchor
            out2 = self.model(z2)
            out3 = self.model(z3)

        
        if 'R_gate' not in out1.keys():
            gate = None
        else:
            gate = out1["R_gate"]
        preds = {
            "x1": x1, "x2": x2, "x3": x3,
            "spec1": spec1, "spec2": spec2, "spec3": spec3,
            "z1_in": z1, "z2_in": z2, "z3_in": z3,
            "pos_idx": pos_idx,
            "labels": labels,
            "R_gate": gate,
            "out1": out1, "out2": out2, "out3": out3,

            # Parsed components (anchor + other views)
            "R1": out1["R"],  "R1_detach": out1['R_detach'], "Zs1": out1["Z_sem"], "S1": out1["S_list"], "z1_hat": out1["Z"], 
            "R2": out2["R"],  "R2_detach": out2['R_detach'], "Zs2": out2["Z_sem"], "S2": out2["S_list"], "z2_hat": out2["Z"],
            "R3": out3["R"],  "R3_detach": out3['R_detach'], "Zs3": out3["Z_sem"], "S3": out3["S_list"], "z3_hat": out3["Z"],
        }

        if self.probe is not None:

            '''probe_z1_hat = self.probe(out1["R_detach"], use_grl=True)
            probe_z2_hat = self.probe(out2["R_detach"], use_grl=True)
            probe_z3_hat = self.probe(out3["R_detach"], use_grl=True)

            probe_preds = { 
                "probe_z1_hat": probe_z1_hat, "probe_z2_hat": probe_z2_hat, "probe_z3_hat": probe_z3_hat, 
            }

            if self.classifier_probe is not None:

                probe_S1_list = [head(probe_z1_hat) for head in self.classifier_probe]
                probe_S2_list = [head(probe_z2_hat) for head in self.classifier_probe]
                probe_S3_list = [head(probe_z3_hat) for head in self.classifier_probe]

                probe_preds["probe_S1_list"] = probe_S1_list
                probe_preds["probe_S2_list"] = probe_S2_list
                probe_preds["probe_S3_list"] = probe_S3_list

            

            preds['probe_preds'] = probe_preds'''

            true_cat = torch.cat([out1["R"], out1["Z_sem"]], dim=-1).to(self.device)
            false_cat = torch.cat([out1["R"], out2["Z_sem"]], dim=-1).to(self.device)

            probe_z_true = self.probe(true_cat)
            probe_z_false= self.probe(false_cat)

            probe_preds = {}
            if self.classifier_probe is not None:

                probe_pred_true = self.classifier_probe(probe_z_true)
                probe_pred_false = self.classifier_probe(probe_z_false)

                probe_preds["probe_pred_true"] = probe_pred_true
                probe_preds["probe_pred_false"] = probe_pred_false
            

            preds['probe_preds'] = probe_preds
        return preds
    

    def add_noise(self, r, mask, start_std=0.1, end_std=0.5, end_pct=0.2):
        """
        Adds Mask-Scaled Gaussian noise to R with a linearly WARM-UP schedule.
        
        mask      : The gating activations, sigma(g(Z)). Shape must match r.
        start_std : Initial noise std (usually 0 to let the AE learn basic recon first)
        end_std   : Maximum noise std (the final bottleneck pressure)
        """
        # 1. Warm-up schedule (increase pressure over time, then hold constant)
        progress = min(self.steps / (self.total_steps * end_pct), 1.0)
        current_std = start_std * (1 - progress) + end_std * progress
        
        # 2. Generate the raw base noise
        raw_noise = torch.randn_like(r) * current_std
        
        # 3. THE TRAP: Scale the noise by the mask (or by R)
        # If mask -> 0, noise -> 0. The escape hatch is open.
        scaled_noise = raw_noise * mask 
        
        return r + scaled_noise
    

    def attenuate_R(self, r, start_keep=1.0, end_keep=0.65, end_pct=0.2):
        """
        Mask-aware multiplicative attenuation of R.

        Goal:
            Reduce decoder reliance on R instead of adding noise.

        Args
        ----
        r : tensor
            Structure branch tensor.
        start_keep : float
            Initial keep factor (1.0 = no attenuation).
        end_keep : float
            Final minimum keep factor.
            Example 0.65 means attenuated dims keep between 65%-100%.
        end_pct : float
            Fraction of training used for warmup.

        Returns
        -------
        attenuated r
        """

        # warmup schedule
        progress = min(self.steps / (self.total_steps * end_pct), 1.0)
        current_keep = start_keep * (1 - progress) + end_keep * progress

        alpha = torch.empty_like(r).uniform_(current_keep, 1.0)

        return r * alpha

    def get_loss(self, *, x1, R1, Zs1, S1, S2, S3, z1_hat, pos_idx, R2=None, R3=None, z2_hat=None, z3_hat=None, R1_detach = None, probe_preds=None, R_gate = None, spec1 = None, labels=None, train=True, skip_decoder=False):

        def safe(t, name):
            mask = torch.isfinite(t)
            if not mask.all():
                print(f"[NaN/Inf detected] in {name}")
                asd
            return torch.where(mask, t, torch.zeros_like(t))
        
        total_contrast = 0.0
        losses_by_semantic = {}
        pos_semantics = []

        loss_dict = {}

        if self.use_upgrad and train==True:
            loss_list = []
            
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
            y = labels[sem_name].to(self.device) if labels is not None else None

            s1 = S1[k]
            s2 = S2[k]
            s3 = S3[k]

            pos, neg = self.build_pos_neg(s1, s2, s3, pos_idx, k)

            if y is not None:
                loss_k = self.contrastive_loss_fn(s1, y)
            else:
                loss_k = self.contrastive_loss_fn(s1, pos, neg)
            loss_k = safe(loss_k, f'contrastive {sem_name}')

            losses_by_semantic[sem_name] = loss_k.detach().item()
            if self.use_upgrad and train==True:
                loss_list.append(loss_k)
            total_contrast = total_contrast + loss_k
            pos_semantics.append(pos)

        loss_dict['contrast_total'] = total_contrast
        loss_dict['cs'] = losses_by_semantic
        
        pos_S_sum = torch.stack(pos_semantics, dim=0).sum(dim=0)  # [B, D]
        S_sum = torch.stack(S1, dim=0).sum(dim=0)

        loss_dict["_pos_S_sum"] = pos_S_sum.detach() if not train else pos_S_sum


        # -------------------------
        # Decomposition loss
        # -------------------------
        if self.decomp_mode == "inverted":
            decomp_loss = self.decomp_loss_fn(Zs1, pos_S_sum)
        elif self.decomp_mode == "inverted_detach":
            decomp_loss = self.decomp_loss_fn(Zs1.detach(), pos_S_sum)
        elif self.decomp_mode == "regular_detach":
            decomp_loss = self.decomp_loss_fn(Zs1.detach(), S_sum)
        else:
            decomp_loss = self.decomp_loss_fn(Zs1, S_sum)
        
        decomp_loss = safe(decomp_loss, 'decomp loss')

        loss_dict['decomp'] = decomp_loss
        if self.use_upgrad and train==True:
            loss_list.append(decomp_loss)

        # -------------------------
        # Coverage loss
        # -------------------------
        z1_hat_rec = self.model.reconstruct(Z=z1_hat)

        if self.coverage_mode == "inverted":
            pos_S_sum_hat = self.model.reconstruct(Z=pos_S_sum)
            coverage_loss = self.coverage_loss_fn(pos_S_sum_hat, z1_hat_rec)
        elif self.coverage_mode == "inverted_detach":
            pos_S_sum_hat = self.model.reconstruct(Z=pos_S_sum)
            coverage_loss = self.coverage_loss_fn(pos_S_sum_hat, z1_hat_rec.detach())
        elif self.coverage_mode == "z_sem":
            Zs1_hat = self.model.reconstruct(Z=Zs1)
            coverage_loss = self.coverage_loss_fn(Zs1_hat, z1_hat_rec)
        else:
            S_sum_hat = self.model.reconstruct(Zs1=S_sum)
            coverage_loss = self.coverage_loss_fn(S_sum_hat, z1_hat_rec)

        coverage_loss=safe(coverage_loss, 'coverage_loss')

        loss_dict['coverage'] = coverage_loss
        if self.use_upgrad and train==True:
            loss_list.append(coverage_loss)


        residual_loss = z1_hat.new_tensor(0.0)
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
            #R_label = R1 if R1_detach is None else R1_detach
            R_label = R1
            residual_loss = residual_loss + safe(self.residual_loss_fn(R_label, S1[k].detach()), f'residual loss {sem_name}')
        
        loss_dict['residual'] = residual_loss
        if self.use_upgrad and train==True:
            loss_list.append(residual_loss)

        residual_reg_loss = z1_hat.new_tensor(0.0)
        if self.residual_reg is not None:
            residual_reg_loss = safe(self.residual_reg(R_gate, z1_hat.detach()), 'residual reg')
            loss_dict['residual_reg'] = residual_reg_loss

        

        recon_loss = z1_hat.new_tensor(0.0)
        if self.recon_loss_fn is not None and not skip_decoder:
            audio_size = x1.shape[1]


            '''S_only = z1_hat

            z_pos = pos_S_sum + self.add_noise(R1, R_gate)

            Z_cat = torch.cat([S_only, z_pos], dim=0)
            spec_cat = self.decode(Z_cat)   

            label_cat = torch.cat([spec1, spec1], dim=0)
            recon_loss = safe(self.recon_loss_fn(spec_cat, label_cat), 'recon loss')'''



            '''S_only = z1_hat

            z_pos = pos_S_sum + self.add_noise(R1, R_gate)
            #z_pos = pos_S_sum + R1

            Z_cat = torch.cat([S_only, z_pos], dim=0)


            spec_cat = self.decode(Z_cat)   

            audio_cat = self.ae.reconstruct_audio(spec_cat)
            audio_cat = audio_cat[:, :audio_size]

            s_only_audios, x_pos_audio = audio_cat.chunk(2, dim=0)

            recon_loss = safe(self.recon_loss_fn(x_pos_audio, x1), 'recon loss') + safe(self.recon_loss_fn(s_only_audios, x1), 'recon loss')'''
            

            '''spec = self.decode(z1_hat)   

            recon_audios = self.ae.reconstruct_audio(spec)[:, :audio_size]

            recon_loss = safe(self.recon_loss_fn(recon_audios, x1), 'recon loss')'''

            if self.recon_mode == 'z_sum':
                S_only = z1_hat

                z_pos = pos_S_sum + self.add_noise(R1, R_gate)

                Z_cat = torch.cat([S_only, z_pos], dim=0)
                spec_cat = self.decode(Z_cat)   
                recon_audios = self.ae.reconstruct_audio(spec_cat)[:, :audio_size]

                s_only_audios, x_pos_audio = recon_audios.chunk(2, dim=0)

                recon_loss = safe(self.recon_loss_fn(x_pos_audio, x1), 'recon loss') + safe(self.recon_loss_fn(s_only_audios, x1), 'recon loss')
            elif self.recon_mode == 'z_sem':
                S_only = z1_hat

                z_pos = pos_S_sum

                Z_cat = torch.cat([S_only, z_pos], dim=0)
                spec_cat = self.decode(Z_cat)   

                recon_audios = self.ae.reconstruct_audio(spec_cat)[:, :audio_size]

                s_only_audios, x_pos_audio = recon_audios.chunk(2, dim=0)

                recon_loss = safe(self.recon_loss_fn(x_pos_audio, x1), 'recon loss') + safe(self.recon_loss_fn(s_only_audios, x1), 'recon loss')
            elif self.recon_mode == 'sum':

                spec = self.decode(pos_S_sum + self.attenuate_R(R1))   

                recon_audios = self.ae.reconstruct_audio(spec)[:, :audio_size]

                recon_loss = safe(self.recon_loss_fn(recon_audios, x1), 'recon loss')

            loss_dict['recon'] = recon_loss
            if self.use_upgrad and train==True:
                loss_list.append(recon_loss)

            
        # -------------------------
        # Probe losses
        # -------------------------
        probe_total_contrast = z1_hat.new_tensor(0.0)
        #probe_recon_loss = z1_hat.new_tensor(0.0)   
        if probe_preds is not None:
            '''probe_losses_by_semantic = {}
            probe_total_contrast=0.0
            probe_recon_loss = self.probe_loss_fn(probe_preds['probe_z1_hat'], pos_S_sum.detach())
            loss_dict['probe_recon_loss'] = probe_recon_loss

            if self.classifier_probe is not None:
                for sem_name, info in self.semantic_info.items():
                    k = info["head_index"]
                    y = labels[sem_name].to(self.device) if labels is not None else None

                    s1 = probe_preds['probe_S1_list'][k]
                    s2 = probe_preds['probe_S2_list'][k]
                    s3 = probe_preds['probe_S3_list'][k]

                    pos, neg = self.build_pos_neg(s1, s2, s3, pos_idx, k)

                    if y is not None:
                        loss_k = self.contrastive_loss_fn(s1, y)
                    else:
                        loss_k = self.contrastive_loss_fn(s1, pos, neg)
                    loss_k = safe(loss_k, f'adv conrtrast {sem_name}')

                    probe_losses_by_semantic[sem_name] = loss_k.detach().item()
                    if self.use_upgrad and train==True:
                        loss_list.append(loss_k)
                    probe_total_contrast = probe_total_contrast + loss_k

                loss_dict['probe_total_contrast'] = probe_total_contrast
                loss_dict['probe_losses_by_semantic'] = probe_losses_by_semantic'''
            
            all_preds = torch.cat([probe_preds['probe_pred_false'], probe_preds['probe_pred_true']], dim=0)

            targets = torch.cat([
                torch.zeros(R1.shape[0], dtype=torch.long, device=R1.device),
                torch.ones(R1.shape[0], dtype=torch.long, device=R1.device)
            ], dim=0).to(self.device)


            probe_classifier_loss = self.classifier_loss_fn(all_preds, targets)
            loss_dict['probe_classifier_loss'] = probe_classifier_loss

        z_norm = safe(z1_hat.pow(2).mean(), 'norm')
        loss_dict['z_norm'] = z_norm

        '''dZ_pos = (z1_hat - z2_hat).detach()
        dZ_neg = (z1_hat - z3_hat).detach()

        dR_pos = R1 - R2
        dR_neg = R1 - R3

        var_Z_pos = (dZ_pos * dZ_pos).sum(dim=-1).clamp(min=1e-6)
        var_Z_neg = (dZ_neg * dZ_neg).sum(dim=-1).clamp(min=1e-6)

        cov_RZ_pos = (dR_pos * dZ_pos).sum(dim=-1)
        cov_RZ_neg = (dR_neg * dZ_neg).sum(dim=-1)

        ratio_pos = cov_RZ_pos / var_Z_pos
        ratio_neg = cov_RZ_neg / var_Z_neg


        loss_pos = (1.0 - ratio_pos).abs().mean()

        loss_neg = (ratio_neg).abs().mean()'''
        loss_match=None
        if self.mmd_loss:
            sim_pos = F.cosine_similarity(R1, R2, dim=-1)
            sim_neg = F.cosine_similarity(R1, R3, dim=-1)

            E_sim_pos = sim_pos.mean()
            E_sim_neg = sim_neg.mean()

            loss_mmd = (E_sim_pos - E_sim_neg).abs()

            #loss_match = loss_pos + loss_neg + loss_mmd
            loss_match = loss_mmd*10
            loss_dict['loss_match'] = loss_match


        # -------------------------
        # Total loss
        # -------------------------
        total_loss = (
            self.lambda_contrast * total_contrast
            + self.lambda_decomp * decomp_loss
            + self.lambda_residual * residual_loss
            + self.lambda_coverage * coverage_loss
            + (self.lambda_recon * recon_loss if self.recon_loss_fn is not None else 0.0)
            + self.lambda_residual_loss * residual_reg_loss
            + z_norm*0.1
            + loss_match if loss_match is not None else 0.0
            
            
        )

        loss_dict['total'] = total_loss

        if self.use_upgrad and train==True:
            return loss_dict, loss_list
        return loss_dict


    def train_step(self, batch, skip_decoder=False):
        self.model.train()
        if self.train_decoder and not skip_decoder:
            self.decoder.train()


        with torch.amp.autocast(self.device, enabled=self.use_amp, dtype=torch.float16):
            preds = self.extract_preds(batch)            

            self.optimizer.zero_grad()


            if self.use_upgrad:
                loss_dict, jd_losses = self.get_loss(
                    x1=preds["x1"],
                    R1=preds["R1"],
                    Zs1=preds["Zs1"],
                    S1=preds["S1"],
                    S2=preds["S2"],
                    S3=preds["S3"],
                    z1_hat=preds["z1_hat"],
                    pos_idx=preds["pos_idx"],
                    R2=preds['R2'],
                    z2_hat=preds['z2_hat'],
                    R3=preds['R3'],
                    z3_hat=preds['z3_hat'],
                    R1_detach=preds['R1_detach'],
                    R_gate=preds['R_gate'],
                    spec1 = preds['spec1'],
                    labels=preds["labels"],
                    probe_preds=None if self.probe is None else preds['probe_preds'],
                    train=True,
                    skip_decoder=skip_decoder
                )
                    
                torchjd.backward(jd_losses, self.jd_agg)

            else:
                loss_dict = self.get_loss(
                    x1=preds["x1"],
                    R1=preds["R1"],
                    Zs1=preds["Zs1"],
                    S1=preds["S1"],
                    S2=preds["S2"],
                    S3=preds["S3"],
                    z1_hat=preds["z1_hat"],
                    pos_idx=preds["pos_idx"],
                    R1_detach=preds['R1_detach'],
                    R2=preds['R2'],
                    z2_hat=preds['z2_hat'],
                    R3=preds['R3'],
                    z3_hat=preds['z3_hat'],
                    R_gate=preds['R_gate'],
                    spec1 = preds['spec1'],
                    probe_preds=None if self.probe is None else preds['probe_preds'],
                    labels=preds["labels"],
                    train=True,
                    skip_decoder=skip_decoder
                )
                self.scaler.scale(loss_dict["total"]).backward()

            if self.grad_clip_norm is not None:
                params = list(self.model.parameters())
                if self.train_decoder:
                    params += list(self.decoder.parameters())
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.steps += 1
            #check_bad_gradients(self.model)

            '''stats = {
            
                "total": loss_dict["total"].item(),
                "decomp": loss_dict["decomp"].item(),
                "residual": loss_dict["residual"].item(),
                "contrast_by_semantic": loss_dict["contrast_by_semantic"],
                "coverage_loss": loss_dict["coverage"].item(),
                **({"recon": loss_dict["recon"].detach().item()} if self.recon_loss_fn is not None and not skip_decoder else {}),
                **({"residual_reg": loss_dict["residual_reg"].detach().item()} if self.residual_reg is not None else {}),      
                **({"probe_recon_loss": loss_dict["probe_recon_loss"].detach().item()} if self.residual_reg is not None else {}),  
                **({"probe_losses_by_semantic": loss_dict["probe_losses_by_semantic"].detach().item()} if self.residual_reg is not None else {}),  
                **({"probe_recon_loss": loss_dict["probe_recon_loss"].detach().item()} if self.residual_reg is not None else {}),    
            }'''
            

            self.loss_agg.update(loss_dict)
            return loss_dict


    @torch.no_grad()
    def validation_step(self, batch, n_show=5, epoch=0):
        self.model.eval()
        if self.train_decoder:
            self.decoder.eval()
        with torch.amp.autocast(self.device, enabled=self.use_amp, dtype=torch.float16):

            preds = self.extract_preds(batch)

            loss_dict = self.get_loss(
                x1=preds["x1"],
                R1=preds["R1"],
                Zs1=preds["Zs1"],
                S1=preds["S1"],
                S2=preds["S2"],
                S3=preds["S3"],
                z1_hat=preds["z1_hat"],
                pos_idx=preds["pos_idx"],
                R1_detach=preds['R1_detach'],
                R2=preds['R2'],
                z2_hat=preds['z2_hat'],
                R3=preds['R3'],
                z3_hat=preds['z3_hat'],
                R_gate=preds['R_gate'],
                probe_preds=None if self.probe is None else preds['probe_preds'],
                spec1 = preds['spec1'],
                labels=preds["labels"],
                train=False,
            )

            '''self.val_loss_agg.update({
                "total": loss_dict["total"].item(),
                "decomp": loss_dict["decomp"].item(),
                "residual": loss_dict["residual"].item(),
                "contrast_by_semantic": loss_dict["contrast_by_semantic"],
                "coverage_loss": loss_dict["coverage"].item(),
                **({"recon": loss_dict["recon"].item()} if self.recon_loss_fn is not None else {}),
                **({"residual_reg": loss_dict["residual_reg"].item()} if self.residual_reg is not None else {}),
            }, n=preds["z1_in"].shape[0])'''
            self.val_loss_agg.update(loss_dict)

            # --- Online metrics (keep your existing logic, but reuse values from preds/loss_dict)
            S1_sum = sum(preds["S1"])
            EV_sem, EV_full = self.ev_metric(preds["z1_hat"], S1_sum, preds["R1"])

            # These already match your prior validation code
            X_ae = self.decode(preds["z1_hat"])
            X_sr = self.decode(loss_dict["_pos_S_sum"] + preds["R1"])
            dec_gap = self.gap_metric(X_ae, X_sr)

            mrr_list = []
            for sem_name, info in self.semantic_info.items():
                k = info["head_index"]
                sA = preds["S1"][k]
                sP = preds["S3"][k]
                mrr_list.append(self.mrr_metric(sA, sP))
            mrr_mean = sum(mrr_list) / len(mrr_list)

            self.metric_agg.update({
                "mrr": mrr_mean,
                "EV_sem": EV_sem,
                "EV_full": EV_full,
                "decoder_gap": dec_gap
            }, n=preds["z1_in"].shape[0])

            if n_show > 0:
                # Keep your existing visualization call signature
                batch_for_vis = (preds["x1"], preds["x2"], preds["x3"],
                                preds["z1_hat"], preds["z2_hat"], preds["z3_hat"],
                                preds["pos_idx"])
                self._visualize_val(batch_for_vis, preds["out1"], preds["out2"], preds["out3"], n_show=n_show, epoch=epoch)
                self.did_visualize = True

    @torch.no_grad()
    def _visualize_val(self, batch, out1, out2, out3, n_show=5, epoch=0):
        
    
        x1, x2, x3, z1, z2, z3, pos_idx = batch
        R1, Zs1, S1, z1_hat = out1["R"], out1["Z_sem"], out1["S_list"], out1["Z"]
        R2, Zs2, S2, z2_hat = out2["R"], out2["Z_sem"], out2["S_list"], out2["Z"]
        R3, Zs3, S3, z3_hat = out3["R"], out3["Z_sem"], out3["S_list"], out3["Z"]
    
        if self.mode=='audio':
            rec_dict = {
                "spec1": self.ae.pre_process_audio(x1),
                "spec2": self.ae.pre_process_audio(x2),
                "spec3": self.ae.pre_process_audio(x3),
                "orig_x1": self.decode(z1_hat),
                "orig_x2": self.decode(z2_hat),
                "orig_x3": self.decode(z3_hat),
            }
        else:
            rec_dict = {
                "true1": x1,
                "true2": x2,
                "true3": x3,
                "orig_x1": self.decode(z1_hat),
                "orig_x2": self.decode(z2_hat),
                "orig_x3": self.decode(z3_hat),
            }

    
        # -------------------------------
        # Swapped reconstructions
        # -------------------------------
        for sem_name, info in self.semantic_info.items():
            k = info["head_index"]
    
            s1 = S1[k]
            s2 = S2[k]
            s3 = S3[k]
    
            mask = pos_idx[:, k].bool().unsqueeze(1)
    
            pos = torch.where(mask, s3, s2)
            neg = torch.where(mask, s2, s3)
    
            other_sum = sum(S1[j] for j in range(len(S1)) if j != k)
    
            rec_dict[f"{sem_name}_pos"] = self.decode(pos + other_sum + R1)
            rec_dict[f"{sem_name}_neg"] = self.decode(neg + other_sum + R1)
    
        # ==========================================================
        # IMAGE MODE (unchanged logic)
        # ==========================================================
        if self.mode == "image":

            save = self.save_dir is not None
            if save:
                img_dir = os.path.join(self.save_dir, f"{epoch}")
                os.makedirs(img_dir, exist_ok=True)
            cols = len(rec_dict)
            fig, axes = plt.subplots(n_show, cols, figsize=(cols * 2, n_show * 2))
    
            for row in range(n_show):
                for col, (name, data) in enumerate(rec_dict.items()):
                    ax = axes[row, col]
                    img = data[row].detach().cpu().numpy()
                    img = np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
                    ax.imshow(img)
                    ax.axis("off")
                    if row == 0:
                        ax.set_title(name, fontsize=10)
    
            plt.tight_layout()
            if save:
                fig.savefig(f"imgs.png")
            else:                        
              plt.show()
            return
    
        # ==========================================================
        # AUDIO MODE
        # ==========================================================
        if self.mode == "audio":

            save = self.save_dir is not None
            if save:
                spec_dir = os.path.join(self.save_dir, f"{epoch}/specs")
                audio_dir = os.path.join(self.save_dir, f"{epoch}/audio")
                os.makedirs(spec_dir, exist_ok=True)
                os.makedirs(audio_dir, exist_ok=True)

            # compute magnitudes once
            mag_dict = {}
            for name, spec in rec_dict.items():
                spec = spec.detach().cpu()
                real = spec[:, 0]
                imag = spec[:, 1]
                mag = torch.sqrt(real**2 + imag**2 + 1e-8)
                mag_dict[name] = mag

            B = min(n_show, next(iter(mag_dict.values())).shape[0])
            cols = len(mag_dict)

            fig, axes = plt.subplots(B, cols, figsize=(cols * 3, B * 2.5))
            if B == 1:
                axes = axes[None, :]
            if cols == 1:
                axes = axes[:, None]

            for row in range(B):
                for col, (name, mag) in enumerate(mag_dict.items()):
                    ax = axes[row, col]
                    img = mag[row].log1p().numpy()
                    ax.imshow(
                        img,
                        origin="lower",
                        aspect="auto",
                        cmap="magma",
                    )
                    ax.axis("off")
                    if row == 0:
                        ax.set_title(name, fontsize=10)

                    # -------------------------------
                    # Save spectrogram
                    # -------------------------------
                    if save:
                        fname = f"{name}_sample{row}.png"
                        path = os.path.join(spec_dir, fname)
                        plt.imsave(path, img, cmap="magma", origin="lower")

            plt.tight_layout()

            if save:
                fig.savefig(os.path.join(spec_dir, 'specs.png'))
            else:        
                
                plt.show()

            # -------------------------------
            # Audio rendering
            # -------------------------------
            for name, spec in rec_dict.items():
                spec = spec[:B].to(self.device)
                audio = self.ae.reconstruct_audio(spec)
                audio = audio.detach().cpu().numpy()

                for i in range(B):
                    if save:
                        fname = f"{name}_sample{i}.wav"
                        path = os.path.join(audio_dir, fname)
                        wavfile.write(
                            path,
                            self.ae.sr,
                            audio[i]
                        )
                    else:
                        display(Audio(audio[i], rate=self.ae.sr))

    def on_validation_epoch_end(self):

        print("\n[Validation Losses]")
        self.val_loss_agg.print()
        print("\n[Validation Metrics]")
        self.metric_agg.print()

        self.val_loss_agg.reset()
        self.metric_agg.reset()
