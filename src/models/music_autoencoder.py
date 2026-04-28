import torch
import torchaudio
import torch.nn as nn
from torch.optim import Adam
import torchaudio.transforms as T
import torch.nn.functional as F


from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

import os
import matplotlib.pyplot as plt
import wandb
import time

def save_examples(x, recon, save_dir, save_name, max_images=8, step=0):
    """
    Save side-by-side ORIGINAL vs RECONSTRUCTED spectrograms,
    using the same visualization style as make_figure().
    """

    os.makedirs(save_dir, exist_ok=True)

    x = x.detach().cpu()
    recon = recon.detach().cpu()

    B = min(x.shape[0], max_images)

    # x and recon are (B, 2, F, T)
    real_x = x[:, 0]
    imag_x = x[:, 1]
    mag_x  = torch.sqrt(real_x**2 + imag_x**2 + 1e-8)

    real_r = recon[:, 0]
    imag_r = recon[:, 1]
    mag_r  = torch.sqrt(real_r**2 + imag_r**2 + 1e-8)

    # Layout:
    #   For each sample:
    #       Original: [Mag | Real | Imag]
    #       Recon   : [Mag | Real | Imag]
    #
    fig, axs = plt.subplots(
        B, 6, figsize=(20, 4 * B)
    )

    if B == 1:
        axs = axs.reshape(1, 6)

    for i in range(B):

        # ------- Original -------
        axs[i, 0].imshow(mag_x[i].log1p(), origin="lower", aspect="auto", cmap='magma')
        axs[i, 0].set_title(f"Sample {i} — Original Mag")
        axs[i, 0].set_xticks([]); axs[i, 0].set_yticks([])

        axs[i, 1].imshow(real_x[i], origin="lower", aspect="auto", cmap='seismic')
        axs[i, 1].set_title("Original Real")
        axs[i, 1].set_xticks([]); axs[i, 1].set_yticks([])

        axs[i, 2].imshow(imag_x[i], origin="lower", aspect="auto", cmap='seismic')
        axs[i, 2].set_title("Original Imag")
        axs[i, 2].set_xticks([]); axs[i, 2].set_yticks([])

        # ------- Reconstruction -------
        axs[i, 3].imshow(mag_r[i].log1p(), origin="lower", aspect="auto", cmap='magma')
        axs[i, 3].set_title(f"Sample {i} — Recon Mag")
        axs[i, 3].set_xticks([]); axs[i, 3].set_yticks([])

        axs[i, 4].imshow(real_r[i], origin="lower", aspect="auto", cmap='seismic')
        axs[i, 4].set_title("Recon Real")
        axs[i, 4].set_xticks([]); axs[i, 4].set_yticks([])

        axs[i, 5].imshow(imag_r[i], origin="lower", aspect="auto", cmap='seismic')
        axs[i, 5].set_title("Recon Imag")
        axs[i, 5].set_xticks([]); axs[i, 5].set_yticks([])

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{save_name}_comparison.png")
    plt.savefig(out_path)
    plt.close(fig)

    wandb.log({"spectrogram_comparison": wandb.Image(out_path)}, step=step)



def get_onecycle_scheduler(
    optimizer,
    max_lr: float,
    total_steps: int = None,
    epochs: int = None,
    steps_per_epoch: int = None,
    warmup_steps: int = None,
    pct_start: float = None,
    anneal_strategy: str = 'cos',
    cycle_momentum: bool = True,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
):
    """
    Creates a OneCycleLR scheduler.

    You must supply either:
      - total_steps (int): total number of optimizer.step() calls, OR
      - epochs (int) AND steps_per_epoch (int).

    warmup_steps or pct_start determines the warmup fraction:
      - pct_start = warmup_steps / total_steps
      - or supply pct_start directly (float in (0,1)).
    """

    # Derive pct_start if warmup_steps was given
    if pct_start is None:
        if warmup_steps is None or total_steps is None:
            raise ValueError("Provide either pct_start or (warmup_steps and total_steps).")
        pct_start = warmup_steps / total_steps

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        last_epoch=-1)
    

def get_warmup_scheduler(optimizer, warmup_steps: int):
    """
    Linear warmup: LR grows linearly from 0 -> 1 (multiplier) over warmup_steps.
    """
    return LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))


def get_decay_scheduler(optimizer, step_size: int, gamma: float):
    """
    Step decay: multiplies LR by gamma every `step_size` steps.
    """
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_combined_scheduler(
    optimizer,
    warmup_steps: int,
    decay_step_size: int,
    decay_gamma: float
):
    """
    Sequential scheduler: warmup for `warmup_steps`, then step-decay.
    """
    warmup_sched = get_warmup_scheduler(optimizer, warmup_steps)
    decay_sched = get_decay_scheduler(optimizer, decay_step_size, decay_gamma)
    # milestones=[warmup_steps] tells SequentialLR to switch at that step
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, decay_sched],
        milestones=[warmup_steps]
    )


class MusicAE(nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        loss_aggregator,
        spec_class,
        sr=22050,
        lr = 1e-4,
        weight_decay = 0,
        warmup_steps: int = 1000,
        total_steps=100000,
        device: torch.device = None,
        clip_value=2,
        save_dir = 'res',
        padded_spec_width = 256
    ):
        super().__init__()


        self.spec_class = spec_class


        self.model = backbone
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # instantiate optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer, self.scheduler= self.configure_optimizers()
        self.clip_value=clip_value
        
        self.loss_aggregator = loss_aggregator
        self.save_dir = save_dir

        self.global_step = 0
        self.specs = None
        

        self.emb_size = self.model.embed_dim

        self.sr=sr
        self.padded_spec_width=padded_spec_width


        self.original_len=0

    def forward(self, x):
        with torch.no_grad():
            x = self.pre_process_audio(x)
        
        out = self.model(x, noise_mu=0.)
        #self.specs = x[:, :, :, :self.original_len]
        out = out[:, :, :, :self.original_len]
        return out, x[:, :, :, :self.original_len]

    def encode(self, x):
        with torch.no_grad():
            x = self.pre_process_audio(x)
        
        z = self.model.encoder(x)
        return z

    def decode(self, z):
        return self.model.decoder(z)[:, :, :, :self.original_len]

    def get_pred(self, x):
        return self(x)

    def get_device(self):
        """Returns the current device of the model."""
        return next(self.parameters()).device

    def pre_process_audio(self, audios: torch.Tensor, spec = False) -> torch.Tensor:
        """
        Pre-process raw audio waveforms into spectrograms in the expected format.
        """

        audios = audios.float()
        with torch.amp.autocast(self.device, enabled=False):
            specs =  self.spec_class(audios)

        self.original_len = specs.shape[-1]

       
        target_len = self.padded_spec_width
        padding = target_len - specs.shape[-1]
        if padding > 0:
            specs = F.pad(specs, (0, padding), value=0)

        if len(specs.shape) < 4:
            specs = specs.unsqueeze(1)

        


        return specs


    def calculate_loss(self, recon, x, recon_audio, specs):
        real_pred, imag_pred = recon[:, 0], recon[:, 1]
        real_tgt,  imag_tgt  = specs[:, 0],  specs[:, 1]

        mag_pred = torch.sqrt(real_pred**2 + imag_pred**2 + 1e-8)
        mag_tgt  = torch.sqrt(real_tgt**2  + imag_tgt**2  + 1e-8)

        '''mag_pred = recon[:, 0]
        mag_tgt  = specs[:, 0]'''

        input_lookup = {'recon': recon, 'x': x, "specs": specs, 'recon_audio':recon_audio, "mag_pred": mag_pred, "mag": mag_tgt}
        loss, loss_dict = self.loss_aggregator(input_lookup)
        return loss, loss_dict


    def calculate_metrics(self, recon, target):
        """
        Computes evaluation metrics like MS-SSIM.
        Assumes input is in [0, 1] or normalized consistently.
        """
        '''with torch.no_grad():
            mssim = ms_ssim(recon, target, data_range=1.0, size_average=True).item()'''
        return {"mssim": 1}
    
    def reconstruct_audio(self, spec):
        spec = spec.float()
        with torch.amp.autocast(self.device, enabled=False):

            return self.spec_class.inverse(spec)


    def training_step(self, batch, batch_idx):
        
        x = batch
        x = x.to(self.device)
        


        recon, spec = self.get_pred(x)
        recon_audio = self.reconstruct_audio(recon)

        #print(recon.min(), recon.max())
    

        loss, loss_dict = self.calculate_loss(recon, x, recon_audio, spec)

        self.optimizer_step(loss)

        return loss_dict

    def validation_step(self, batch, batch_idx, max_images=8, epoch = 0):
        """
        Evaluation step for validation or testing.
        Logs reconstruction loss and evaluation metrics, and saves sample reconstructions.
        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(self.device)
    
        with torch.no_grad():
            recon, specs = self.get_pred(x)
            recon_audio = self.reconstruct_audio(recon)
            loss, loss_dict = self.calculate_loss(recon, x, recon_audio, specs)
            
            metrics = self.calculate_metrics(recon, x)
    
            # Save reconstructions
            if max_images > 0:
                save_examples(specs, recon, save_dir = f'{self.save_dir}/{epoch}', save_name=f'recon_{epoch}_{batch_idx}', max_images=max_images, step=self.global_step)
                self.save_audio_examples(
                    original_specs=specs,     # the original CQT/NSGT
                    pred_specs=recon,              # model predicted CQT/NSGT
                    save_dir=f'{self.save_dir}/{epoch}',
                    prefix=f"audio_{epoch}_{batch_idx}",
                    max_audios=4
                )
            
        return {**loss_dict, **metrics}

    

    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = get_onecycle_scheduler(
            optimizer,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            max_lr = self.lr
        )
        
        return optimizer, scheduler
    
    def optimizer_step(self, total_loss):
        self.optimizer.zero_grad()

        total_loss.backward()


        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)

        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return  # skip this step

        # step optimizer
        self.optimizer.step()

        # scheduler updates normally
        self.scheduler.step()
        self.global_step += 1

    def save_audio_examples(self, original_specs, pred_specs, save_dir, prefix, max_audios=4):
        """
        Save a few audio examples reconstructed from their CQT/NSGT representations.
        - original_specs: [B, 2, T, F]
        - pred_specs:     [B, 2, T, F]
        """
        os.makedirs(save_dir, exist_ok=True)

        # Limit how many examples we save
        B = min(original_specs.size(0), max_audios)

        # Move to CPU and detach
        original_specs = original_specs[:B].detach()
        pred_specs     = pred_specs[:B].detach()

        # Reconstruct audio from CQT/NSGT
        original_audio = self.reconstruct_audio(original_specs).cpu()  # [B, samples]
        pred_audio     = self.reconstruct_audio(pred_specs).cpu()      # [B, samples]

        # Ensure tensors are CPU float32 for torchaudio
        original_audio = original_audio.cpu().float()
        pred_audio     = pred_audio.cpu().float()

        for i in range(B):
            # Each is [samples]
            orig = original_audio[i].unsqueeze(0)  # torchaudio expects [channels, samples]
            pred = pred_audio[i].unsqueeze(0)

            orig_path = os.path.join(save_dir, f"{prefix}_orig_{i}.wav")
            pred_path = os.path.join(save_dir, f"{prefix}_pred_{i}.wav")

            torchaudio.save(orig_path, orig, self.sr)
            torchaudio.save(pred_path, pred, self.sr)

        wandb.log({"original_audio": wandb.Audio(orig_path)}, step=self.global_step)
        wandb.log({"reconstructed_audio": wandb.Audio(pred_path)}, step=self.global_step)

        print(f"[Audio] Saved {B} original + predicted audio files in {save_dir}")

    
