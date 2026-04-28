import torch
import torchaudio
import torch.nn as nn
from torch.optim import Adam
import torchaudio.transforms as T
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


from torch.cuda.amp import autocast, GradScaler

import os
import matplotlib.pyplot as plt
import wandb
import time


class InstrumentClassifier(nn.Module):
    def __init__(self, backbone: torch.nn.Module, spec_class, sr=22050, lr = 1e-4, weight_decay = 0,  warmup_steps: int = 1000, total_steps=100000, device: torch.device = None, clip_value=2, save_dir = 'res', padded_spec_width = 256):
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
        self.optimizer = self.configure_optimizers()
        self.clip_value=clip_value
        
        self.save_dir = save_dir

        self.global_step = 0
        

        self.sr=sr
        self.padded_spec_width=padded_spec_width


        self.original_len=0

    def forward(self, x):
        with torch.no_grad():
            x = self.pre_process_audio(x)
        
        out = self.model(x)
        #self.specs = x[:, :, :, :self.original_len]
        
        return out

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


    def calculate_loss(self, preds, labels):
        """
        preds: logits tensor of shape [B, C]
        labels: int tensor of shape [B]
        """
        loss = F.cross_entropy(preds, labels)

        loss_dict = {
            "total_loss": loss.item(),
            "ce_loss": loss.item()
        }

        return loss, loss_dict


    def calculate_metrics(self, preds, labels):
        """
        preds: logits tensor [B, C]
        labels: int tensor [B]
        """
        pred_classes = torch.argmax(preds, dim=1)

        y_true = labels.detach().cpu().numpy()
        y_pred = pred_classes.detach().cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        return {
            "acc": acc,
            "F1": f1
        }
    


    def training_step(self, batch, batch_idx):
        
        x = batch[0]
        x = x.to(self.device)

        labels = batch[1]
        labels = labels.to(self.device)
        


        preds = self.get_pred(x)

    

        loss, loss_dict = self.calculate_loss(preds, labels)

        self.optimizer_step(loss)

        return loss_dict

    def validation_step(self, batch, batch_idx, max_images=8, epoch = 0):
        """
        Evaluation step for validation or testing.
        Logs reconstruction loss and evaluation metrics, and saves sample reconstructions.
        """
        x = batch[0]
        x = x.to(self.device)

        labels = batch[1]
        labels = labels.to(self.device)
    
        with torch.no_grad():
            preds = self.get_pred(x)
            loss, loss_dict = self.calculate_loss(preds, labels)
            
            metrics = self.calculate_metrics(preds, labels)
    
            
        return {**loss_dict, **metrics}

    

    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        return optimizer
    
    def optimizer_step(self, total_loss):
        self.optimizer.zero_grad()

        total_loss.backward()


        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)

        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return  # skip this step

        # step optimizer
        self.optimizer.step()

  
        self.global_step += 1


    
