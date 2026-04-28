import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_ae(ae, criterion, optimizer, train_dl, test_dl, epochs = 100, val_interval=5, device='cuda', disable_tqdm=True, save_dir=None, scheduler=None, start_epoch=0, best_val_loss=None):

    if best_val_loss is None:
        best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        # ----------------- TRAIN -----------------
        ae.train()
        running_loss = 0.0
        running_loss_dict = {k: 0.0 for k in ["mse","l1","charb","ssim","grad","lpips"]}

        
    
        for imgs in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", disable=disable_tqdm):
            imgs = imgs.to(device)
    
            # forward
            outputs = ae(imgs)
            loss, loss_dict = criterion(outputs, imgs)
    
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    
            # accumulate
            running_loss += loss.item() * imgs.size(0)
            for k, v in loss_dict.items():
                running_loss_dict[k] += v.item() * imgs.size(0)
    
        train_loss = running_loss / len(train_dl.dataset)
        train_loss_dict = {k: v / len(train_dl.dataset) for k, v in running_loss_dict.items()}
    
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | "
              + " | ".join([f"{k}:{v:.4f}" for k,v in train_loss_dict.items()]))
    
        # ----------------- SHOW SAMPLE IMAGES + VALID -----------------
        if epoch % val_interval == 0:
            ae.eval()
            val_loss = 0.0
            val_loss_dict = {k: 0.0 for k in ["mse","l1","charb","ssim","grad","lpips"]}
            with torch.no_grad():
                for imgs in test_dl:
                    imgs = imgs.to(device)
                    outputs = ae(imgs)
                    loss, loss_dict = criterion(outputs, imgs)
                    val_loss += loss.item() * imgs.size(0)
                    for k, v in loss_dict.items():
                        val_loss_dict[k] += v.item() * imgs.size(0)
    
            val_loss = val_loss / len(test_dl.dataset)
            val_loss_dict = {k: v / len(test_dl.dataset) for k, v in val_loss_dict.items()}

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                
                # Save latest checkpoint
                latest_ckpt_path = os.path.join(save_dir, "latest.ckpt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': ae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, latest_ckpt_path)
                
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = os.path.join(save_dir, "best.ckpt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': ae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, best_ckpt_path)
                    print(f"✅ Saved new best model at epoch {epoch+1} (val_loss={val_loss:.4f})")
    
            print(f"Val Loss: {val_loss:.4f} | "
                  + " | ".join([f"{k}:{v:.4f}" for k,v in val_loss_dict.items()]))
    
            imgs = imgs[:8].cpu()          # last batch (take 8 samples)
            outputs = outputs[:8].cpu()
    
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            for i in range(8):
                axes[0, i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
                axes[0, i].axis("off")
                axes[1, i].imshow(outputs[i].permute(1, 2, 0).clamp(0, 1))
                axes[1, i].axis("off")
            plt.suptitle(f"Epoch {epoch+1} Reconstructions")

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()