import os
import math
import random
import importlib
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def load_config(module_path: str):
    mod = importlib.import_module(module_path)
    return mod.CONFIG


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets_and_loaders(cfg):
    train_ds = cfg["data"]["train_dataset_cls"](**cfg["data"]["train_dataset_kwargs"])
    val_ds = cfg["data"]["val_dataset_cls"](**cfg["data"]["val_dataset_kwargs"])

    train_dl = DataLoader(train_ds, **cfg["data"]["train_loader_kwargs"])
    val_dl = DataLoader(val_ds, **cfg["data"]["val_loader_kwargs"])

    return train_ds, val_ds, train_dl, val_dl


def build_model(cfg, train_ds, device):
    num_classes = len(train_ds.program_to_indices)

    backbone_kwargs = dict(cfg["model"]["backbone_kwargs"])
    backbone_kwargs["num_classes"] = num_classes

    spec_class = cfg["model"]["spec_cls"](**cfg["model"]["spec_kwargs"])

    backbone = cfg["model"]["backbone_cls"](**backbone_kwargs).to(device)

    model_kwargs = dict(cfg["model"]["model_kwargs"])
    model = cfg["model"]["model_cls"](backbone, spec_class, **model_kwargs).to(device)

    return model, num_classes


def build_optimizer_and_scheduler(cfg, model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    scheduler = None
    if cfg.get("scheduler", {}).get("use", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["scheduler"]["t_max"],
            eta_min=cfg["scheduler"]["eta_min"],
        )

    return optimizer, scheduler


def save_checkpoint(save_path, epoch, model, optimizer, best_f1, cfg, class_map):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1,
        "config": cfg,
        "program_to_indices": class_map,
    }
    torch.save(ckpt, save_path)


def maybe_load_checkpoint(ckpt_path, model, optimizer=None, device="cpu"):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return 0, -1.0

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])


    start_epoch = ckpt.get("epoch", -1) + 1
    best_f1 = ckpt.get("best_f1", -1.0)
    return start_epoch, best_f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    losses = []
    all_preds = []
    all_labels = []

    for audios, labels in tqdm(loader, desc="valid", leave=False):
        labels = labels.to(device, non_blocking=True).long().to(device)
        audios = audios.to(device, non_blocking=True).float().to(device)

       
        logits = model(audios)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)

        losses.append(loss.item() * labels.size(0))
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss = sum(losses) / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": val_loss,
        "acc": acc,
        "F1": macro_f1,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip_norm,
    log_every,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    all_preds = []
    all_labels = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)

    for batch_idx, (audios, labels) in pbar:
        labels = labels.to(device, non_blocking=True).long().to(device)
        audios = audios.to(device, non_blocking=True).float().to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(audios)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        if (batch_idx + 1) % log_every == 0:
            pbar.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
            )

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    epoch_loss = total_loss / total_examples
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": epoch_loss,
        "acc": acc,
        "F1": macro_f1,
    }


def main(config_module: str):
    cfg = load_config(config_module)

    seed_everything(cfg["runtime"].get("seed", 42))

    if cfg["runtime"].get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if cfg["runtime"].get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, train_dl, val_dl = build_datasets_and_loaders(cfg)
    model, num_classes = build_model(cfg, train_ds, device)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)


    save_dir = cfg["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = cfg["paths"].get("ckpt", None)
    start_epoch, best_f1 = maybe_load_checkpoint(
        ckpt_path=ckpt_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Best F1 so far: {best_f1:.4f}")

    epochs = cfg["training"]["epochs"]
    validate_every = cfg["training"]["validate_every"]
    save_every = cfg["training"]["save_every"]
    grad_clip_norm = cfg["training"].get("grad_clip_norm", None)
    log_every = cfg["training"].get("log_every", 50)

    for epoch in range(start_epoch, epochs):
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Epoch {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
        )

        print(
            f"train | loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['acc']:.4f} "
            f"F1={train_metrics['F1']:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % validate_every == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_dl,
                device=device
            )

            print(
                f"valid | loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} "
                f"F1={val_metrics['F1']:.4f}"
            )

            if val_metrics["F1"] > best_f1:
                best_f1 = val_metrics["F1"]
                best_path = os.path.join(save_dir, "best_ckpt.pt")
                save_checkpoint(
                    save_path=best_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_f1=best_f1,
                    cfg=cfg,
                    class_map=train_ds.program_to_indices,
                )
                print(f"Saved best checkpoint to {best_path}")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            last_path = os.path.join(save_dir, "last_ckpt.pt")
            save_checkpoint(
                save_path=last_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_f1=best_f1,
                cfg=cfg,
                class_map=train_ds.program_to_indices,
            )
            print(f"Saved checkpoint to {last_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", type=str, required=True, help="Python module path, e.g. classifier_config")
    args = parser.parse_args()

    main(args.config_module)