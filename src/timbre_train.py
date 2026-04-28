import importlib
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import argparse

from torch.profiler import profile, schedule, ProfilerActivity, record_function

def load_config(module_path: str):
    mod = importlib.import_module(module_path)
    return mod.CONFIG


def build_ae(cfg):
    backbone = cfg["ae"]["backbone_cls"](**cfg["ae"]["backbone_kwargs"])
    spec_class = cfg["ae"]["spec_cls"](**cfg["ae"]["spec_kwargs"])

    ae = cfg["ae"]["ae_cls"](
        backbone,
        spec_class=spec_class,
        **cfg["ae"]["ae_kwargs"],
    )

    if cfg["ae"].get("freeze_ae", False):
        for p in ae.parameters():
            p.requires_grad = False

    return ae


def build_datasets_and_loaders(cfg):
    train_ds = cfg["data"]["train_dataset_cls"](**cfg["data"]["train_dataset_kwargs"])
    val_ds = cfg["data"]["val_dataset_cls"](**cfg["data"]["val_dataset_kwargs"])

    train_dl = DataLoader(train_ds, **cfg["data"]["train_loader_kwargs"])
    val_dl = DataLoader(val_ds, **cfg["data"]["val_loader_kwargs"])

    return train_ds, val_ds, train_dl, val_dl


def build_conde(cfg, ae):
    conde_kwargs = dict(cfg["conde"]["kwargs"])
    conde_kwargs["encoder"] = ae.model.encoder
    conde_model = cfg["conde"]["cls"](**conde_kwargs)
    return conde_model


def build_losses(cfg):
    losses_cfg = cfg["losses"]

    def instantiate(name_fn, name_kwargs):
        fn = losses_cfg[name_fn]
        kwargs = losses_cfg.get(name_kwargs, {})
        if fn is None:
            return None
        return fn(**kwargs)

    return {
        "contrastive_loss": instantiate("contrastive_loss_fn", "contrastive_loss_kwargs"),
        "residual_loss": instantiate("residual_loss_fn", "residual_loss_kwargs"),
        "coverage_loss": instantiate("coverage_loss_fn", "coverage_loss_kwargs"),
        "decomp_loss": instantiate("decomp_loss_fn", "decomp_loss_kwargs"),
        "recon_loss": instantiate("recon_loss_fn", "recon_loss_kwargs"),
        "residual_reg": instantiate("residual_reg_fn", "residual_reg_kwargs"),
    }


def build_trainer(cfg, conde_model, ae, train_dl, built_losses):
    trainer_kwargs = dict(cfg["trainer"]["kwargs"])
    trainer_kwargs["contrastive_loss_fn"] = built_losses["contrastive_loss"]
    trainer_kwargs["decomp_loss_fn"] = built_losses["decomp_loss"]
    trainer_kwargs["residual_loss_fn"] = built_losses["residual_loss"]
    trainer_kwargs["coverage_loss_fn"] = built_losses["coverage_loss"]
    trainer_kwargs["recon_loss_fn"] = built_losses["recon_loss"]
    trainer_kwargs["residual_reg_fn"] = built_losses.get('residual_reg_loss', None)

    trainer_kwargs["decoder"] = ae.model.decoder
    trainer_kwargs["ae"] = ae
    trainer_kwargs["semantic_info"] = cfg["semantic_info"]
    trainer_kwargs["total_steps"] = len(train_dl) * cfg["training"]["epochs"]
    trainer_kwargs["save_dir"] = cfg["paths"]["save_dir"]

    trainer = cfg["trainer"]["cls"](conde_model, **trainer_kwargs)
    return trainer


def load_ae(cfg, ae, load_mode):

    ckpt_path = cfg["paths"].get("ckpt")
    if not ckpt_path:
        return
    
    if load_mode == 'ae':
        CKPT = torch.load(ckpt_path)
        new_CKPT = {}

        for key, value in CKPT.items():
            if not key.startswith('model.'):
                new_key = 'model.' + key
            else:
                new_key = key
            new_CKPT[new_key] = value
            
        ae.load_state_dict(new_CKPT)
    elif load_mode == 'conde_ae':
        ckpt = torch.load(ckpt_path)

        conde_sd = ckpt["conde_model"]

        encoder_sd = {
            k.replace("encoder.", "", 1): v
            for k, v in conde_sd.items()
            if k.startswith("encoder.")
        }

        ae.model.encoder.load_state_dict(encoder_sd)
        ae.model.decoder.load_state_dict(ckpt["decoder"])

def load_conde(cfg, ae, conde_model):
    load_mode = cfg['runtime'].get("load_method", 'none')

    ckpt_path = cfg["paths"].get("ckpt")
    if not ckpt_path:
        return
    

    ckpt = torch.load(ckpt_path)
    conde_model.load_state_dict(ckpt['conde_model'])
    ae.model.decoder.load_state_dict(ckpt['decoder'])

def load_opt(cfg, trainer, train_dl):

    ckpt_path = cfg["paths"].get("ckpt")
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    opt_ckpt = ckpt['optimizer']
    current_step = len(train_dl) * epoch
    trainer.load_ckpt_states(opt_ckpt, current_step)
    return epoch


def maybe_move_to_cuda(cfg, ae, trainer):
    device = cfg["runtime"]["device"]
    ae.eval()
    if device == "cuda":
        ae.cuda()
        trainer.model.cuda()
        if hasattr(trainer, "decoder") and trainer.decoder is not None:
            trainer.decoder.cuda()

def save_checkpoint(epoch, trainer, ae, ckpt_name=None):
    os.makedirs(trainer.save_dir, exist_ok=True)

    if ckpt_name is None:
        ckpt_name = f"last_ckpt.pt"

    save_path = os.path.join(trainer.save_dir, ckpt_name)

    # Adapt these attribute names if your trainer uses different ones
    state = {
        "epoch": epoch,
        "conde_model": trainer.model.state_dict(),
        "decoder": trainer.decoder.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict() if hasattr(trainer, "scaler") else None,
    }

    torch.save(state, save_path)

    storage_path = os.path.join('/storage/datasets/fernando.tonucci/ckpts', trainer.save_dir.split('/')[-1])
    torch.save(state, storage_path)
    print(f"Saved checkpoint to {save_path} and {storage_path}")


def train_loop(cfg, trainer, train_dl, val_dl, ae, cur_epoch=0):
    epochs = cfg["training"]["epochs"]
    decoder_start = cfg["training"]["decoder_start"]
    validate_every = cfg["training"]["validate_every"]
    save_every = cfg["training"]["save_every"]

    trainer.model.train_encoder = False

    for epoch in range(cur_epoch, epochs, 1):
        print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] EPOCH {epoch}/{epochs}:', end=' ')

        if epoch >= decoder_start:
            trainer.model.train_encoder = True

        for batch in train_dl:
            stats = trainer.train_step(batch, skip_decoder=epoch < decoder_start)

        trainer.on_epoch_end()

        if epoch % validate_every == 0:
            n_show = 3
            for val_batch in val_dl:
                trainer.validation_step(val_batch, n_show=n_show, epoch=epoch)
                n_show = 0
                break
            trainer.on_validation_epoch_end()

        if epoch % save_every == 0:
            save_checkpoint(epoch, trainer, ae)

'''def train_loop(cfg, trainer, train_dl, val_dl, ae):
    epochs = cfg["training"]["epochs"]
    decoder_start = cfg["training"]["decoder_start"]
    validate_every = cfg["training"]["validate_every"]
    save_every = cfg["training"]["save_every"]

    trainer.model.train_encoder = False

    prof_dir = os.path.join(cfg["paths"]["save_dir"], "profiler")
    os.makedirs(prof_dir, exist_ok=True)

    use_cuda = cfg["runtime"]["device"] == "cuda"
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    # Profile a small number of steps only
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        global_step = 0

        for epoch in range(epochs):
            print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] EPOCH {epoch}/{epochs}:', end=' ')

            if epoch >= decoder_start:
                trainer.model.train_encoder = True

            for batch_idx, batch in enumerate(tqdm(train_dl)):
                with record_function("trainer.train_step"):
                    stats = trainer.train_step(batch, skip_decoder=epoch < decoder_start)

                prof.step()
                global_step += 1

                # Stop after profiler collected enough active steps
                if global_step >= 5:
                    break

            trainer.on_epoch_end()

            if epoch % validate_every == 0:
                n_show = 3
                for val_batch in val_dl:
                    with record_function("trainer.validation_step"):
                        trainer.validation_step(val_batch, n_show=n_show, epoch=epoch)
                    n_show = 0
                    break
                trainer.on_validation_epoch_end()

            if epoch % save_every == 0:
                with record_function("save_checkpoint"):
                    save_checkpoint(epoch, trainer, ae)

            if global_step >= 5:
                break

    print("\n===== TOP CUDA OPS =====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    print("\n===== TOP CPU OPS =====")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    trace_path = os.path.join(prof_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"Profiler trace saved to: {trace_path}")'''


def main(config_module="config.timbre_slakh_ae"):
    cfg = load_config(config_module)

    torch.backends.cudnn.benchmark = cfg["runtime"]["cudnn_benchmark"]
    torch.autograd.set_detect_anomaly(cfg["runtime"]["detect_anomaly"])

    ae = build_ae(cfg)
    load_mode = cfg['runtime'].get("load_method", 'none')
    cur_epoch=0
    if load_mode == 'ae' or 'conde_ae':
        load_ae(cfg, ae, load_mode)
    train_ds, val_ds, train_dl, val_dl = build_datasets_and_loaders(cfg)
    conde_model = build_conde(cfg, ae)
    if load_mode == 'conde':
        load_conde(cfg, ae, conde_model)
    built_losses = build_losses(cfg)
    trainer = build_trainer(cfg, conde_model, ae, train_dl, built_losses)
    if load_mode == 'conde':
        cur_epoch = load_opt(cfg, trainer, train_dl)

    maybe_move_to_cuda(cfg, ae, trainer)
    train_loop(cfg, trainer, train_dl, val_dl, ae, cur_epoch=cur_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_module",
        type=str,
        default="config.timbre_slakh_ae",
        help="Path to the config module"
    )

    args = parser.parse_args()

    main(config_module=args.config_module)