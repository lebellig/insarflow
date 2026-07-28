import math
import os
from omegaconf import DictConfig, OmegaConf
from typing import Callable, Any
from wandb.integration.lightning.fabric import WandbLogger

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from torchmetrics.aggregation import MeanMetric
from wandb.integration.lightning.fabric import WandbLogger
from wandb_osh.hooks import TriggerWandbSyncHook

from pathlib import Path

import logging

from insarflow.model import InSARFlow
from insarflow.metrics import compute_all_metrics
from insarflow.utils.logger import setup_logging, fabric_print

torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def get_experiment_name(cfg: DictConfig, hydra_config: dict):
    choices: dict = OmegaConf.to_container(hydra_config.runtime.choices)
    dataset = choices["dataset"]
    budget = f"Ns{cfg.n_training_samples}" if cfg.n_training_samples is not None else "NsFull"
    exp_name = f"{cfg.experiment_name}_{dataset}_{cfg.insarflow.backbone_name}_{cfg.insarflow.manifold_name}_{budget}"
    return exp_name


def get_ckpt_dir_path(save_dir: Path, exp_name: str) -> tuple[Path, Any]:
    ckpt_dir = save_dir / exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if len(ckpts) > 0:
        ckpt_path = sorted(
            ckpts,
            key=lambda x: int(x.stem),
            reverse=True,
        )[0]
        return ckpt_dir, ckpt_path
    else:
        return ckpt_dir, None


def get_wandb_logger(
    save_dir: Path, experiment_name: str, cfg: DictConfig
) -> tuple[WandbLogger, Callable]:
    run_offline = os.environ.get("WANDB_MODE", "online") == "offline"
    if run_offline:
        communication_dir = save_dir / "wandb_communication_dir"
        communication_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Wandb-osh communication dir is {communication_dir}")
        osh_callback = TriggerWandbSyncHook(communication_dir=communication_dir)
    else:

        def osh_callback():
            pass

    wandb_logger = WandbLogger(
        project="InSARFlow",
        name=experiment_name,
        save_dir=save_dir,
        offline=run_offline,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    return wandb_logger, osh_callback


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


@torch.no_grad()
def run_validation(
    fabric: Fabric,
    insarflow: InSARFlow,
    val_dl: DataLoader,
    sampling_steps: int,
    step: int,
    trigger_log: Callable,
) -> None:
    """Full pass over the val set: logs loss + all phase metrics.

    Loss is computed on every batch.  Denoising (for phase metrics) also
    runs on every batch using EMA weights.

    RMSE and PSNR are derived from the averaged MSE rather than being
    averaged directly, which is the mathematically correct aggregation.
    """
    insarflow.eval()

    accum_loss      = 0.0
    # Only MSE, MAE and coherence are linearly summable across batches.
    accum_mse       = 0.0
    accum_mae       = 0.0
    accum_coherence = 0.0
    num_batches     = 0

    for batch in val_dl:
        x0, x1 = batch["x0"].float(), batch["x1"].float()

        # Loss (flow-matching objective)
        accum_loss += insarflow.loss(x0, x1).item()

        # Denoising + phase metrics
        x1_gen = insarflow.denoise(
            x0, steps=sampling_steps, method="midpoint", use_ema=True
        )
        m = compute_all_metrics(x1, x1_gen)
        accum_mse       += m["circular_mse"]
        accum_mae       += m["circular_mae"]
        accum_coherence += m["phase_coherence"]
        num_batches     += 1

    insarflow.train()

    if num_batches == 0:
        return

    avg_loss      = accum_loss      / num_batches
    avg_mse       = accum_mse       / num_batches
    avg_mae       = accum_mae       / num_batches
    avg_coherence = accum_coherence / num_batches
    avg_rmse      = math.sqrt(avg_mse)
    avg_psnr      = (
        10.0 * math.log10(math.pi ** 2 / avg_mse) if avg_mse > 0 else float("inf")
    )

    fabric.print(
        "| iter {:6d} | VAL loss {:.4f}"
        " | rmse {:.4f} rad"
        " | coherence {:.4f}"
        " | psnr {:.2f} dB".format(
            step, avg_loss, avg_rmse, avg_coherence, avg_psnr
        )
    )

    logs = {
        "val/loss":            avg_loss,
        "val/circular_mse":    avg_mse,
        "val/circular_mae":    avg_mae,
        "val/circular_rmse":   avg_rmse,
        "val/phase_coherence": avg_coherence,
        "val/psnr":            avg_psnr,
    }
    for key, value in logs.items():
        fabric.log(key, value, step=step)

    trigger_log()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    experiment_name = get_experiment_name(cfg, hydra_cfg)
    save_dir = Path(cfg.save_dir)
    ckpt_dir, ckpt_path = get_ckpt_dir_path(save_dir, experiment_name)
    wandb_logger, trigger_log = get_wandb_logger(save_dir, experiment_name, cfg)

    ######################################################
    # FABRIC
    fabric: Fabric = instantiate(
        cfg.fabric,
        loggers=wandb_logger,
        devices=int(os.getenv("SLURM_GPUS_ON_NODE", torch.cuda.device_count())),
        num_nodes=int(os.getenv("SLURM_NNODES", 1)),
    )

    fabric.seed_everything(cfg.seed)
    fabric.launch()
    setup_logging(rank=fabric.global_rank)
    fabric_print(fabric)
    fabric.print(f"Launch {experiment_name} experiment")

    ######################################################
    # DATASET
    train_ds = instantiate(cfg.dataset.train)
    if cfg.n_training_samples is not None:
        n = min(cfg.n_training_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        fabric.print(f"Training budget: {n} samples (out of {len(instantiate(cfg.dataset.train))})")
    train_dl = DataLoader(train_ds, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_ds = instantiate(cfg.dataset.val)
    val_dl = DataLoader(val_ds, batch_size=cfg.dataset.batch_size, shuffle=False)

    test_ds = instantiate(cfg.dataset.test)
    test_dl = DataLoader(test_ds, batch_size=cfg.dataset.batch_size, shuffle=False)

    fabric.print(
        f"Train {len(train_dl)} batches"
        f" | Val {len(val_dl)} batches ({len(val_ds)} samples)"
        f" | Test {len(test_dl)} batches"
    )
    train_dl, val_dl, test_dl = fabric.setup_dataloaders(train_dl, val_dl, test_dl)
    train_dl = infiniteloop(train_dl)
    test_dl  = infiniteloop(test_dl)
    # val_dl is kept as a plain (finite) dataloader — iterated fully each val step
    fabric.print("Datasets initialized")

    ######################################################
    # INSARFLOW
    if ckpt_path is None:
        insarflow: InSARFlow = instantiate(cfg.insarflow)
    else:
        insarflow, ckpt = InSARFlow.from_checkpoint(ckpt_path)

    ######################################################
    # OPTIM
    optimizer = torch.optim.Adam(insarflow.parameters(), lr=cfg.lr)

    def warmup_lr(step):
        return min(step, cfg.warmup_steps) / cfg.warmup_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    ######################################################
    # TRAINING

    insarflow, optimizer = fabric.setup(insarflow, optimizer)
    if ckpt_path is not None:
        step = insarflow.load_training_state(ckpt, optimizer, scheduler)
        fabric.print(f"Load checkpoint at {ckpt_path}")
    else:
        step = 0

    interval_loss = MeanMetric()
    fabric.to_device(interval_loss)
    fabric.barrier()
    fabric.print("Start training")

    while step < cfg.max_training_steps + 1:
        optimizer.zero_grad()
        batch = next(train_dl)
        x0, x1 = batch["x0"].float(), batch["x1"].float()

        loss = insarflow.loss(x0, x1)
        interval_loss.update(loss.item())

        fabric.backward(loss)
        fabric.clip_gradients(insarflow, optimizer, clip_val=cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        insarflow.update_ema()

        # --- Training loss log ---
        if not (step + 1) % cfg.print_n_steps:
            interval_loss_value = interval_loss.compute()
            interval_loss.reset()
            fabric.print(
                "| iter {:6d} | loss {:8.3f} ".format(step + 1, interval_loss_value)
            )
            fabric.log("train/loss", interval_loss_value, step=step + 1)
            fabric.log("trainer/lr", scheduler.get_last_lr()[0])
            trigger_log()

        # --- Checkpoint ---
        if not step % cfg.save_n_steps:
            if fabric.is_global_zero:
                insarflow.save_checkpoint(
                    ckpt_dir / Path(f"{step}.ckpt"),
                    step=step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        # --- Validation: full pass over val set ---
        if not (step + 1) % cfg.val_every_n_steps:
            run_validation(
                fabric=fabric,
                insarflow=insarflow,
                val_dl=val_dl,
                sampling_steps=cfg.sampling_steps,
                step=step + 1,
                trigger_log=trigger_log,
            )

        # --- Sample images from test set ---
        if not (step + 1) % cfg.sample_every_n_steps:
            batch = next(test_dl)
            x0, x1 = batch["x0"][:cfg.n_samples].float(), batch["x1"][:cfg.n_samples].float()
            x1gen = insarflow.denoise(x0)
            plot_gen(
                fabric=fabric,
                img_size=cfg.dataset.img_size,
                x0=x0,
                x1_gen=x1gen,
                x1=x1,
            )

        step += 1


def plot_gen(
    fabric: Fabric,
    img_size: int,
    x0: torch.Tensor,
    x1_gen: torch.Tensor,
    x1: torch.Tensor,
) -> None:
    x1gen = rearrange(x1_gen, "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)
    x1    = rearrange(x1,     "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)
    x0    = rearrange(x0,     "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)

    concatx = torch.cat([x0, x1gen, x1], dim=-2)
    normalized_batch = concatx / (2 * np.pi)
    colored_images = np.stack(
        [plt.cm.twilight(img.numpy())[0, :, :, :3] for img in normalized_batch.cpu()]
    )
    colored_images_tensor = torch.tensor(colored_images).permute(0, 3, 1, 2)
    grid_image = make_grid(colored_images_tensor, nrow=4, pad_value=1)
    grid_image_np = grid_image.permute(1, 2, 0).numpy()
    fabric.log("model", wandb.Image(grid_image_np))


if __name__ == "__main__":
    main()
