import math
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from hydra.utils import instantiate

from insarflow.model import InSARFlow
from insarflow.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_gen(
    img_size: int,
    x0: torch.Tensor,
    x1_gen: torch.Tensor,
    x1: torch.Tensor,
    save_path: str,
) -> None:
    x1gen = rearrange(x1_gen, "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)
    x1    = rearrange(x1,     "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)
    x0    = rearrange(x0,     "b (c h w) -> b c h w", c=1, h=img_size, w=img_size)

    concatx = torch.cat([x0, x1gen, x1], dim=-2)
    normalized_batch = concatx / (2 * np.pi)
    colored_images = np.stack(
        [plt.cm.twilight(img.float().numpy())[0, :, :, :3] for img in normalized_batch.cpu()]
    )
    colored_images_tensor = torch.tensor(colored_images).permute(0, 3, 1, 2)
    grid_image = make_grid(colored_images_tensor, nrow=4, pad_value=1)
    grid_image_np = grid_image.permute(1, 2, 0).numpy()
    plt.imsave(save_path, grid_image_np)


def kl_divergence_histograms(
    x1: torch.Tensor,
    x1_gen: torch.Tensor,
    n_bins: int = 256,
    eps: float = 1e-10,
) -> float:
    """Mean KL divergence D_KL(P_true || P_gen) over the batch.

    Both tensors are expected to contain phase values in [0, 2π].
    For each sample a histogram is estimated; a small epsilon is added
    to avoid log(0).
    """
    x1_np     = x1.detach().float().cpu().numpy()      # (B, ...)
    x1_gen_np = x1_gen.detach().float().cpu().numpy()  # (B, ...)
    kl_vals = []
    for p_flat, q_flat in zip(x1_np.reshape(len(x1_np), -1),
                               x1_gen_np.reshape(len(x1_gen_np), -1)):
        p_hist, bin_edges = np.histogram(p_flat, bins=n_bins, range=(0.0, 2 * np.pi), density=True)
        q_hist, _         = np.histogram(q_flat, bins=bin_edges,                       density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        p = p_hist * bin_width + eps
        q = q_hist * bin_width + eps
        p /= p.sum()
        q /= q.sum()
        kl_vals.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(kl_vals))


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    print(
        f"  {label:<20s}"
        f"  circ_mse={metrics['circular_mse']:.6f}"
        f"  circ_mae={metrics['circular_mae']:.6f}"
        f"  circ_rmse={metrics['circular_rmse']:.4f} rad"
        f"  coherence={metrics['phase_coherence']:.4f}"
        f"  psnr={metrics['psnr']:.2f} dB"
        f"  kl_div={metrics['kl_divergence']:.6f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    print("Experiment name:", cfg.name)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    run = wandb.init(
        project="InSARFlow",
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=["inference"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"Loading InSARFlow model from {cfg.checkpoint} onto {cfg.device}...")
    model, checkpoint_dict = InSARFlow.from_checkpoint(cfg.checkpoint, device=cfg.device)
    model = model.bfloat16()  # match training precision
    model.eval()

    img_size = checkpoint_dict.get("config", {}).get("img_size", 256)
    print("Image size:", img_size)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Instantiating dataset from config...")
    test_ds = instantiate(cfg.dataset.test, img_size=img_size)
    test_dl = DataLoader(test_ds, batch_size=cfg.dataset.batch_size, shuffle=False)

    out_dir = os.path.join(cfg.output_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference loop
    # Accumulate MSE, MAE and coherence (all linearly summable).
    # RMSE and PSNR are re-derived from avg_mse at the end so they are
    # mathematically correct (averaging RMSE / PSNR directly would not be).
    # ------------------------------------------------------------------
    accum = {"circular_mse": 0.0, "circular_mae": 0.0, "phase_coherence": 0.0, "kl_divergence": 0.0}
    num_batches = 0

    print("Starting inference...")
    for i, batch in enumerate(tqdm(test_dl, desc="Running Inference")):
        x0 = batch["x0"].bfloat16().to(cfg.device)
        x1 = batch["x1"].bfloat16().to(cfg.device)

        with torch.no_grad():
            x1_gen = model.denoise(
                x0, steps=cfg.n_sampling_steps, method="midpoint", use_ema=True
            )

        batch_metrics = compute_all_metrics(x1, x1_gen)
        batch_metrics["kl_divergence"] = kl_divergence_histograms(x1, x1_gen)

        # Per-batch wandb logging
        wandb.log(
            {f"batch/{k}": v for k, v in batch_metrics.items()},
            step=i,
        )

        # Accumulate the linearly-summable quantities
        for k in accum:
            accum[k] += batch_metrics[k]
        num_batches += 1

        # Save grid image
        save_path = os.path.join(out_dir, f"{cfg.name}_batch_{i}.png")
        plot_gen(img_size, x0, x1_gen, x1, save_path)

    # ------------------------------------------------------------------
    # Final averages
    # ------------------------------------------------------------------
    if num_batches == 0:
        print("No batches processed.")
        wandb.finish()
        return

    avg_mse       = accum["circular_mse"]       / num_batches
    avg_mae       = accum["circular_mae"]        / num_batches
    avg_coherence = accum["phase_coherence"]     / num_batches
    avg_kl_div    = accum["kl_divergence"]       / num_batches
    avg_rmse      = math.sqrt(avg_mse)
    avg_psnr      = (
        10.0 * math.log10(math.pi ** 2 / avg_mse) if avg_mse > 0 else float("inf")
    )

    final_metrics = {
        "circular_mse":    avg_mse,
        "circular_mae":    avg_mae,
        "circular_rmse":   avg_rmse,
        "phase_coherence": avg_coherence,
        "psnr":            avg_psnr,
        "kl_divergence":   avg_kl_div,
    }

    # Log final averages to wandb
    wandb.log({f"test/{k}": v for k, v in final_metrics.items()})

    # Print summary
    print(f"\n--- Inference Results ({num_batches} batches) ---")
    _print_metrics("average", final_metrics)

    # Save results to disk
    results_path = os.path.join(out_dir, f"{cfg.name}_metrics.txt")
    with open(results_path, "w") as f:
        f.write(f"Inference results for: {cfg.name}\n")
        f.write(f"Checkpoint:            {cfg.checkpoint}\n")
        f.write(f"Batches evaluated:     {num_batches}\n\n")
        for k, v in final_metrics.items():
            f.write(f"{k:<20s}: {v:.6f}\n")
    print(f"Metrics saved to {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
