from __future__ import annotations

import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

CMAP, NORM = "twilight_shifted", Normalize(vmin=0, vmax=2 * np.pi)

def setup_logging(rank: int):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%d/%m/%Y | %H:%M:%S",
    )


def fabric_print(fabric):
    fabric.print = lambda *args, **kwargs: (
        logging.info(" ".join(str(a) for a in args))
        if fabric.global_rank == 0
        else None
    )

def show(
    raw: np.ndarray | torch.Tensor,
    denoised: np.ndarray | torch.Tensor,
    truth: np.ndarray | torch.Tensor,
    title: str,
    img_size: int,
    cell_in: float = 2.6,
    gap_px: int = 5,
    dpi: int = 100,
):
    """One row per sample: noisy input, model output, ground truth.

    The grid is laid out in absolute inches so the panels sit exactly ``gap_px``
    apart. Each cell is square and matches the square image, so ``imshow`` fills
    it edge to edge instead of shrinking and leaving slack inside the axes.
    """
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()

    n_rows, n_cols = len(raw), 3
    gap_in = gap_px / dpi

    # Margins in inches: room for the suptitle + column titles on top, and for
    # the colorbar on the right.
    left_in, right_in = 0.10, 0.85
    bottom_in, top_in = 0.10, 0.60

    grid_w = n_cols * cell_in + (n_cols - 1) * gap_in
    grid_h = n_rows * cell_in + (n_rows - 1) * gap_in
    fig_w = left_in + grid_w + right_in
    fig_h = bottom_in + grid_h + top_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # wspace/hspace are fractions of the average cell size, so gap_in / cell_in
    # lands exactly gap_px between neighbours.
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        left=left_in / fig_w,
        right=(left_in + grid_w) / fig_w,
        bottom=bottom_in / fig_h,
        top=(bottom_in + grid_h) / fig_h,
        wspace=gap_in / cell_in,
        hspace=gap_in / cell_in,
    )

    for row in range(n_rows):
        for col, (img, label) in enumerate(
            [(raw[row], "raw"), (denoised[row], "denoised"), (truth[row], "ground truth")]
        ):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img.reshape(img_size, img_size), cmap=CMAP, norm=NORM)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=12, pad=4)

    fig.suptitle(title, fontsize=13, y=1 - 0.12 / fig_h)
    # Its own axes, so the colorbar cannot steal width from the image grid.
    cax = fig.add_axes(
        [
            (left_in + grid_w + 0.12) / fig_w,
            bottom_in / fig_h,
            0.13 / fig_w,
            grid_h / fig_h,
        ]
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, label="wrapped phase [rad]"
    )
    plt.show()