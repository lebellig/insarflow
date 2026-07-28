"""Metrics for comparing wrapped InSAR phase images.

All functions accept tensors of arbitrary shape (e.g. ``(B, 1, H, W)`` or
``(B, H*W)``) whose values are wrapped phase in ``[0, 2π)``.  Every function
returns a Python float (scalar) so results can be logged directly.

Circular operations rely on the identity:
    angular_diff(a, b) = atan2(sin(b − a), cos(b − a))
which maps any raw difference into (−π, π], correctly handling the 2π wrap.
"""

import math
import torch


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _circular_diff(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Element-wise angular difference (pred − true) mapped to (−π, π]."""
    return torch.atan2(
        torch.sin(y_pred - y_true),
        torch.cos(y_pred - y_true),
    )


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def circular_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Circular Mean Squared Error (radians²).

    Args:
        y_true: Ground-truth wrapped phase, any shape, values in [0, 2π).
        y_pred: Predicted wrapped phase, same shape as ``y_true``.

    Returns:
        Scalar MSE of angular differences.
    """
    return _circular_diff(y_true, y_pred).pow(2).mean().item()


def circular_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Circular Mean Absolute Error (radians).

    L1 analog of :func:`circular_mse`.  More robust to large outliers /
    localised artefacts than MSE.

    Returns:
        Scalar MAE of angular differences.
    """
    return _circular_diff(y_true, y_pred).abs().mean().item()


def circular_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Circular Root Mean Squared Error (radians).

    Same unit as the phase itself, making it directly interpretable as a
    typical per-pixel angular error.

    Returns:
        Scalar RMSE of angular differences.
    """
    return math.sqrt(circular_mse(y_true, y_pred))


def phase_coherence(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean complex coherence between two wrapped-phase images.

    Computed as:

        γ = |E[exp(i · Δφ)]|  ∈ [0, 1]

    where Δφ = pred − true is the per-pixel angular difference.

    * γ = 1 → images are identical (all differences are 0).
    * γ → 0 → differences are uniformly distributed on the circle
              (predictions are uncorrelated with ground truth).

    This is the standard InSAR coherence estimator (without a spatial window),
    applied globally across the full image batch.

    Returns:
        Scalar coherence value in [0, 1].
    """
    diff = y_pred - y_true
    # Compute mean of unit phasors via real/imag parts (avoids complex dtype)
    re = torch.cos(diff).mean()
    im = torch.sin(diff).mean()
    return (re ** 2 + im ** 2).sqrt().item()


def psnr(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    max_phase: float = math.pi,
) -> float:
    """Peak Signal-to-Noise Ratio adapted for circular phase data (dB).

    Uses circular MSE as the noise term and ``max_phase`` as the dynamic
    range (default π — the maximum meaningful angular error on the torus).

    PSNR = 10 · log₁₀(max_phase² / circular_MSE)

    Higher is better.  Returns ``inf`` when the two images are identical.

    Args:
        y_true: Ground-truth wrapped phase.
        y_pred: Predicted wrapped phase.
        max_phase: Dynamic range in radians (default: π).

    Returns:
        PSNR in dB.
    """
    mse = circular_mse(y_true, y_pred)
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_phase ** 2 / mse)


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """Compute all phase metrics at once.

    Useful for logging a single dict to WandB / a results file without
    calling each metric function individually.

    Returns:
        Dictionary with keys: ``circular_mse``, ``circular_mae``,
        ``circular_rmse``, ``phase_coherence``, ``psnr``.
    """
    cmse = circular_mse(y_true, y_pred)
    return {
        "circular_mse": cmse,
        "circular_mae": circular_mae(y_true, y_pred),
        "circular_rmse": math.sqrt(cmse),
        "phase_coherence": phase_coherence(y_true, y_pred),
        "psnr": psnr(y_true, y_pred),
    }
