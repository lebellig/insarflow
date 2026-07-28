# InSARFlow

**Riemannian Flow Matching for InSAR Denoising and Generation**

`InSARFlow` is a Python library implementing generative models for Interferometric Synthetic Aperture Radar (InSAR) phase data. It uses **Riemannian flow matching** on the flat torus to handle the inherent 2π-periodicity of SAR phase signals.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone git@github.com:lebellig/insarflow.git
cd insarflow

# Install with uv (creates a virtual environment automatically)
uv sync

# For development extras
uv sync --extra dev

# Or install as a package in another project
uv add insarflow
```

`uv sync` picks the right PyTorch build for the machine it runs on, with no extra flags:

| Platform | Source | Build |
|---|---|---|
| Linux / Windows | `pytorch-cu128` index | CUDA 12.8 |
| macOS (or any other platform) | PyPI | CPU + MPS |

This is driven by the `sys_platform` markers on `torch` / `torchvision` in
`[tool.uv.sources]`. To force a CPU-only install on Linux or Windows, point those markers
at `https://download.pytorch.org/whl/cpu` instead of the `cu128` index.

---

## Quick Start

```python
import torch
from insarflow import InSARFlow

# Initialize model
model = InSARFlow(img_size=128)

# Denoise a batch of InSAR phase images
# x0: noisy input, shape (batch, H*W), values in [0, 2π)
x0 = torch.rand(4, 128 * 128) * 2 * torch.pi
x1_denoised = model.denoise(x0, steps=50, method="midpoint", use_ema=True)

# Training
x1_clean = torch.rand(4, 128 * 128) * 2 * torch.pi
loss = model.loss(x0, x1_clean)
loss.backward()
```

### Training from the CLI

```bash
# Train on Mexico dataset (default)
uv run python -m insarflow.train

# Train on simulation dataset
uv run python -m insarflow.train dataset=simulation
```

### Running inference

```bash
# Inference on Mexico dataset (default)
uv run python -m insarflow.inference \
  name=my_experiment \
  checkpoint=/path/to/checkpoint.ckpt

# Inference on simulation dataset
uv run python -m insarflow.inference \
  dataset=simulation \
  name=my_experiment \
  checkpoint=/path/to/checkpoint.ckpt
```

---

## Project Structure

```
insarflow/
├── pyproject.toml                          # Package metadata and dependencies (UV/Hatchling)
├── .python-version                         # Python version pin (3.11)
├── README.md                               # This file
│
├── configs/                                # Hydra configuration files
│   ├── train.yaml                          # Training config (defaults to Mexico dataset)
│   ├── inference.yaml                      # Inference config (defaults to Mexico dataset)
│   └── dataset/                            # Dataset config group
│       ├── mexico.yaml                     # Mexico dataset (real InSAR, 128×128, batch 16)
│       └── simulation.yaml                 # Simulation dataset (synthetic InSAR, 256×256, batch 4)
│
└── insarflow/                                # Main Python package
    ├── __init__.py                         # Public API: InSARFlow, EMAModel
    ├── model.py                            # Core model: InSARFlow + EMAModel
    ├── train.py                            # Training entry point (Hydra + Lightning Fabric)
    ├── inference.py                        # Inference entry point (Hydra)
    │
    ├── models/                             # Neural network backbones
    │   ├── __init__.py
    │   └── fcdm.py                         # FCDM variants (Small / Base / Large / XLarge)
    │
    ├── flow_matching/                      # Riemannian flow matching framework
    │   ├── __init__.py
    │   │
    │   ├── path/                           # Probability paths
    │   │   ├── __init__.py                 # Exports: ProbPath, GeodesicProbPath, PathSample
    │   │   ├── path.py                     # Abstract ProbPath base class
    │   │   ├── path_sample.py              # PathSample dataclass
    │   │   ├── geodesic.py                 # GeodesicProbPath (Riemannian geodesic interpolation)
    │   │   └── scheduler/
    │   │       ├── __init__.py             # Exports all schedulers
    │   │       └── scheduler.py            # Scheduler, ConvexScheduler, CondOTScheduler
    │   │
    │   ├── solver/                         # ODE solvers
    │   │   ├── __init__.py                 # Exports: RiemannianODESolver, Solver
    │   │   ├── solver.py                   # Abstract Solver base class
    │   │   └── riemannian_ode_solver.py    # RiemannianODESolver (Euler / midpoint / RK4 on manifold)
    │   │
    │   └── utils/                          # Utility functions and manifolds
    │       ├── __init__.py                 # Exports: expand_tensor_like, ModelWrapper
    │       ├── utils.py                    # expand_tensor_like
    │       ├── model_wrapper.py            # ModelWrapper abstract class
    │       └── manifolds/
    │           ├── __init__.py             # Exports: Manifold, Euclidean, FlatTorus, geodesic
    │           ├── manifold.py             # Abstract Manifold + Euclidean
    │           ├── torus.py                # FlatTorus: [0, 2π]^D (main manifold for InSAR)
    │           └── utils.py                # geodesic() path generator
    │
    ├── data/                               # Dataset classes
    │   ├── __init__.py
    │   ├── simulation.py                   # SimulationInSARDataset (synthetic InSAR)
    │   └── mexico.py                       # MexicoDataset (real InSAR, TIFF format)
    │
    └── utils/                              # Project utilities
        ├── __init__.py
        └── logger.py                       # setup_logging(), fabric_print()
```

---

## Architecture

### InSARFlow Model (`insarflow/model.py`)

The core model connects all components:

```
Noisy InSAR phase input  [batch, H×W]  ∈ [0, 2π)
          │
          ▼
  FlatTorus.expmap()      — project onto torus manifold
          │
          ▼
  Reshape to image        [batch, 1, H, W]
          │
          ▼
  FCDM backbone           — predict velocity field v_θ(x_t, t)
          │
          ▼
  FlatTorus.proju()       — project v onto tangent space
          │
          ▼
  Reshape to flat         [batch, H×W]

Training loss:  L = E‖ v_θ(x_t, t) − ẋ_t ‖²   (flow matching L2 loss)
Inference:      Riemannian ODE solve  x_0 → x_1  (Euler / midpoint / RK4)
```

### Flow Matching on the Flat Torus

InSAR phase data is 2π-periodic, making it naturally suited to the **flat torus** T^D = [0, 2π)^D. The key operations are:

| Operation | Formula |
|-----------|---------|
| `expmap(x, u)` | `(x + u) mod 2π` |
| `logmap(x, y)` | `atan2(sin(y−x), cos(y−x))` |
| `projx(x)` | `x mod 2π` |
| `proju(x, u)` | `u` (flat torus has trivial tangent projection) |

**GeodesicProbPath** interpolates between source x₀ and target x₁ using geodesics on the torus, parameterized by a `CondOTScheduler` (α_t = t, σ_t = 1−t).

### Model Backbones

Backbones are selected via the `backbone_name` config key.

#### FCDM family (`models/fcdm.py`)

Fully Convolutional U-Net with ConvNeXt blocks and adaLN-Zero timestep conditioning. Fully convolutional — no `img_size` dependency.

| Class | `backbone_name` | Parameters | `hidden_size` | `depth` |
|-------|----------------|-----------|--------------|---------|
| `FCDMSmall` | `"FCDMSmall"` | ~10M | 128 | `[2,4,8,4,2]` |
| `FCDMBase` | `"FCDMBase"` | ~40M | 256 | `[2,4,8,4,2]` — **default** |
| `FCDMLarge` | `"FCDMLarge"` | ~155M | 512 | `[2,4,8,4,2]` |
| `FCDMXLarge` | `"FCDMXLarge"` | ~230M | 512 | `[3,6,12,6,3]` |

To switch backbone, set `backbone_name` in `train.yaml`:

```yaml
insarflow:
  _target_: insarflow.model.InSARFlow
  img_size: ${dataset.img_size}
  backbone_name: FCDMBase   # or FCDMSmall, FCDMLarge, FCDMXLarge
```

---

## Schedulers

| Scheduler | α_t | σ_t |
|-----------|-----|-----|
| `CondOTScheduler` | t | 1−t |

---

## ODE Solvers

**`RiemannianODESolver`** — manifold-aware solver with three methods:

| Method | Description |
|--------|-------------|
| `euler` | First-order Euler step with manifold projection |
| `midpoint` | Second-order midpoint method |
| `rk4` | Fourth-order Runge-Kutta |

---

## Datasets

### `SimulationInSARDataset`

Synthetic InSAR data. Expected directory structure:

```
root_dir/
├── interf/          # Noisy wrapped interferograms (*.tif)
└── originWrapped/   # Clean wrapped phase (*.tif)
```

Last 10,000 samples are used as the test split.

### `MexicoDataset`

Real InSAR data from Mexico. Expected directory structure:

```
root_dir/
├── raw/
│   ├── train/       # Noisy phase patches (*.tif)
│   └── test/
└── clean/
    ├── train/       # Clean phase patches (*.tif)
    └── test/
```

Test region is defined spatially: `500 < x < 750` and `1500 < y < 17500`.

---

## Configuration

Training and inference are configured via [Hydra](https://hydra.cc/). All YAML configs are in `configs/`.

### Config structure

```
configs/
├── train.yaml          defaults: [dataset: mexico]
├── inference.yaml      defaults: [dataset: mexico]
└── dataset/
    ├── mexico.yaml     # @package _global_.dataset  →  cfg.dataset.*
    └── simulation.yaml # @package _global_.dataset  →  cfg.dataset.*
```

The `dataset` config group is selected at the command line with `dataset=mexico` (default) or `dataset=simulation`. All dataset-specific values — `img_size`, `batch_size`, and dataset instantiation targets — are defined in the group config and accessed as `cfg.dataset.*`.

### Key training parameters

| Parameter | Config key | Default | Description |
|-----------|-----------|---------|-------------|
| Learning rate | `lr` | `1e-4` | Adam learning rate |
| EMA decay | `ema_decay` | `0.999` | EMA weight decay |
| Gradient clipping | `grad_clip` | `0.5` | Max gradient norm |
| Warmup | `warmup_steps` | `1000` | LR warmup steps |
| Total steps | `max_training_steps` | `100000` | Training budget |
| ODE steps | `sampling_steps` | `50` | Steps at inference |

### Dataset parameters (in `configs/dataset/`)

| Parameter | Config key | Mexico | Simulation |
|-----------|-----------|--------|------------|
| Image size | `dataset.img_size` | `128` | `256` |
| Batch size | `dataset.batch_size` | `16` | `4` |
| Train dataset | `dataset.train` | `MexicoDataset` | `SimulationInSARDataset` |
| Test dataset | `dataset.test` | `MexicoDataset` | `SimulationInSARDataset` |

---

## Checkpointing

```python
# Save
model.save_checkpoint("checkpoint.ckpt", step=1000, optimizer=optimizer, scheduler=scheduler)

# Load
model, checkpoint = InSARFlow.from_checkpoint("checkpoint.ckpt", device="cuda")
step = model.load_training_state(checkpoint, optimizer, scheduler)
```

---

## Dependencies

Core runtime dependencies (managed by `uv`):

- `torch >= 2.6.0` — PyTorch
- `lightning >= 2.5.1` — Lightning Fabric for distributed training
- `torchdiffeq >= 1.0.6` — ODE solvers
- `einops >= 0.8.1` — Tensor rearrangements
- `hydra-core >= 1.3.2` — Configuration management
- `wandb >= 0.19.9` — Experiment tracking
- `tifffile >= 2025.3.30` — InSAR TIFF loading
- `timm >= 1.0.15` — Vision model utilities

See `pyproject.toml` for the full list.

---

## License

Parts of this codebase are derived from:

- **Flow matching utilities** (`flow_matching/`): Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under CC-BY-NC.

The InSAR-specific code is provided under **CC-BY-NC-4.0**.
