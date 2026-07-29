# InSARFlow

**Riemannian Flow Matching for InSAR Denoising and Generation**

`InSARFlow` implements generative models for Interferometric Synthetic Aperture Radar (InSAR)
phase data. It runs **flow matching on the flat torus** T^D = [0, 2π)^D, which handles the
inherent 2π-periodicity of SAR phase natively instead of treating it as an unconstrained
real-valued signal.

---

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:lebellig/insarflow.git
cd insarflow
uv sync                 # creates the virtual environment
uv sync --extra dev     # with development extras
```

`uv sync` picks the right PyTorch build automatically: the `pytorch-cu128` index (CUDA 12.8)
on Linux/Windows, PyPI (CPU + MPS) on macOS. This is driven by the `sys_platform` markers on
`torch` / `torchvision` in `[tool.uv.sources]`; to force CPU-only on Linux or Windows, point
them at `https://download.pytorch.org/whl/cpu`.

---

## Downloading the checkpoints

`ckpt/` is git-ignored, so the pretrained weights are not distributed with the repository. They
live in a [shared Google Drive folder](https://drive.google.com/drive/folders/14O9JmA2hXnp62npSp_I89TL9bHc3eNfS?usp=sharing)
instead. Both files are ~2 GB (~4 GB total).

Download `mexico.ckpt` and `simulation.ckpt` and place them in a `ckpt/` directory at the
repository root:

```
insarflow/
└── ckpt/
    ├── mexico.ckpt        # trained on real Mexico interferograms
    └── simulation.ckpt    # trained on synthetic data
```

These are the paths the code and the demo notebook expect.

---

## Loading a checkpoint

`InSARFlow.from_checkpoint` rebuilds the model from the config stored inside the `.ckpt`
(backbone, manifold, image size) and loads the weights — you don't need to know the
architecture up front. `denoise` then solves the flow-matching ODE on the flat torus,
starting from the noisy phase.

```python
import numpy as np
import torch

from insarflow.data import MexicoDataset
from insarflow.model import InSARFlow
from insarflow.utils.logger import show

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMG_SIZE, N_SAMPLES, N_STEPS = 256, 3, 5

model, _ = InSARFlow.from_checkpoint("ckpt/mexico.ckpt", device=DEVICE)
model.eval()

dataset = MexicoDataset(split="test", root_dir="data/mexico", img_size=IMG_SIZE)
raw = torch.stack([torch.as_tensor(np.float32(dataset[i]["x0"])) for i in range(N_SAMPLES)])
clean = torch.stack([torch.as_tensor(np.float32(dataset[i]["x1"])) for i in range(N_SAMPLES)])

denoised = model.denoise(raw.to(DEVICE), steps=N_STEPS, method="midpoint", use_ema=True)

# One row per sample: noisy input → model output → ground truth
show(raw, denoised, clean, "Mexico", img_size=IMG_SIZE)
```

Because the architecture comes from the checkpoint, a model trained on one domain can be
applied to another without any reconfiguration — e.g. denoising real Mexico interferograms
with a model trained only on synthetic data:

```python
simulation_model, _ = InSARFlow.from_checkpoint("ckpt/simulation.ckpt", device=DEVICE)
simulation_model.eval()

crossdomain = simulation_model.denoise(raw.to(DEVICE), steps=N_STEPS, method="midpoint", use_ema=True)
show(raw, crossdomain, clean, "Cross-Domain", img_size=IMG_SIZE)
```

See [`demo.ipynb`](demo.ipynb) for the full runnable version covering both datasets.
Note that `ckpt/` and `data/` are git-ignored: the checkpoints come from the Drive folder above,
and the datasets are not distributed with the repository.

### Saving and resuming

```python
model.save_checkpoint("checkpoint.ckpt", step=1000, optimizer=optimizer, scheduler=scheduler)

model, checkpoint = InSARFlow.from_checkpoint("checkpoint.ckpt", device="cuda")
step = model.load_training_state(checkpoint, optimizer, scheduler)
```

---

## Training and inference from the CLI

Both entry points are configured with [Hydra](https://hydra.cc/); configs live in `configs/`.
The `dataset` group selects the domain and carries all dataset-specific values — `img_size`,
`batch_size`, and the dataset targets — accessed as `cfg.dataset.*`. Note the two entry points
default differently: **`train.yaml` defaults to `mexico`, `inference.yaml` to `simulation`.**
Either can be overridden with `dataset=`.

```bash
# Training  (mexico by default)
uv run python -m insarflow.train
uv run python -m insarflow.train dataset=simulation

# Inference  (simulation by default)
uv run python -m insarflow.inference \
  name=my_experiment \
  checkpoint=/path/to/checkpoint.ckpt

uv run python -m insarflow.inference \
  dataset=mexico \
  name=my_experiment \
  checkpoint=/path/to/checkpoint.ckpt
```

The package also installs `insarflow-train` and `insarflow-inference` console scripts, which are
equivalent to the `python -m` invocations above.

Key training parameters (in `configs/train.yaml`):

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Learning rate | `lr` | `1e-4` |
| EMA decay | `ema_decay` | `0.999` |
| Gradient clipping | `grad_clip` | `0.5` |
| LR warmup steps | `warmup_steps` | `500` |
| Total steps | `max_training_steps` | `100000` |
| ODE sampling steps | `sampling_steps` | `50` |

(`configs/inference.yaml` spells the same knob `n_sampling_steps`.)

Both dataset groups currently use `img_size: 256` and `batch_size: 4`; they differ only in the
dataset class they instantiate (`MexicoDataset` vs. `SimulationInSARDataset`) and the data roots.

---

## How it works

Noisy phase, shaped `[batch, H×W]` and valued in `[0, 2π)`, is projected onto the torus with
`expmap`, reshaped to `[batch, 1, H, W]`, passed through the backbone to predict the velocity
field v_θ(x_t, t), projected back onto the tangent space with `proju`, and flattened again.

- **Training:** `L = E‖ v_θ(x_t, t) − ẋ_t ‖²` (flow-matching L2 loss).
- **Inference:** a Riemannian ODE solve x₀ → x₁, with `euler`, `midpoint`, or `rk4` steps.

The flat torus makes the manifold operations trivial to compute:

| Operation | Formula |
|-----------|---------|
| `expmap(x, u)` | `(x + u) mod 2π` |
| `logmap(x, y)` | `atan2(sin(y−x), cos(y−x))` |
| `projx(x)` | `x mod 2π` |
| `proju(x, u)` | `u` |

`GeodesicProbPath` interpolates between source x₀ and target x₁ along torus geodesics,
parameterized by `CondOTScheduler` (α_t = t, σ_t = 1−t).

### Backbones

Selected via the `backbone_name` config key. The FCDM family (`insarflow/models/fcdm.py`) is a
fully convolutional U-Net with ConvNeXt blocks and adaLN-Zero timestep conditioning — being
fully convolutional, it has no `img_size` dependency.

| `backbone_name` | Parameters | `hidden_size` | `depth` |
|----------------|-----------|--------------|---------|
| `FCDMSmall` | ~32M | 128 | `[2,4,8,4,2]` |
| `FCDMBase` — **default** | ~126M | 256 | `[2,4,8,4,2]` |
| `FCDMLarge` | ~501M | 512 | `[2,4,8,4,2]` |
| `FCDMXLarge` | ~695M | 512 | `[3,6,12,6,3]` |

(Counts measured at `in_channels=1, out_channels=1`.)

---

## Datasets

Both datasets take `split="train" | "val" | "test"` and carve out a fixed 100-sample validation
set so the split is identical across runs.

**`SimulationInSARDataset`** — synthetic InSAR. The last 10,000 files of the sorted list are the
test split; `val` is the first 100 of what remains, `train` the rest.

```
root_dir/
├── interf/          # Noisy wrapped interferograms (*.tif)
└── originWrapped/   # Clean wrapped phase (*.tif)
```

**`MexicoDataset`** — real InSAR from Mexico. Here the test split is a separate directory rather
than a slice: `train` and `val` both read the `train/` subdirectories (`val` taking the first 100
files), while `test` reads the `test/` subdirectories. Files are shuffled with a fixed seed of 42.

```
root_dir/
├── raw/{train,test}/           # Noisy wrapped interferogram patches (*.tif)
├── clean/{train,test}/         # Clean wrapped interferogram patches (*.tif)
├── raw_image/{train,test}/     # Noisy SAR image patches — complex64 (*.npy)
├── clean_image/{train,test}/   # Clean SAR image patches — complex64 (*.npy)
└── metadata/{train,test}/      # Per-patch metadata (*.txt), contains time_diff
```

For `val` and `test`, `__getitem__` also returns the SAR images and the temporal baseline.

---

## License

Flow matching utilities under `insarflow/flow_matching/` are derived from work
Copyright (c) Meta Platforms, Inc. and affiliates, licensed under CC-BY-NC.
The InSAR-specific code is provided under **CC-BY-NC-4.0**.

---

## Project structure

```
insarflow/
├── pyproject.toml                          # Package metadata and dependencies (UV/Hatchling)
├── .python-version                         # Python version pin (3.11)
├── README.md                               # This file
├── demo.ipynb                              # Denoising demo on both datasets + cross-domain
│
├── configs/                                # Hydra configuration files
│   ├── train.yaml                          # Training config (defaults to Mexico dataset)
│   ├── inference.yaml                      # Inference config (defaults to Simulation dataset)
│   └── dataset/                            # Dataset config group
│       ├── mexico.yaml                     # Mexico dataset (real InSAR, 256×256, batch 4)
│       └── simulation.yaml                 # Simulation dataset (synthetic InSAR, 256×256, batch 4)
│
└── insarflow/                              # Main Python package
    ├── __init__.py                         # Public API: InSARFlow, EMAModel
    ├── model.py                            # Core model: InSARFlow + EMAModel + registries
    ├── train.py                            # Training entry point (Hydra + Lightning Fabric)
    ├── inference.py                        # Inference entry point (Hydra)
    ├── metrics.py                          # Circular MSE/MAE/RMSE, phase coherence, PSNR
    │
    ├── models/                             # Neural network backbones
    │   ├── __init__.py                     # Exports: FCDMSmall / Base / Large / XLarge
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
    │   │       ├── __init__.py             # Exports: Scheduler, ConvexScheduler, CondOTScheduler, SchedulerOutput
    │   │       └── scheduler.py            # Scheduler, ConvexScheduler, CondOTScheduler
    │   │
    │   ├── solver/                         # ODE solvers
    │   │   ├── __init__.py                 # Exports: Solver, RiemannianODESolver
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
    │   ├── __init__.py                     # Exports: SimulationInSARDataset, MexicoDataset
    │   ├── simulation.py                   # SimulationInSARDataset (synthetic InSAR)
    │   ├── mexico.py                       # MexicoDataset (real InSAR, TIFF format)
    │   └── build.py                        # Standalone script: slices full-image .npy into patches
    │
    ├── evaluation/                         # Baseline comparisons
    │   └── baselines/phinet/               # PhiNet baseline: test_demo.py, visualize_results.py
    │
    └── utils/                              # Project utilities
        ├── __init__.py                     # Exports: setup_logging, fabric_print
        └── logger.py                       # setup_logging(), fabric_print(), show()
```
