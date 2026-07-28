import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Dict, Any, Tuple

from insarflow.flow_matching.utils.manifolds import Euclidean, FlatTorus
from insarflow.models.fcdm import FCDMSmall, FCDMBase, FCDMLarge, FCDMXLarge

from insarflow.flow_matching.path import GeodesicProbPath
from insarflow.flow_matching.path.scheduler import CondOTScheduler
from insarflow.flow_matching.solver import RiemannianODESolver

from copy import deepcopy
import logging

BACKBONE_REGISTRY = {
    # FCDM family (Fully Convolutional Diffusion Models)
    "FCDMSmall": FCDMSmall,
    "FCDMBase": FCDMBase,
    "FCDMLarge": FCDMLarge,
    "FCDMXLarge": FCDMXLarge,
}

MANIFOLD_REGISTRY = {
    "FlatTorus": FlatTorus,
    "Euclidean": Euclidean,
}

logger = logging.getLogger("InSARFlow")


class EMAModel(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self.averaged_model = deepcopy(model)

        for param in self.averaged_model.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module) -> None:
        """Updates the EMA weights based on the current model weights."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.averaged_model.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def forward(self, *args, **kwargs):
        return self.averaged_model(*args, **kwargs)


class InSARFlow(nn.Module):
    def __init__(
        self,
        img_size: int,
        backbone_name: str = "FCDMBase",
        backbone_class: Optional[type] = None,
        manifold_name: str = "FlatTorus",
        use_ema: bool = True,
        ema_decay: float = 0.999,
    ) -> None:
        """
        Args:
            img_size: Spatial size of the square input image.
            backbone_name: Key in BACKBONE_REGISTRY. Used when instantiating
                from a Hydra config (e.g. ``backbone_name: FCDMBase``).
                Ignored when ``backbone_class`` is provided explicitly.
            backbone_class: Backbone class to use directly (Python API).
                Overrides ``backbone_name`` when given.
            manifold_name: Key in MANIFOLD_REGISTRY (e.g. ``FlatTorus``,
                ``Euclidean``).
            use_ema: Whether to maintain an EMA copy of the velocity field.
            ema_decay: EMA decay rate.
        """
        super().__init__()
        self.img_size = img_size
        self.manifold_name = manifold_name

        if backbone_class is None:
            backbone_class = BACKBONE_REGISTRY.get(backbone_name)
            if backbone_class is None:
                raise ValueError(
                    f"Unknown backbone '{backbone_name}'. "
                    f"Available: {list(BACKBONE_REGISTRY)}"
                )

        manifold_cls = MANIFOLD_REGISTRY.get(manifold_name)
        if manifold_cls is None:
            raise ValueError(
                f"Unknown manifold '{manifold_name}'. "
                f"Available: {list(MANIFOLD_REGISTRY)}"
            )
        self.manifold = manifold_cls()
        self.path = GeodesicProbPath(
            scheduler=CondOTScheduler(), manifold=self.manifold
        )

        self.vecfield = backbone_class(img_size=img_size, in_channels=1, out_channels=1)
        self.use_ema = use_ema
        self.ema = EMAModel(self.vecfield, decay=ema_decay) if use_ema else None
        logger.info(
            f"InSARFlow initialized | Backbone: {backbone_class.__name__} | Manifold: {manifold_name} | Image Size: {img_size} | EMA: {use_ema}"
        )
        # print number of parameters (in millions)
        num_params = sum(p.numel() for p in self.vecfield.parameters())
        logger.info(f"Number of parameters: {num_params / 1e6:.2f} M")

    def update_ema(self):
        if self.use_ema and self.ema is not None:
            self.ema.update(self.vecfield)

    def save_checkpoint(self, path: str, step: int = 0, optimizer=None, scheduler=None):
        backbone_name = self.vecfield.__class__.__name__
        checkpoint = {
            "config": {
                "img_size": self.img_size,
                "backbone_name": backbone_name,
                "manifold_name": self.manifold_name,
                "use_ema": self.use_ema,
            },
            "model_state_dict": self.state_dict(),
            "step": step,
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(
            f"Checkpoint saved successfully | Path: {path} | Step: {step} | Backbone: {backbone_name} | EMA: {self.use_ema}"
        )

    @classmethod
    def from_checkpoint(
        cls, ckpt_path, device="cpu"
    ) -> Tuple["InSARFlow", Dict[str, Any]]:
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint["config"]

        backbone_name = config.get("backbone_name", "FCDMBase")
        manifold_name = config.get("manifold_name", "FlatTorus")
        use_ema = config.get("use_ema", False)

        model = cls(
            img_size=config["img_size"],
            backbone_name=backbone_name,
            manifold_name=manifold_name,
            use_ema=use_ema,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, checkpoint

    def load_training_state(
        self,
        checkpoint: Dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        loaded_components = []

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loaded_components.append("Optimizer")

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            loaded_components.append("Scheduler")

        step = checkpoint.get("step", 0)

        if loaded_components:
            logger.info(
                f"Training state restored | Step: {step} | Components: {', '.join(loaded_components)}"
            )
        else:
            logger.info(
                f"No optimizer/scheduler state found. Resuming from Step: {step}"
            )

        return step

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, use_ema_weights: bool = False
    ) -> torch.Tensor:
        x = self.manifold.projx(x)
        xnn = rearrange(
            x, "b (c h w) -> b c h w", c=1, h=self.img_size, w=self.img_size
        )
        t = t.to(dtype=xnn.dtype)  # ODE solver emits float32 t; align with model dtype
        if use_ema_weights:
            v = self.ema(t, xnn)
        else:
            v = self.vecfield(t, xnn)
        v = rearrange(v, "b c h w -> b (c h w)")

        return self.manifold.proju(x, v)

    def loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        center = torch.zeros_like(x0)
        x0 = self.manifold.expmap(center, x0)
        x1 = self.manifold.expmap(center, x1)

        t = torch.rand(len(x1), device=x0.device)
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)

        return torch.pow(
            self(path_sample.x_t, path_sample.t[:, 0]) - path_sample.dx_t, 2
        ).mean()

    @torch.no_grad()
    def denoise(
        self,
        x0: torch.Tensor,
        steps: int = 100,
        method: str = "midpoint",
        use_ema: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor:

        center = torch.zeros_like(x0)
        x_init = self.manifold.expmap(center, x0)

        class SolverWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent = parent_model

            def forward(self, x, t, **kwargs):
                return self.parent(x, t, use_ema_weights=use_ema)

        wrapper = SolverWrapper(self)
        solver = RiemannianODESolver(velocity_model=wrapper, manifold=self.manifold)

        return solver.sample(
            x_init=x_init,
            step_size=1.0 / steps,
            method=method,
            return_intermediates=return_intermediates,
            verbose=False,
        )
