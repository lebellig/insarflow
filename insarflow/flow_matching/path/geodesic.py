# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import Tensor
from torch.func import jvp, vmap

from insarflow.flow_matching.path.path import ProbPath

from insarflow.flow_matching.path.path_sample import PathSample
from insarflow.flow_matching.path.scheduler import ConvexScheduler
from insarflow.flow_matching.utils import expand_tensor_like

from insarflow.flow_matching.utils.manifolds import geodesic, Manifold


class GeodesicProbPath(ProbPath):
    r"""The ``GeodesicProbPath`` class represents a probability path defined through geodesic interpolation on a Riemannian manifold.

    .. math::

        X_t = \psi_t(X_0 | X_1) = \exp_{X_1}(\kappa_t \log_{X_1}(X_0)),

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\kappa_t`.
        manifold (Manifold): The manifold on which the probability path is defined.

    """

    def __init__(self, scheduler: ConvexScheduler, manifold: Manifold):
        self.scheduler = scheduler
        self.manifold = manifold

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from the Riemannian probability path with geodesic interpolation.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: A conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        t = expand_tensor_like(input_tensor=t, expand_to=x_1[..., 0:1]).clone()

        def cond_u(x_0, x_1, t):
            path = geodesic(self.manifold, x_0, x_1)
            x_t, dx_t = jvp(
                lambda t: path(self.scheduler(t).alpha_t),
                (t,),
                (torch.ones_like(t).to(t),),
            )
            return x_t, dx_t

        x_t, dx_t = vmap(cond_u)(x_0, x_1, t)
        x_t = x_t.reshape_as(x_1)
        dx_t = dx_t.reshape_as(x_1)

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
