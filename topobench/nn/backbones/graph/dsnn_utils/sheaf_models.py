r"""Learners that predict directed sheaf restriction maps from features.

The paper learns the restriction maps end to end as a function of the incident
node features (Sec. 3, "Learnable Sheaf Laplacian"):

.. math::

    \mathcal{F}^{0}_{u \lhd e} = \Phi(x_u \Vert x_v),

reshaped into a :math:`d \times d` matrix. That is the entirety of what the
paper specifies about :math:`\Phi`, so we follow the Neural Sheaf Diffusion
reference implementation whose hyperparameter names the paper reuses verbatim
(``sheaf_act``, ``d``, ``add_lp``, ``add_hp``): a single bias-free linear layer
on the concatenated endpoint features followed by an activation.

Adapted from https://github.com/twitter-research/neural-sheaf-diffusion.

Deviation from the TopoBench Neural Sheaf Diffusion port: ``relu`` and
``sigmoid`` are accepted here. The paper searches
``sheaf_act`` over ``{elu, tanh, relu}`` (App. F) but that port only
implements ``{id, tanh, elu}``, so its ``relu`` grid point is unreachable.
"""

# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

SHEAF_ACTIVATIONS = ("id", "tanh", "elu", "relu", "sigmoid")


def get_sheaf_activation(sheaf_act):
    """Resolve a ``sheaf_act`` name to a callable.

    Parameters
    ----------
    sheaf_act : str
        Activation name, one of ``SHEAF_ACTIVATIONS``.

    Returns
    -------
    callable
        The activation, applied elementwise to the predicted parameters.

    Raises
    ------
    ValueError
        If ``sheaf_act`` is not one of ``SHEAF_ACTIVATIONS``.
    """
    if sheaf_act == "id":
        return lambda x: x
    if sheaf_act == "tanh":
        return torch.tanh
    if sheaf_act == "elu":
        return F.elu
    if sheaf_act == "relu":
        return F.relu
    if sheaf_act == "sigmoid":
        return torch.sigmoid
    raise ValueError(
        f"Unsupported act {sheaf_act!r}; expected one of {SHEAF_ACTIVATIONS}"
    )


class SheafLearner(nn.Module):
    """Base class for modules that predict restriction maps from features.

    Subclasses implement :meth:`forward`. :meth:`set_L` exists so the
    diffusion model can stash the realised transport maps for inspection,
    matching the Neural Sheaf Diffusion interface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.L = None

    @abstractmethod
    def forward(self, x, edge_index):
        """Predict restriction-map parameters for every edge.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.

        Returns
        -------
        torch.Tensor
            Predicted parameters, one row per edge.

        Raises
        ------
        NotImplementedError
            Always, in this abstract base class.
        """
        raise NotImplementedError

    def set_L(self, weights) -> None:
        """Store a detached copy of the realised transport maps.

        Parameters
        ----------
        weights : torch.Tensor
            Transport maps to record.
        """
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    r"""Predict restriction maps from concatenated endpoint features.

    Computes the :math:`\mathcal{F}^{0}_{u \lhd e} = \Phi(x_u \Vert x_v)` of
    Sec. 3 ("Learnable Sheaf Laplacian") with a single bias-free linear layer
    followed by ``sheaf_act``. The concatenation is ordered, so
    :math:`\Phi(x_u \Vert x_v) \neq \Phi(x_v \Vert x_u)` and the two
    restriction maps of an edge differ, which is what makes the sheaf
    non-trivial.

    Parameters
    ----------
    in_channels : int
        Number of input channels per node. The linear layer therefore has
        ``2 * in_channels`` inputs.
    out_shape : tuple of int
        Shape of the predicted parameters per edge: ``(d,)`` for diagonal
        maps, ``(k,)`` for an orthogonal parameterization consuming ``k``
        values, or ``(d, d)`` for general maps.
    sheaf_act : str, optional
        Activation name, one of ``SHEAF_ACTIVATIONS``. Default is ``"tanh"``.

    Raises
    ------
    ValueError
        If ``out_shape`` is not of length 1 or 2, or if ``sheaf_act`` is not
        one of ``SHEAF_ACTIVATIONS``.
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: tuple[int, ...],
        sheaf_act: str = "tanh",
    ) -> None:
        super().__init__()
        if len(out_shape) not in (1, 2):
            raise ValueError(
                f"out_shape must have length 1 or 2, got {out_shape}"
            )
        self.in_channels = in_channels
        self.out_shape = out_shape
        num_out = 1
        for dim in out_shape:
            num_out *= dim
        self.linear1 = nn.Linear(in_channels * 2, num_out, bias=False)
        self.act = get_sheaf_activation(sheaf_act)
        self.sheaf_act = sheaf_act

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_shape={self.out_shape}, sheaf_act={self.sheaf_act!r})"
        )

    def forward(self, x, edge_index):
        """Predict restriction-map parameters for every edge.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge indices of shape ``[2, num_edges]``.

        Returns
        -------
        torch.Tensor
            Parameters of shape ``[num_edges, *out_shape]``.
        """
        row, col = edge_index
        maps = self.linear1(torch.cat([x[row], x[col]], dim=1))
        maps = self.act(maps)
        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        return maps.view(-1, self.out_shape[0])
