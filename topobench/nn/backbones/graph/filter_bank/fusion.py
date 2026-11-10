"""Fusion protocol for :class:`FilterBankGNN`.

A :class:`Fusion` combines the Q per-channel outputs of a filter bank
into a single signal. It is the filter-bank analog of the ``Basis``
protocol:
the backbone holds the channel weights ``γ`` and stays agnostic to how
the channels are combined, so summation, concatenation, or attention
fusions can be swapped via a Hydra ``_target_`` with no backbone change.
"""

from __future__ import annotations

from torch import Tensor, nn


class Fusion(nn.Module):
    """Abstract fusion: combine Q channel outputs into one signal.

    Subclasses implement :meth:`forward`, mapping the list of per-channel
    filtered signals and the channel weights ``γ`` to a single tensor.
    """

    def forward(self, channel_outs: list[Tensor], gamma: Tensor) -> Tensor:
        """Combine the per-channel outputs.

        Parameters
        ----------
        channel_outs : list of Tensor, each shape ``[N, F]``
            The Q outputs ``g_q(L̃) h``, one per channel.
        gamma : Tensor, shape ``[Q]``
            The channel weights ``γ_q`` (owned by the backbone).

        Returns
        -------
        Tensor, shape ``[N, F]``
            The fused signal.
        """
        raise NotImplementedError


class SumFusion(Fusion):
    """Weighted-sum fusion ``y = Σ_q γ_q · channel_outs[q]``.

    The default fusion, used by ACM-GNN, FAGCN, GNN-LF/HF, and FiGURe
    (Liao et al. 2024, Eq. (3) with summation fusion).
    """

    def forward(self, channel_outs: list[Tensor], gamma: Tensor) -> Tensor:
        """Return the γ-weighted sum of the channel outputs.

        Parameters
        ----------
        channel_outs : list of Tensor, each shape ``[N, F]``
            The Q outputs ``g_q(L̃) h``.
        gamma : Tensor, shape ``[Q]``
            The channel weights ``γ_q``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``Σ_q γ_q · channel_outs[q]``.
        """
        y = gamma[0] * channel_outs[0]
        for q in range(1, len(channel_outs)):
            y = y + gamma[q] * channel_outs[q]
        return y
