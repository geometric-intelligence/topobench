"""Topological Neural Networks over the Air (AirTNN).

This module implements the AirTNN backbone[1] adapted for the TopoBench
training framework. AirTNN is a cell-complex convolutional architecture whose
topological shift operator embeds a wireless communication model -- channel
fading and additive white noise -- directly into the lower/upper neighborhood
filtering. Setting ``snr_db = 100`` recovers the ideal (noise-free) cell-complex
filter.

Adapted from Fiorellino et al. [1]. The reference implementation operates on
dense ``[batch, num_cells, features]`` tensors with a shared Laplacian; here we
operate on TopoBench's flat ``[num_cells, features]`` rank-1 signal with
block-diagonal *sparse* lower/upper Laplacians, so each shift is a single
``torch.sparse.mm`` and the batch dimension disappears.

[1] Fiorellino, Battiloro, Di Lorenzo. "Topological Neural Networks over the
    Air." https://arxiv.org/abs/2502.10070
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear


class AirTNNLayer(nn.Module):
    r"""A single AirTNN cell-complex convolutional layer.

    Implements the over-the-air topological filter of [1, Eq. (10)-(13)]. The
    output is a sum of (i) lower-shifted, (ii) upper-shifted, and (iii) order-0
    (self) contributions, each passed through a learnable linear map:

    .. math::
        y = H_0 x + \sum_{j=1}^{k+1} \big( U_j\, S_u^{(j)} x + L_j\, S_d^{(j)} x \big),

    where :math:`S_u, S_d` are the upper/lower shift operators. Over the air,
    each application of a shift multiplies the operator entries by sampled
    channel fading and adds white noise (see :meth:`_shift`).

    Notes
    -----
    The shift operators :math:`S_d, S_u` are instantiated as the lower/upper
    Hodge Laplacians :math:`L_d = B_1^\top B_1` and :math:`L_u = B_2 B_2^\top`
    supplied by TopoBench, matching the cell-complex FIR filter of [1, Eq. (2)]
    (the standard convolutional filter of Yang et al. and Barbarossa--
    Sardellitti). Per [1, Eq. (8)-(9)], the over-the-air channel gains are
    placed on exactly the nonzero entries of the shift operator; since the
    Hodge Laplacians carry a populated diagonal (the lower/upper cell degrees),
    fading is applied to those diagonal entries as well, by design -- the paper
    constrains only the *off-support* entries (:math:`[S]_{ij}=0` iff cells are
    not neighbors) and leaves :math:`S` as the Laplacian.

    The order-0 (self) contribution :math:`H_0 x` -- the :math:`w_0`,
    :math:`S^0 = I` term of the filter polynomial -- is realized by the
    separate ``h_lin`` branch acting directly on ``x``. It never passes through
    :meth:`_fade` or the noise injection, so each cell retains exact, un-faded
    knowledge of its own state independently of the diagonal of :math:`S`.

    Parameters
    ----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    k : int, optional
        Filter-order parameter: each neighborhood branch spans shift orders
        ``1..k+1`` (i.e. ``k+1`` learnable maps per branch), in addition to the
        order-0 self term. Default: 1.
    snr_db : float, optional
        Channel signal-to-noise ratio in dB. Use ``100`` for the ideal,
        noise-free filter (default: 100).
    delta : float, optional
        Scale of the (Rayleigh) channel fading (default: 1.0).
    """

    def __init__(self, c_in, c_out, k=1, snr_db=100, delta=1.0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.snr_db = snr_db
        self.snr_lin = 10 ** (snr_db / 10)
        self.delta = float(delta)

        # One linear map per shift order (1..k+1) for each of the two
        # neighborhoods, plus an order-0 self map.
        self.up_lins = nn.ModuleList(
            [Linear(c_in, c_out, bias=False) for _ in range(k + 1)]
        )
        self.low_lins = nn.ModuleList(
            [Linear(c_in, c_out, bias=False) for _ in range(k + 1)]
        )
        self.h_lin = Linear(c_in, c_out, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize all learnable linear maps."""
        for lin in self.up_lins:
            lin.reset_parameters()
        for lin in self.low_lins:
            lin.reset_parameters()
        self.h_lin.reset_parameters()

    @staticmethod
    def _white_noise(x, snr_lin):
        r"""Sample additive white Gaussian noise scaled to the signal power.

        Parameters
        ----------
        x : torch.Tensor
            Signal of shape ``[num_cells, features]``.
        snr_lin : float
            Linear-scale SNR (already converted from dB).

        Returns
        -------
        torch.Tensor
            Noise tensor with the same shape as ``x``. Sampled under
            ``no_grad`` -- noise is a channel effect, not a learnable quantity.
        """
        with torch.no_grad():
            power = (x.detach() ** 2).sum() / x.numel()
            std = torch.sqrt(power / snr_lin)
            return torch.randn_like(x) * std

    @staticmethod
    def _fade(laplacian, delta):
        r"""Apply Rayleigh channel fading to the nonzero entries of a sparse op.

        Each existing connection (nonzero of the sparse Laplacian) is scaled by
        an independent fading magnitude :math:`|h|`, with
        :math:`h \sim \mathcal{CN}(0, \delta^2)`.

        Parameters
        ----------
        laplacian : torch.Tensor
            Coalesced sparse ``[num_cells, num_cells]`` shift operator.
        delta : float
            Fading scale.

        Returns
        -------
        torch.Tensor
            A new coalesced sparse tensor with faded values. The fading is
            sampled under ``no_grad``; gradients still flow through the dense
            signal in the subsequent ``sparse.mm``.
        """
        values = laplacian.values()
        with torch.no_grad():
            fading = (
                torch.randn(
                    values.shape,
                    dtype=torch.complex64,
                    device=laplacian.device,
                )
                * delta
            ).abs()
        return torch.sparse_coo_tensor(
            laplacian.indices(), values * fading, laplacian.shape
        ).coalesce()

    def _shift(self, x, laplacian):
        r"""Apply one over-the-air topological shift ``L @ x``.

        Ideal channel (``snr_db == 100``) is a plain sparse matmul; otherwise
        the operator is faded and white noise is added. Fading is applied to
        every nonzero of the Hodge Laplacian per [1, Eq. (8)-(9)], diagonal
        included; the un-faded self term is handled separately by ``h_lin``.

        Parameters
        ----------
        x : torch.Tensor
            Signal of shape ``[num_cells, features]``.
        laplacian : torch.Tensor
            Coalesced sparse lower or upper Laplacian.

        Returns
        -------
        torch.Tensor
            Shifted signal of shape ``[num_cells, features]``.
        """
        if self.snr_db == 100:
            return torch.sparse.mm(laplacian, x)
        return torch.sparse.mm(
            self._fade(laplacian, self.delta), x
        ) + self._white_noise(x, self.snr_lin)

    def forward(self, x, down_laplacian, up_laplacian):
        r"""Forward pass of the AirTNN layer.

        Parameters
        ----------
        x : torch.Tensor
            Rank-1 cell signal of shape ``[num_cells, c_in]``.
        down_laplacian : torch.Tensor
            Coalesced sparse lower Laplacian ``L_d`` (``[num_cells, num_cells]``).
        up_laplacian : torch.Tensor
            Coalesced sparse upper Laplacian ``L_u`` (``[num_cells, num_cells]``).

        Returns
        -------
        torch.Tensor
            Output signal of shape ``[num_cells, c_out]``.
        """
        x_up = self._shift(x, up_laplacian)
        x_low = self._shift(x, down_laplacian)
        out = self.up_lins[0](x_up) + self.low_lins[0](x_low)

        for up_lin, low_lin in zip(
            self.up_lins[1:], self.low_lins[1:], strict=True
        ):
            x_up = self._shift(x_up, up_laplacian)
            x_low = self._shift(x_low, down_laplacian)
            out = out + up_lin(x_up) + low_lin(x_low)

        return out + self.h_lin(x)


class AirTNN(nn.Module):
    r"""AirTNN backbone: a stack of :class:`AirTNNLayer` over a cell complex.

    Mirrors the rank-1 contract of TopoBench's ``CCCN`` backbone -- it consumes
    the rank-1 signal and the lower/upper Laplacians and returns rank-1
    embeddings -- so it can reuse ``CCCNWrapper`` without modification.

    Parameters
    ----------
    in_channels : int
        Number of input (and hidden) channels. Held constant across layers to
        match the feature-encoder output and the readout hidden dim.
    n_layers : int, optional
        Number of AirTNN layers (default: 2).
    k : int, optional
        Per-layer filter-order parameter; each layer spans neighborhood shift
        orders ``1..k+1`` (default: 1).
    snr_db : float, optional
        Channel SNR in dB; ``100`` is the ideal noise-free filter (default: 100).
    delta : float, optional
        Channel fading scale (default: 1.0).
    dropout : float, optional
        Dropout applied to the input of each layer (default: 0.0).
    last_act : bool, optional
        Whether to apply the activation after the final layer (default: False).
    """

    def __init__(
        self,
        in_channels,
        n_layers=2,
        k=1,
        snr_db=100,
        delta=1.0,
        dropout=0.0,
        last_act=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.last_act = last_act
        self.layers = nn.ModuleList(
            [
                AirTNNLayer(
                    in_channels, in_channels, k=k, snr_db=snr_db, delta=delta
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, down_laplacian, up_laplacian):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Rank-1 cell signal of shape ``[num_cells, in_channels]``.
        down_laplacian : torch.Tensor
            Coalesced sparse lower Laplacian.
        up_laplacian : torch.Tensor
            Coalesced sparse upper Laplacian.

        Returns
        -------
        torch.Tensor
            Rank-1 embeddings of shape ``[num_cells, in_channels]``.
        """
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(
                F.dropout(x, p=self.dropout, training=self.training),
                down_laplacian,
                up_laplacian,
            )
            if not (i == last and not self.last_act):
                x = x.relu()
        return x
