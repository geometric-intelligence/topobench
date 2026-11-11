"""Topological Neural Networks over the Air (AirTNN).

Faithful implementation of Fiorellino, Battiloro, Di Lorenzo, *Topological
Neural Networks over the Air*, 2025 — `arXiv:2502.10070
<https://arxiv.org/abs/2502.10070>`_.

An AirTNN layer (Eq. (14)) is a bank of *topological filters over-the-air*
(AirTF, Eq. (12)) followed by a pointwise nonlinearity: shift-and-sum filtering
of a rank-``k`` signal over the **lower** and **upper** neighbourhoods of a
regular cell complex (Sec. 2), where every shift is transmitted through a
wireless channel — per-link Rayleigh fading gains form the *air* shift
operators (Eqs. (8)-(9)) and AWGN is added at every hop (Eqs. (4)-(5)),
resampled i.i.d. across successive transmissions (Eqs. (10)-(11)).

With ideal channels (``air_enabled=False``, i.e. :math:`h_{ij}=1`,
:math:`n_i=0`) the layer reduces **exactly** to the standard cell-complex FIR
filter bank of Eqs. (2)-(3) — this reduction is asserted by the unit tests.
Channel randomness is active in *both* training and inference, per the
paper's training scheme (Sec. 3, "Training of AirTNNs").
"""

import torch
import torch.nn as nn


def _pattern(op):
    """Extract the directed off-diagonal connectivity pattern of an operator.

    Parameters
    ----------
    op : torch.Tensor
        Either a sparse matrix (e.g. a down/up Laplacian, whose off-diagonal
        sparsity encodes the lower/upper neighbourhoods of Sec. 2) or a dense
        ``[2, E]`` integer edge-index.

    Returns
    -------
    tuple of torch.Tensor
        ``(rows, cols)`` index tensors of the neighbourhood pattern, i.e. the
        support of the shift operators :math:`S^{(d)}, S^{(u)}` of Eq. (2)
        and of their over-the-air counterparts of Eqs. (8)-(9).
    """
    if op.is_sparse:
        idx = op.coalesce().indices()
        mask = idx[0] != idx[1]
        return idx[0][mask], idx[1][mask]
    return op[0], op[1]


class AirTF(nn.Module):
    r"""Topological filter over-the-air — Eq. (12).

    A linear combination of multi-shifted signals over the lower and upper
    neighbourhoods, :math:`y=\sum_{p=0}^{P} w^{(d)}_p x^{(d,p)}
    + \sum_{p=0}^{P} w^{(u)}_p x^{(u,p)}`, where the ``p``-shifted signals
    follow Eqs. (10)-(11): each hop applies an air shift operator whose
    nonzeros are fresh i.i.d. Rayleigh fading gains (Eqs. (8)-(9)) and adds
    AWGN (Eqs. (4)-(5)). Multi-feature filter banks use weight matrices
    :math:`W^{(d)}_p, W^{(u)}_p` as in Eqs. (13)-(14).

    Parameters
    ----------
    in_channels : int
        Number of input features :math:`F_{in}`.
    out_channels : int
        Number of output features :math:`F_{out}`.
    filter_order : int, optional
        Filter length :math:`P` of Eq. (12) (default: 2).
    delta : float, optional
        Rayleigh fading scale :math:`\delta` (default: 1.0).
    snr_db : float, optional
        Signal-to-noise ratio in dB fixing the AWGN variance per hop,
        relative to the current signal power (default: 20.0).
    air_enabled : bool, optional
        If False, ideal channels (:math:`h_{ij}=1`, no noise): the filter
        reduces exactly to the cell-complex FIR filter of Eq. (2)
        (default: True).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        filter_order=2,
        delta=1.0,
        snr_db=20.0,
        air_enabled=True,
    ):
        super().__init__()
        self.P = filter_order
        self.delta = delta
        self.snr_db = snr_db
        self.air_enabled = air_enabled
        scale = (in_channels * (filter_order + 1)) ** -0.5
        # W^{(d)}_p and W^{(u)}_p of Eqs. (13)-(14), p = 0..P.
        self.w_down = nn.Parameter(
            scale * torch.randn(filter_order + 1, in_channels, out_channels)
        )
        self.w_up = nn.Parameter(
            scale * torch.randn(filter_order + 1, in_channels, out_channels)
        )

    def _air_shift(self, x, rows, cols, n):
        """One over-the-air shift — Eqs. (4)-(9).

        Parameters
        ----------
        x : torch.Tensor
            Rank-``k`` signal ``[N, F]``.
        rows : torch.Tensor
            Row indices of the neighbourhood pattern.
        cols : torch.Tensor
            Column indices of the neighbourhood pattern.
        n : int
            Number of ``k``-cells.

        Returns
        -------
        torch.Tensor
            The shifted signal :math:`S_{air} x + n` (or the ideal
            :math:`S x` when ``air_enabled`` is False).
        """
        if self.air_enabled:
            u = torch.rand(rows.numel(), device=x.device)
            vals = self.delta * torch.sqrt(-2.0 * torch.log(u + 1e-12))
        else:
            vals = torch.ones(rows.numel(), device=x.device)
        s = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n))
        out = torch.sparse.mm(s, x)
        if self.air_enabled:
            sigma2 = x.pow(2).mean() * 10.0 ** (-self.snr_db / 10.0)
            out = out + torch.randn_like(out) * torch.sqrt(sigma2 + 1e-12)
        return out

    def forward(self, x, pat_down, pat_up):
        """Apply the AirTF of Eq. (12) with the filter bank of Eq. (14).

        Parameters
        ----------
        x : torch.Tensor
            Input rank-``k`` signal ``[N, F_in]``.
        pat_down : tuple of torch.Tensor
            Lower-neighbourhood pattern ``(rows, cols)``.
        pat_up : tuple of torch.Tensor
            Upper-neighbourhood pattern ``(rows, cols)``.

        Returns
        -------
        torch.Tensor
            Output signal ``[N, F_out]``.
        """
        n = x.shape[0]
        y = x @ self.w_down[0] + x @ self.w_up[0]  # p = 0 terms
        xd = xu = x
        for p in range(1, self.P + 1):  # Eqs. (10)-(11)
            xd = self._air_shift(xd, pat_down[0], pat_down[1], n)
            xu = self._air_shift(xu, pat_up[0], pat_up[1], n)
            y = y + xd @ self.w_down[p] + xu @ self.w_up[p]
        return y


class AirTNN(nn.Module):
    r"""AirTNN — stack of AirTF banks and pointwise nonlinearities, Eq. (14).

    Operates on rank-1 (edge) signals of a regular cell complex, "without
    loss of generality" per the paper's Sec. 2: lower neighbours share a
    vertex, upper neighbours share a 2-cell (polygon / triangle).

    Parameters
    ----------
    in_channels : int
        Number of channels (kept constant across layers, mirroring the
        sibling cell backbones).
    n_layers : int, optional
        Number of AirTNN layers :math:`L` (default: 2).
    filter_order : int, optional
        Filter length :math:`P` of Eq. (12) (default: 2).
    delta : float, optional
        Rayleigh fading scale :math:`\delta` (default: 1.0).
    snr_db : float, optional
        Per-hop AWGN SNR in dB (default: 20.0).
    air_enabled : bool, optional
        If False, ideal channels: the network reduces exactly to the
        cell-complex convolutional network of Eq. (3) (default: True).
    dropout : float, optional
        Dropout rate between layers (default: 0.0).
    last_act : bool, optional
        If True, apply the nonlinearity after the last layer
        (default: False).
    """

    def __init__(
        self,
        in_channels,
        n_layers=2,
        filter_order=2,
        delta=1.0,
        snr_db=20.0,
        air_enabled=True,
        dropout=0.0,
        last_act=False,
    ):
        super().__init__()
        self.d = dropout
        self.last_act = last_act
        self.layers = nn.ModuleList(
            AirTF(
                in_channels,
                in_channels,
                filter_order,
                delta,
                snr_db,
                air_enabled,
            )
            for _ in range(n_layers)
        )

    def forward(self, x, Ld, Lu):
        """Forward pass over the lower/upper neighbourhood structure.

        Parameters
        ----------
        x : torch.Tensor
            Rank-1 (edge) signal ``[N_1, F]``.
        Ld : torch.Tensor
            Operator encoding the lower neighbourhood (e.g. the rank-1 down
            Laplacian); only its off-diagonal sparsity pattern is used, per
            the definition of :math:`S^{(d)}` in Eq. (2).
        Lu : torch.Tensor
            Operator encoding the upper neighbourhood (e.g. the rank-1 up
            Laplacian); pattern of :math:`S^{(u)}` in Eq. (2).

        Returns
        -------
        torch.Tensor
            Output edge signal ``[N_1, F]``.
        """
        pat_down = _pattern(Ld)
        pat_up = _pattern(Lu)
        for i, layer in enumerate(self.layers):
            x = layer(
                nn.functional.dropout(x, p=self.d, training=self.training),
                pat_down,
                pat_up,
            )
            if i == len(self.layers) - 1 and self.last_act is False:
                break
            x = x.relu()  # gamma of Eq. (14)
        return x
