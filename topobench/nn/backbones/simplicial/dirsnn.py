"""Directed Simplicial Neural Network (Dir-SNN) backbone.

Implements the model introduced in *Higher-Order Topological Directionality
and Directed Simplicial Neural Networks* by M. Lecha, A. Cavallo,
F. Dominici, E. Isufi, and C. Battiloro (ICASSP 2025, arXiv:2409.08389).

This module is a TopoBench port of Andrea Cavallo's reference
implementation at https://github.com/ManuelLecha/DirSNN. As in the paper
experiments, the network operates on edge (1-simplex) signals and
propagates them through a collection of directed lower and upper edge
adjacencies that encode higher-order topological directionality
(Sec. III, Eqs. (3)-(4)).

References
----------
.. [1] M. Lecha, A. Cavallo, F. Dominici, E. Isufi, C. Battiloro.
       *Higher-Order Topological Directionality and Directed Simplicial
       Neural Networks.* ICASSP 2025. arXiv:2409.08389.
"""

import torch
from torch.nn.parameter import Parameter


class DirSNN(torch.nn.Module):
    r"""Directed Simplicial Neural Network backbone.

    Implements the Dir-SNN architecture of Lecha et al. 2024
    (arXiv:2409.08389). Edge features are first projected to a common
    hidden dimension by an MLP and then refined by a stack of
    :class:`DirSNNLayer` modules. Each layer aggregates over the
    directed lower :math:`\mathcal{A}^{i,j}_{\downarrow,1}` (Eq. 3) and
    upper :math:`\mathcal{A}^{i,j}_{\uparrow,1}` (Eq. 4) edge
    adjacencies that are supplied as a list of sparse/dense tensors.

    The number and identity of the adjacencies are not fixed by the
    model: the user is expected to pass between one and ten matrices,
    typically the four lower adjacencies
    (:math:`\mathcal{A}^{0,0}_{\downarrow,1},
    \mathcal{A}^{0,1}_{\downarrow,1},
    \mathcal{A}^{1,0}_{\downarrow,1},
    \mathcal{A}^{1,1}_{\downarrow,1}`) and the six upper adjacencies
    (:math:`\mathcal{A}^{0,1}_{\uparrow,1},
    \mathcal{A}^{0,2}_{\uparrow,1},
    \mathcal{A}^{1,2}_{\uparrow,1},
    \mathcal{A}^{1,0}_{\uparrow,1},
    \mathcal{A}^{2,0}_{\uparrow,1},
    \mathcal{A}^{2,1}_{\uparrow,1}`) used in Sec. IV-A of the paper.

    Parameters
    ----------
    edge_channels : int
        Dimension of the input edge features.
    n_layers : int, optional
        Number of stacked Dir-SNN layers, by default 2.
    n_hid : int, optional
        Hidden dimension on edges (input/output of every layer),
        by default 32.
    conv_order : int, optional
        Order :math:`K` of the polynomial filter applied to every
        adjacency (Chebyshev-like expansion of Eq. (6)/(7)),
        by default 1.
    n_adjs : int, optional
        Number of directed adjacency matrices that will be supplied at
        forward time. Must match the length of ``adjs`` in
        :meth:`forward`, by default 1.
    aggr_norm : bool, optional
        Whether to row-normalise each aggregation step by the
        neighbourhood size, by default False.
    update_func : str or None, optional
        Activation applied after the per-layer linear update
        (``"relu"``, ``"leaky_relu"``, ``"sigmoid"`` or ``None``),
        by default None.
    """

    def __init__(
        self,
        edge_channels,
        n_layers=2,
        n_hid=32,
        conv_order=1,
        n_adjs=1,
        aggr_norm=False,
        update_func=None,
    ):
        super().__init__()
        self.edge_channels = edge_channels
        self.n_hid = n_hid
        self.out_channels = n_hid
        self.n_layers = n_layers
        self.conv_order = conv_order
        self.n_adjs = n_adjs
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        # Initial linear projection of edge features (Eq. (2) of the
        # paper maps topological signals to a learnable embedding before
        # message passing).
        self.in_linear_1 = torch.nn.Linear(edge_channels, n_hid)

        self.layers = torch.nn.ModuleList(
            DirSNNLayer(
                in_channels_1=n_hid,
                out_channels_1=n_hid,
                conv_order=conv_order,
                n_adjs=n_adjs,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_1, adjs):
        r"""Run the Dir-SNN forward pass.

        Iterates the update of Eq. (10) of Lecha et al. 2024
        (arXiv:2409.08389) for ``n_layers`` rounds.

        Parameters
        ----------
        x_1 : torch.Tensor
            Edge feature tensor of shape ``(n_edges, edge_channels)``.
        adjs : sequence of torch.Tensor
            Tuple or list of directed edge adjacency matrices, each of
            shape ``(n_edges, n_edges)``. Length must equal ``n_adjs``.

        Returns
        -------
        torch.Tensor
            Updated edge features of shape ``(n_edges, n_hid)``.
        """
        x_1 = self.in_linear_1(x_1)
        for layer in self.layers:
            x_1 = layer(x_1, adjs)
        return x_1


class DirSNNLayer(torch.nn.Module):
    r"""Single Dir-SNN message-passing layer.

    Implements Eqs. (6)-(10) of Lecha et al. 2024 (arXiv:2409.08389)
    specialised to edge signals. Given ``n_adjs`` directed adjacencies
    :math:`\{A^{(s)}\}_{s=1}^{n\_adjs}`, the layer computes

    .. math::
        y = \sigma\!\left(
            x W_0
            + \sum_{s=1}^{n\_adjs} \sum_{k=1}^{K} (A^{(s)})^{k} x \, W_{s,k}
        \right),

    where :math:`K` is ``conv_order``, every term shares the same
    learnable tensor ``weight_1`` via a single ``einsum`` contraction,
    and :math:`\sigma` is the optional ``update_func``. The identity
    term :math:`x W_0` corresponds to keeping the simplex own feature
    in Eq. (10) (the :math:`x_\sigma^l` argument of :math:`\phi`).

    Parameters
    ----------
    in_channels_1 : int
        Input edge feature dimension.
    out_channels_1 : int
        Output edge feature dimension.
    conv_order : int
        Polynomial order :math:`K` of the filter applied to every
        adjacency. Must be strictly positive.
    n_adjs : int, optional
        Number of directed adjacency matrices used as message passing
        operators, by default 1.
    aggr_norm : bool, optional
        Whether to row-normalise every aggregation step, by default
        False.
    update_func : str or None, optional
        Pointwise activation applied to the layer output, by default
        None. Supported values are ``"relu"``, ``"leaky_relu"``,
        ``"sigmoid"``.
    initialization : str, optional
        Weight initialisation: ``"xavier_uniform"`` or
        ``"xavier_normal"``, by default ``"xavier_normal"``.
    """

    def __init__(
        self,
        in_channels_1,
        out_channels_1,
        conv_order,
        n_adjs=1,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        self.in_channels_1 = in_channels_1
        self.out_channels_1 = out_channels_1
        self.conv_order = conv_order
        self.n_adjs = n_adjs
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        # weight_1 stores all filter taps in a single tensor that is
        # contracted in one einsum, mirroring the shared aggregator
        # convention of Eq. (6)/(7).
        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                conv_order * n_adjs + 1,
            )
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Initialise the learnable weights.

        Applies Xavier initialisation as in the reference Dir-SNN
        implementation; this is purely an implementation detail and is
        not specified by the paper text.

        Parameters
        ----------
        gain : float, optional
            Gain factor passed to ``torch.nn.init.xavier_*_`` (default
            1.414, matching ``sqrt(2)`` for ReLU-style activations).
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Row-normalise an aggregation by neighbourhood size.

        Optional helper invoked when ``aggr_norm=True``. The behaviour
        is unchanged from the reference Dir-SNN code and is not a
        specific paper equation.

        Mathematical note (reference-faithful, deliberate
        double-normalisation): for ``conv_order > 1`` this routine is
        invoked after *every* power of the adjacency
        (``conv_operator``, ``conv_operator^2``, ...), but each call
        normalises by the row sums of the *original* ``conv_operator``
        rather than by the row sums of the higher-power propagation
        operator that was actually used. The effective propagator at
        step ``k`` is therefore ``D^{-1} A^k`` rather than ``(D^{-1}
        A)^k`` (i.e. a single random walk normalisation followed by raw
        powers, not powers of the random-walk-normalised operator).
        This matches the upstream reference implementation
        (``compute_adj.py`` and the DirSNN model code therein)
        verbatim and is preserved here so our results stay numerically
        comparable to the paper's artefacts; it is *not* the standard
        GCN-style row normalisation.

        Parameters
        ----------
        conv_operator : torch.Tensor
            Adjacency matrix used to define the neighbourhoods. Dense
            or sparse, shape ``(n_edges, n_edges)``.
        x : torch.Tensor
            Aggregated features, shape ``(n_edges, num_channels)``.

        Returns
        -------
        torch.Tensor
            Normalised features with the same shape as ``x``.
        """
        if conv_operator.is_sparse:
            neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        else:
            neighborhood_size = torch.sum(conv_operator, dim=1)
        neighborhood_size_inv = 1.0 / neighborhood_size
        neighborhood_size_inv[~torch.isfinite(neighborhood_size_inv)] = 0.0
        x = torch.einsum("i,ij->ij", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0.0
        return x

    def update(self, x):
        r"""Apply the pointwise activation of Eq. (10).

        The update function :math:`\phi` in Eq. (10) of Lecha et al.
        2024 is implemented as a fixed nonlinearity selected by
        ``update_func``.

        Parameters
        ----------
        x : torch.Tensor
            Pre-activation output of shape
            ``(n_edges, out_channels_1)``.

        Returns
        -------
        torch.Tensor
            Activated tensor.

        Raises
        ------
        ValueError
            If ``update_func`` is not one of ``"relu"``,
            ``"leaky_relu"``, ``"sigmoid"``. The reference Dir-SNN
            implementation silently returned ``None`` here, which
            propagated to ``forward`` and caused opaque downstream
            crashes; we raise instead so the misconfiguration is
            visible at the first call site.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "leaky_relu":
            return torch.nn.functional.leaky_relu(x)
        raise ValueError(
            f"Unknown update_func: {self.update_func!r}; expected one of "
            "{'relu', 'leaky_relu', 'sigmoid'} or None."
        )

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Apply repeated propagation by a directed adjacency.

        Computes ``conv_operator^k @ x`` for ``k = 1, ..., conv_order``
        and stacks the results along the last axis. This realises the
        polynomial filter used inside the intra-neighbourhood
        aggregator :math:`\psi_{\mathcal{A}}` of Eqs. (6)-(7) of the
        paper.

        Parameters
        ----------
        conv_operator : torch.Tensor
            Directed adjacency matrix of shape ``(n_edges, n_edges)``.
            Dense or sparse; for sparse tensors the propagation uses
            ``torch.sparse.mm``.
        conv_order : int
            Polynomial order :math:`K`.
        x : torch.Tensor
            Edge feature tensor of shape
            ``(n_edges, num_channels)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_edges, num_channels, conv_order)``
            with ``X[..., k] = conv_operator^{k+1} @ x``.
        """
        num_simplices, num_channels = x.shape
        out = torch.empty(
            size=(num_simplices, num_channels, conv_order),
            device=x.device,
            dtype=x.dtype,
        )

        def _propagate(op, y):
            """Apply ``op @ y`` selecting sparse or dense matmul.

            Parameters
            ----------
            op : torch.Tensor
                Square propagation operator. May be sparse or dense.
            y : torch.Tensor
                Right-hand-side tensor to be propagated.

            Returns
            -------
            torch.Tensor
                The matrix product ``op @ y``.
            """
            if op.is_sparse:
                return torch.sparse.mm(op, y)
            return torch.mm(op, y)

        out[:, :, 0] = _propagate(conv_operator, x)
        if self.aggr_norm:
            out[:, :, 0] = self.aggr_norm_func(conv_operator, out[:, :, 0])
        for k in range(1, conv_order):
            out[:, :, k] = _propagate(conv_operator, out[:, :, k - 1])
            if self.aggr_norm:
                out[:, :, k] = self.aggr_norm_func(conv_operator, out[:, :, k])
        return out

    def forward(self, x_1, adjs):
        r"""Run one Dir-SNN message-passing step.

        Implements Eq. (10) of Lecha et al. 2024 (arXiv:2409.08389):
        the edge feature is updated by concatenating its own value
        with the polynomial expansion of every supplied directed
        adjacency and contracting the result against ``weight_1``.

        Parameters
        ----------
        x_1 : torch.Tensor
            Edge features of shape ``(n_edges, in_channels_1)``.
        adjs : sequence of torch.Tensor
            Directed adjacency matrices, each of shape
            ``(n_edges, n_edges)``. Length must equal ``n_adjs``.

        Returns
        -------
        torch.Tensor
            Updated edge features of shape
            ``(n_edges, out_channels_1)``.

        Raises
        ------
        AssertionError
            If the number of adjacencies in ``adjs`` does not match
            ``n_adjs`` set at construction time.
        """
        assert len(adjs) == self.n_adjs, (
            f"Expected {self.n_adjs} adjacencies, got {len(adjs)}."
        )

        # Identity term: x_sigma^l in Eq. (10).
        x_1_all = [
            x_1.unsqueeze(2),
            *(self.chebyshev_conv(adj, self.conv_order, x_1) for adj in adjs),
        ]

        x_1_all = torch.cat(x_1_all, dim=2)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)

        if self.update_func is None:
            return y_1
        return self.update(y_1)
