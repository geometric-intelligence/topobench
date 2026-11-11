"""GREAD: Graph Neural Reaction-Diffusion Networks.

Implementation of the paper "GREAD: Graph Neural Reaction-Diffusion
Networks" (Choi et al., ICML 2023, https://arxiv.org/abs/2211.14208),
adapted from the official reference implementation
https://github.com/jeongwhanchoi/GREAD.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import (
    add_remaining_self_loops,
    scatter,
)


class GREAD(nn.Module):
    r"""GREAD: Graph Neural Reaction-Diffusion Network backbone.

    GREAD (Choi et al., ICML 2023, https://arxiv.org/abs/2211.14208)
    evolves node representations :math:`\mathbf{H}(t)` with the
    reaction-diffusion equation (Section 3.1, Eqs. (6)-(8) of the paper):

    .. math::
        \mathbf{f}(\mathbf{H}(t)) := \frac{d\mathbf{H}(t)}{dt} =
            -\alpha \, \mathbf{L}\mathbf{H}(t)
            + \beta \, \mathbf{r}(\mathbf{H}(t)),

    with initial value :math:`\mathbf{H}(0) = \mathbf{e}(\mathbf{X})`
    (Eq. (6)), where :math:`\mathbf{e}` is an encoding layer,
    :math:`\mathbf{L} = \mathbf{I} - \mathbf{A}` the Laplacian of a
    normalized adjacency matrix :math:`\mathbf{A}`, and
    :math:`\mathbf{r}` one of the reaction terms of Eq. (10) of the
    paper:

    - ``"bspm"``: blurring-sharpening
      :math:`(\mathbf{A} - \mathbf{A}^2)\mathbf{H}(t)`
      (GREAD-BS, Eq. (14)).
    - ``"fisher"``: Fisher
      :math:`\mathbf{H}(t)\odot(1 - \mathbf{H}(t))` (GREAD-F).
    - ``"allen-cahn"``: Allen-Cahn
      :math:`\mathbf{H}(t)\odot(1 - \mathbf{H}(t)^{\circ 2})`
      (GREAD-AC).
    - ``"zeldovich"``: Zeldovich
      :math:`\mathbf{H}(t)\odot(\mathbf{H}(t) -
      \mathbf{H}(t)^{\circ 2})` (GREAD-Z).
    - ``"st"``: source term :math:`\mathbf{H}(0)` (GREAD-ST).
    - ``"fb"``: filter bank (high-pass) reaction (GREAD-FB). Eq. (10)
      of the paper writes it as :math:`\mathbf{L}\mathbf{H}(t)`, while
      the reference implementation computes
      :math:`(\mathbf{I} + \mathbf{A})\mathbf{H}(t)`; this
      implementation follows the reference code.
    - ``"fb3"``: filter bank with identity channel (GREAD-FB*), the
      ``"fb"`` reaction plus :math:`\mathbf{H}(t)`, again following the
      reference code.
    - ``"none"``: no reaction, i.e. pure diffusion.

    The ODE is integrated with the explicit Euler method, the base
    solver in which Eq. (5) of the paper is written and one of the two
    fixed-grid solvers (``euler`` / ``rk4``) selected per dataset by
    the reference implementation
    (https://github.com/jeongwhanchoi/GREAD,
    ``src/gread_params.py``). Integration reaches the terminal time
    ``time`` up to floating-point rounding: full steps of size
    ``step_size`` are taken and, when ``time`` is not an integer
    multiple of ``step_size``, a final shorter step covers the remaining
    duration.

    The scalar gates are parameterized as
    :math:`\alpha = \sigma(\alpha_{\text{train}})` and
    :math:`\beta = \sigma(\beta_{\text{train}})` following
    ``ODEFuncGread.forward`` of the reference code; with
    ``beta_diag=True`` the reaction is instead weighted by a trainable
    diagonal matrix :math:`\beta_W` (the "(VC)" variants in the paper).

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden units.
    reaction_term : str, optional
        Reaction term, one of ``"bspm"``, ``"fisher"``, ``"allen-cahn"``,
        ``"zeldovich"``, ``"st"``, ``"fb"``, ``"fb3"``, ``"none"``
        (default: ``"bspm"``).
    time : float, optional
        Terminal integration time :math:`T`; must be positive
        (default: 3.0).
    step_size : float, optional
        Step size :math:`\tau` of the explicit Euler solver; must be
        positive (default: 1.0).
    beta_diag : bool, optional
        If True, use a trainable diagonal matrix ("vector channel")
        instead of the scalar :math:`\beta` gate (default: False).
    add_source : bool, optional
        If True, add a learnable source term
        :math:`s\,\mathbf{H}(0)` to the dynamics, with :math:`s`
        initialized at zero so the source contributes nothing until
        trained (default: False).
    data_norm : str, optional
        Adjacency normalization: ``"rw"`` for the column-stochastic
        random-walk normalization :math:`\mathbf{A}\mathbf{D}^{-1}`
        (degrees taken over the target index, matching
        ``get_rw_adj(..., norm_dim=1)`` of the reference
        implementation) or ``"gcn"`` for the symmetric normalization
        :math:`\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}`
        (default: ``"rw"``).
    self_loop_weight : float, optional
        Weight of the self loops added to the adjacency matrix before
        normalization (default: 1.0).
    xn_activation : bool, optional
        If True, apply a terminal ReLU to the integrated
        representations :math:`\mathbf{H}(T)` before the output
        dropout (``XN_activation`` in the reference implementation,
        whose CLI default is ``False``; it is enabled only for selected
        dataset configurations). Default: ``False``.
    input_dropout : float, optional
        Dropout rate applied to the input features (default: 0.0).
    dropout : float, optional
        Dropout rate applied to the final representation
        (default: 0.0).
    **kwargs
        Additional arguments (ignored).
    """

    _REACTION_TERMS = (
        "bspm",
        "fisher",
        "allen-cahn",
        "zeldovich",
        "st",
        "fb",
        "fb3",
        "none",
    )

    def __init__(
        self,
        in_channels,
        hidden_channels,
        reaction_term="bspm",
        time=3.0,
        step_size=1.0,
        beta_diag=False,
        add_source=False,
        data_norm="rw",
        self_loop_weight=1.0,
        xn_activation=False,
        input_dropout=0.0,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        if reaction_term not in self._REACTION_TERMS:
            raise ValueError(
                f"Unknown reaction term '{reaction_term}'. "
                f"Expected one of {self._REACTION_TERMS}."
            )
        if data_norm not in ("rw", "gcn"):
            raise ValueError(
                f"Unknown data normalization '{data_norm}'. "
                "Expected 'rw' or 'gcn'."
            )
        if time <= 0:
            raise ValueError(
                f"Terminal time must be positive, got time={time}."
            )
        if step_size <= 0:
            raise ValueError(
                f"Step size must be positive, got step_size={step_size}."
            )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.reaction_term = reaction_term
        self.time = time
        self.step_size = step_size
        self.step_sizes = self._euler_step_sizes(time, step_size)
        self.beta_diag = beta_diag
        self.add_source = add_source
        self.data_norm = data_norm
        self.self_loop_weight = self_loop_weight
        self.xn_activation = xn_activation
        self.input_dropout = input_dropout
        self.dropout = dropout

        # Encoding layer E (m1 in the reference implementation).
        self.encoder = nn.Linear(in_channels, hidden_channels)

        # Scalar gates alpha and beta (sigmoid-activated, initialized at
        # zero as in ODEFunc.__init__ of the reference implementation).
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        if beta_diag:
            self.b_w = nn.Parameter(torch.empty(hidden_channels))
        if add_source:
            self.source_train = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    @staticmethod
    def _euler_step_sizes(time, step_size):
        """Compute the explicit Euler step schedule.

        Full steps of ``step_size`` are taken while they fit within the
        horizon; a final shorter step covers any remaining duration so
        that the step sizes sum to ``time`` up to floating-point
        rounding. At least one step is always taken for a positive
        horizon.

        Parameters
        ----------
        time : float
            Terminal integration time.
        step_size : float
            Nominal step size.

        Returns
        -------
        list of float
            The sizes of the successive Euler steps; they sum to
            ``time`` up to floating-point rounding.
        """
        n_full = int(time / step_size)
        remainder = time - n_full * step_size
        sizes = [step_size] * n_full
        if remainder > 0:
            sizes.append(remainder)
        return sizes

    def reset_parameters(self):
        """Reset the learnable parameters."""
        self.encoder.reset_parameters()
        with torch.no_grad():
            self.alpha_train.zero_()
            self.beta_train.zero_()
        if self.beta_diag:
            nn.init.uniform_(self.b_w, a=-1.0, b=1.0)
        if self.add_source:
            with torch.no_grad():
                self.source_train.zero_()

    def normalize_adjacency(self, edge_index, edge_weight, num_nodes):
        r"""Normalize the adjacency matrix.

        Adds weighted self loops and normalizes the adjacency matrix,
        following ``get_rw_adj`` (``data_norm="rw"``) and
        ``gcn_norm_fill_val`` (``data_norm="gcn"``) of the reference
        implementation (``src/utils.py``).

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index of shape ``[2, num_edges]``.
        edge_weight : torch.Tensor or None
            Optional edge weights of shape ``[num_edges]``.
        num_nodes : int
            Number of nodes.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Normalized edge index and edge weights.
        """
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=torch.float,
                device=edge_index.device,
            )
        if self.self_loop_weight > 0:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index,
                edge_weight,
                fill_value=self.self_loop_weight,
                num_nodes=num_nodes,
            )
        row, col = edge_index[0], edge_index[1]
        if self.data_norm == "rw":
            # A D^{-1}: degrees over the target (column) index, i.e. a
            # column-stochastic operator (norm_dim=1 in the reference
            # implementation).
            deg = scatter(
                edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum"
            )
            deg_inv = deg.pow(-1)
            deg_inv.masked_fill_(deg_inv == float("inf"), 0)
            edge_weight = deg_inv[col] * edge_weight
        else:
            deg = scatter(
                edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum"
            )
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    @staticmethod
    def spmm(edge_index, edge_weight, num_nodes, x):
        r"""Multiply the sparse adjacency matrix with a dense matrix.

        Computes :math:`\mathbf{A}\mathbf{X}`, the equivalent of
        ``ODEFuncGread.sparse_multiply`` of the reference implementation.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index of shape ``[2, num_edges]``.
        edge_weight : torch.Tensor
            Edge weights of shape ``[num_edges]``.
        num_nodes : int
            Number of nodes.
        x : torch.Tensor
            Dense feature matrix of shape ``[num_nodes, channels]``.

        Returns
        -------
        torch.Tensor
            The product :math:`\mathbf{A}\mathbf{X}`.
        """
        row, col = edge_index[0], edge_index[1]
        return scatter(
            edge_weight.unsqueeze(-1) * x[col],
            row,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

    def reaction(self, x, diffusion, x0, edge_index, edge_weight, num_nodes):
        r"""Compute the reaction term :math:`\mathbf{r}(\mathbf{H}, \mathbf{A})`.

        Implements the reaction terms of Eq. (10) of the paper, matching
        ``ODEFuncGread.forward`` of the reference implementation.

        Parameters
        ----------
        x : torch.Tensor
            Current node representations :math:`\mathbf{H}(t)`.
        diffusion : torch.Tensor
            Diffusion term :math:`(\mathbf{A} - \mathbf{I})\mathbf{H}(t)`.
        x0 : torch.Tensor
            Initial node representations :math:`\mathbf{H}(0)`.
        edge_index : torch.Tensor
            Edge index of shape ``[2, num_edges]``.
        edge_weight : torch.Tensor
            Normalized edge weights of shape ``[num_edges]``.
        num_nodes : int
            Number of nodes.

        Returns
        -------
        torch.Tensor
            The reaction term.
        """
        if self.reaction_term == "bspm":
            return -self.spmm(edge_index, edge_weight, num_nodes, diffusion)
        if self.reaction_term == "fisher":
            return -(x - 1) * x
        if self.reaction_term == "allen-cahn":
            return -(x**2 - 1) * x
        if self.reaction_term == "zeldovich":
            return -(x**2 - x) * x
        if self.reaction_term == "st":
            return x0
        if self.reaction_term in ("fb", "fb3"):
            ax = -self.spmm(edge_index, edge_weight, num_nodes, x)
            return x - ax
        return torch.zeros_like(x)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        r"""Forward pass.

        Encodes the input features, integrates the reaction-diffusion
        equation with the explicit Euler method and returns the terminal
        node representations :math:`\mathbf{H}(T)`.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge index of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch assignment vector (unused, present for API
            compatibility).
        edge_weight : torch.Tensor, optional
            Optional edge weights of shape ``[num_edges]``.

        Returns
        -------
        torch.Tensor
            Node representations of shape
            ``[num_nodes, hidden_channels]``.
        """
        num_nodes = x.size(0)
        edge_index, edge_weight = self.normalize_adjacency(
            edge_index, edge_weight, num_nodes
        )

        x = F.dropout(x, self.input_dropout, training=self.training)
        x = self.encoder(x)
        x0 = x

        alpha = torch.sigmoid(self.alpha_train)
        beta = torch.sigmoid(self.beta_train)

        for tau in self.step_sizes:
            ax = self.spmm(edge_index, edge_weight, num_nodes, x)
            diffusion = ax - x
            reaction = self.reaction(
                x, diffusion, x0, edge_index, edge_weight, num_nodes
            )
            if self.beta_diag:
                f = alpha * diffusion + reaction * self.b_w
            elif self.reaction_term == "fb3":
                f = alpha * diffusion + beta * (reaction + x)
            else:
                f = alpha * diffusion + beta * reaction
            if self.add_source:
                f = f + self.source_train * x0
            x = x + tau * f

        if self.xn_activation:
            x = F.relu(x)
        return F.dropout(x, self.dropout, training=self.training)
