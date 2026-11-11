r"""Copresheaf Topological Neural Networks (CTNNs) on a combinatorial complex.

A CTNN [1] equips every cell of a combinatorial complex [2] with its own
feature space (a stalk) and every directed neighbour relation ``y -> x`` with a
learnable transport map ``rho_{y->x}: F(y) -> F(x)``. Messages are transported
into the receiver's own frame *before* aggregation, which is what makes the
propagation anisotropic and direction-aware, in contrast to a cellular sheaf,
which glues data through a shared edge stalk and is therefore symmetric.

Definition 10 of [1] states the higher-order layer over a collection of
neighborhood functions ``N = {N_k}``:

.. math::
    \mathbf{h}_x^{(\ell+1)} = \beta\Biggl(
        \mathbf{h}_x^{(\ell)},
        \bigotimes_{k=1}^{n} \bigoplus_{y \in \mathcal{N}_k(x)}
        \alpha_{\mathcal{N}_k}\bigl(
            \mathbf{h}_x^{(\ell)},
            \rho_{y \to x}^{\mathcal{N}_k}(\mathbf{h}_y^{(\ell)})
        \bigr)
    \Biggr)

The concrete instantiation implemented here is the one [1] evaluates on
topological domains, the *Copresheaf Cellular Transformer* of [1], Appendix
H.5: the message function ``alpha`` is copresheaf attention and ``beta`` is a
residual MLP with normalisation. Within a rank this is copresheaf
self-attention, [1], Definition 11, Eq. (7); across ranks it is copresheaf
cross-attention, [1], Definition 16, Eq. (12). Both compute

.. math::
    m_x = \sum_{y \in \mathcal{N}_k(x)} a_{xy}\, \rho_{y \to x}(v_y),
    \qquad
    a_{xy} = \frac{\exp(\langle q_x, k_y \rangle / \sqrt{p})}
                  {\sum_{y' \in \mathcal{N}_k(x)}
                   \exp(\langle q_x, k_{y'} \rangle / \sqrt{p})}

with ``q_x = W_q h_x``, ``k_y = W_k h_y``, ``v_y = W_v h_y``. Together the two
cases are the body of [1], Algorithm 1, restricted to the sparse neighborhoods
that the complex actually provides. For rank ``0`` and the neighborhoods of
[1], Appendix H.5 this is exactly its displayed layer, whose ``0 <- 0``,
``0 <- 1`` and ``0 <- 2`` message paths are the routes ``up_adjacency-0``,
``down_incidence-1`` and ``2-down_incidence-2`` of
:attr:`CopresheafTNN.neighborhoods`. The last is a *direct* rank-2 to rank-0
transport, as [1], Appendix H.5 writes it; TopoBench materialises it as the
composite ``incidence_1 . incidence_2``, whose support is the set of pairs
``(v, t)`` with ``v`` a vertex of the triangle ``t``.

The transport maps come from the catalogue of [1], Table 18, exposed as
:data:`COPRESHEAF_MAPS`. They act stalk-wise and are evaluated independently
per attention head, so ``rho`` is a ``head_dim x head_dim`` operator
conditioned on the pair ``(q_x, k_y)``. Setting ``rho = Id`` recovers the
Cellular Transformer of [3], as [1], Appendix H.5 observes.

Four conventions are worth stating because [1] leaves them open. The
inter-neighborhood combination ``otimes`` of Definition 10 is summation, which
is how [1], Appendix H.5 writes it. The query/key width ``p`` equals the stalk
width ``head_dim``, which is the setting [1], Table 18 assumes when it sizes
the transport parameters as ``W in R^{2d x d^2}``. ``beta`` runs once per
layer over the combined message, following Definition 10 and the displayed
layer of Appendix H.5; [1], Algorithm 1 instead writes two sequential updates,
one after self-attention and one after cross-attention, which would make the
result depend on the order the neighborhoods happen to be listed in. And the
per-head messages are concatenated with no output projection: [1], Algorithm 1
writes none, though the cost accounting under [1], Table 19 charges an
``O(n d^2)`` step to "combine head outputs". The ``beta`` that immediately
follows is a full-width MLP, so heads are mixed there instead.

[1] Hajij, Bastian, Osentoski, Kabaria, Davenport, Dawood, Cherukuri,
    Kocheemoolayil, Shahmansouri, Lew, Papamarkou, Birdal. "Copresheaf
    Topological Neural Networks: A Generalized Deep Learning Framework."
    NeurIPS 2025. https://arxiv.org/abs/2505.21251
[2] Hajij, Zamzmi, Papamarkou, Miolane, Guzman-Saenz, Ramamurthy, Birdal,
    Schaub. "Topological Deep Learning: Going Beyond Graph Data."
    https://arxiv.org/abs/2206.00606
[3] Ballester, Barsbey, Papillon, Battiloro, Hajij, Miolane, Birdal.
    "Attention Mechanisms for Topological Deep Learning: The Case of Cellular
    Transformers." https://arxiv.org/abs/2405.14094
"""

import math

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax

from topobench.data.utils import get_routes_from_neighborhoods


class HeadwiseLinear(nn.Module):
    r"""Linear map with independent weights per attention head.

    The transport maps of [1], Table 18 are "evaluated independently for each
    attention head", so their parameters carry a leading head axis instead of
    being shared the way :class:`torch.nn.Linear` shares them.

    Parameters
    ----------
    heads : int
        Number of attention heads.
    in_features : int
        Width of the per-head input.
    out_features : int
        Width of the per-head output.
    bias : bool, optional
        Whether to add a per-head bias. Default is True.
    zero_init : bool, optional
        If True, initialise weight and bias to zero. [1], Table 18 zero-initialises
        the SheafFC and SheafMLP maps so that ``rho = Id`` at the start of
        training. Default is False.
    """

    def __init__(
        self, heads, in_features, out_features, bias=True, zero_init=False
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(heads, in_features, out_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(heads, out_features)) if bias else None
        )
        self.zero_init = zero_init
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise the per-head weight and bias."""
        if self.zero_init:
            nn.init.zeros_(self.weight)
        else:
            # Same 1/sqrt(fan_in) scale as torch.nn.Linear, applied per head.
            bound = 1.0 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        r"""Apply the per-head weights.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[num_edges, heads, in_features]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[num_edges, heads, out_features]``.
        """
        out = torch.einsum("ehi,hio->eho", x, self.weight)
        return out if self.bias is None else out + self.bias


class SheafFCMap(nn.Module):
    r"""SheafFC transport map of [1], Table 18.

    .. math::
        \rho_{y \to x} = \mathrm{Id}
            + \tanh\bigl(W\,[q_x;\, k_y]\bigr),
        \qquad W \in \mathbb{R}^{2d \times d^2}

    ``W`` is zero-initialised, so every map starts at the identity and the layer
    starts as the plain Cellular Transformer of [3]. This is the map [1],
    Appendix H.5 instantiates for its simplicial-complex experiment.

    Parameters
    ----------
    heads : int
        Number of attention heads.
    stalk_dim : int
        Stalk width ``d`` per head, i.e. the size of the square transport map.
    """

    is_diagonal = False

    def __init__(self, heads, stalk_dim):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.linear = HeadwiseLinear(
            heads, 2 * stalk_dim, stalk_dim**2, bias=False, zero_init=True
        )

    def forward(self, query, key):
        r"""Build one transport map per directed neighbour pair.

        Parameters
        ----------
        query : torch.Tensor
            Receiver queries ``q_x`` of shape ``[num_edges, heads, stalk_dim]``.
        key : torch.Tensor
            Sender keys ``k_y`` of shape ``[num_edges, heads, stalk_dim]``.

        Returns
        -------
        torch.Tensor
            Transport maps of shape
            ``[num_edges, heads, stalk_dim, stalk_dim]``.
        """
        delta = self.linear(torch.cat([query, key], dim=-1))
        delta = delta.unflatten(-1, (self.stalk_dim, self.stalk_dim))
        return torch.eye(
            self.stalk_dim, device=delta.device, dtype=delta.dtype
        ) + torch.tanh(delta)


class SheafSPDMap(nn.Module):
    r"""SheafSPD transport map of [1], Table 18.

    .. math::
        \rho_{y \to x} = \mathrm{Id} + QQ^{\top},
        \qquad Q = W\,[q_x;\, k_y],
        \qquad W \in \mathbb{R}^{2d \times d^2}

    ``QQ^T`` is positive semidefinite, so ``rho`` is symmetric positive definite
    with all eigenvalues at least one: transport can stretch a stalk but never
    reflect or contract it. [1], Appendix H.5 offers this as the constrained
    alternative to :class:`SheafFCMap`, and [1], Appendix H.1 attributes the
    gain on viscous-diffusion problems to exactly this alignment with a
    diffusion tensor. ``W`` carries no bias, per [1], Table 18.

    Parameters
    ----------
    heads : int
        Number of attention heads.
    stalk_dim : int
        Stalk width ``d`` per head, i.e. the size of the square transport map.
    """

    is_diagonal = False

    def __init__(self, heads, stalk_dim):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.linear = HeadwiseLinear(
            heads, 2 * stalk_dim, stalk_dim**2, bias=False
        )

    def forward(self, query, key):
        r"""Build one symmetric positive definite map per neighbour pair.

        Parameters
        ----------
        query : torch.Tensor
            Receiver queries ``q_x`` of shape ``[num_edges, heads, stalk_dim]``.
        key : torch.Tensor
            Sender keys ``k_y`` of shape ``[num_edges, heads, stalk_dim]``.

        Returns
        -------
        torch.Tensor
            Transport maps of shape
            ``[num_edges, heads, stalk_dim, stalk_dim]``.
        """
        factor = self.linear(torch.cat([query, key], dim=-1))
        factor = factor.unflatten(-1, (self.stalk_dim, self.stalk_dim))
        return torch.eye(
            self.stalk_dim, device=factor.device, dtype=factor.dtype
        ) + factor @ factor.transpose(-1, -2)


class DiagonalMLPMap(nn.Module):
    r"""Diagonal MLP transport map of [1], Table 18.

    .. math::
        \rho_{y \to x} = \mathrm{diag}\bigl(
            \sigma(\mathrm{MLP}[q_x, k_y])
        \bigr)

    with ``sigma`` the logistic function and a two-layer MLP of widths
    ``2d -> 2d -> d``. Restricting transport to the diagonal replaces the
    ``d^2`` parameters and the ``O(d^2)`` per-edge application of the full maps
    by ``O(d)`` of each, the reduction noted under [1], Table 19. It is also the
    parameterisation the copresheaf GCN and GraphSAGE of [1], Section 6.2 use.

    Parameters
    ----------
    heads : int
        Number of attention heads.
    stalk_dim : int
        Stalk width ``d`` per head, i.e. the length of the transport diagonal.
    """

    is_diagonal = True

    def __init__(self, heads, stalk_dim):
        super().__init__()
        self.hidden = HeadwiseLinear(heads, 2 * stalk_dim, 2 * stalk_dim)
        self.out = HeadwiseLinear(heads, 2 * stalk_dim, stalk_dim)

    def forward(self, query, key):
        r"""Build one transport diagonal per directed neighbour pair.

        Parameters
        ----------
        query : torch.Tensor
            Receiver queries ``q_x`` of shape ``[num_edges, heads, stalk_dim]``.
        key : torch.Tensor
            Sender keys ``k_y`` of shape ``[num_edges, heads, stalk_dim]``.

        Returns
        -------
        torch.Tensor
            Transport diagonals of shape ``[num_edges, heads, stalk_dim]``.
        """
        hidden = torch.relu(self.hidden(torch.cat([query, key], dim=-1)))
        return torch.sigmoid(self.out(hidden))


#: The subset of the transport-map catalogue of [1], Table 18 that [1] applies
#: to topological domains. ``sheaf_fc`` is what [1], Appendix H.5 instantiates,
#: ``sheaf_spd`` is the positive-definite constraint it offers alongside, and
#: ``diagonal`` is the ``O(d)`` map of [1], Section 6.2, kept because the full
#: maps cost one ``d x d`` matrix per edge, per head and per neighborhood.
COPRESHEAF_MAPS = {
    "sheaf_fc": SheafFCMap,
    "sheaf_spd": SheafSPDMap,
    "diagonal": DiagonalMLPMap,
}


class CopresheafAttention(nn.Module):
    r"""Copresheaf attention along a single neighborhood function.

    Realises the message function ``alpha_{N_k}`` of [1], Definition 10 as
    attention that transports each value vector into the receiver's stalk before
    weighting it. When sender and receiver share a rank this is copresheaf
    self-attention, [1], Definition 11; when they do not it is copresheaf
    cross-attention, [1], Definition 16. The two differ only in which feature
    matrices the projections read, so a single module covers both, matching the
    shared body of the two loops of [1], Algorithm 1.

    [1], Definition 16 allows the sender and receiver stalks to have different
    widths. Every rank is projected to a common width by the feature encoder
    upstream, so ``channels`` is used for both here.

    Parameters
    ----------
    channels : int
        Feature width of both sender and receiver cells.
    heads : int
        Number of attention heads. Must divide ``channels``.
    copresheaf_map : str
        Key of :data:`COPRESHEAF_MAPS` selecting the transport parameterisation.
    """

    def __init__(self, channels, heads, copresheaf_map):
        super().__init__()
        self.heads = heads
        self.stalk_dim = channels // heads
        self.lin_query = nn.Linear(channels, channels, bias=False)
        self.lin_key = nn.Linear(channels, channels, bias=False)
        self.lin_value = nn.Linear(channels, channels, bias=False)
        self.transport = COPRESHEAF_MAPS[copresheaf_map](heads, self.stalk_dim)

    def _project(self, linear, x, index):
        r"""Project features and gather them onto the neighbour pairs.

        Parameters
        ----------
        linear : torch.nn.Linear
            One of the query, key or value projections.
        x : torch.Tensor
            Cell features of shape ``[num_cells, channels]``.
        index : torch.Tensor
            Cell index of each neighbour pair, of shape ``[num_edges]``.

        Returns
        -------
        torch.Tensor
            Per-pair projections of shape
            ``[num_edges, heads, stalk_dim]``.
        """
        projected = linear(x).view(-1, self.heads, self.stalk_dim)
        return projected[index]

    def forward(self, x_receiver, x_sender, neighborhood):
        r"""Aggregate transported messages over one neighborhood.

        Parameters
        ----------
        x_receiver : torch.Tensor
            Features of the receiving cells, shape ``[num_receivers, channels]``.
        x_sender : torch.Tensor
            Features of the sending cells, shape ``[num_senders, channels]``.
        neighborhood : torch.Tensor
            Sparse neighborhood matrix of shape ``[num_receivers, num_senders]``
            whose nonzero at ``(x, y)`` marks ``y in N_k(x)``, i.e. a directed
            edge ``y -> x`` of the induced graph ``G_{N_k}`` of [1], Section 4.

        Returns
        -------
        torch.Tensor
            Messages ``m_x`` of shape ``[num_receivers, channels]``.
        """
        # Only the support is read: [1], Definition 7 builds the copresheaf
        # neighborhood matrix by replacing each *nonzero* of the binary matrix
        # of Definition 3 with a map, so the stored scalar carries no
        # information beyond being nonzero. The zeros have to be filtered
        # rather than taken from the sparsity pattern, because TopoBench
        # materialises `up_adjacency-r` straight from toponetx, which stores an
        # explicit zero on the diagonal; taking `indices()` alone would read
        # those as a self-loop `y = x` on every cell.
        neighborhood = neighborhood.coalesce()
        support = neighborhood.values().flatten() != 0
        receiver, sender = neighborhood.indices()[:, support]
        num_receivers = x_receiver.size(0)

        query = self._project(self.lin_query, x_receiver, receiver)
        key = self._project(self.lin_key, x_sender, sender)
        value = self._project(self.lin_value, x_sender, sender)

        # a_xy = softmax_{y in N_k(x)}(<q_x, k_y> / sqrt(p)), Eq. (7) and (12).
        score = (query * key).sum(dim=-1) / math.sqrt(self.stalk_dim)
        attention = softmax(score, receiver, num_nodes=num_receivers)

        # rho_{y->x}(v_y), the transport that distinguishes a copresheaf layer
        # from plain attention.
        rho = self.transport(query, key)
        if self.transport.is_diagonal:
            transported = rho * value
        else:
            transported = torch.einsum("ehij,ehj->ehi", rho, value)

        message = attention.unsqueeze(-1) * transported
        aggregated = scatter(
            message, receiver, dim=0, dim_size=num_receivers, reduce="sum"
        )
        return aggregated.flatten(start_dim=1)


class CopresheafTNNLayer(nn.Module):
    r"""One layer of copresheaf-based higher-order message passing.

    Implements [1], Definition 10 for a fixed collection of neighborhood
    functions: every neighborhood contributes its own attention head group and
    its own transport maps, the resulting messages are combined per receiving
    rank by the inter-neighborhood aggregation ``otimes``, and the update
    ``beta`` refreshes the cells that received anything.

    ``otimes`` is summation and ``beta`` is a residual MLP with normalisation,
    both as written in [1], Appendix H.5. Cells of a rank that no neighborhood
    targets are passed through unchanged, since [1], Definition 10 leaves
    ``h_x`` untouched when every ``N_k(x)`` is empty.

    Parameters
    ----------
    channels : int
        Feature width shared by all ranks.
    routes : list of list of int
        One ``[sender_rank, receiver_rank]`` pair per neighborhood, in the order
        the neighborhoods are supplied.
    heads : int
        Number of attention heads per neighborhood.
    copresheaf_map : str
        Key of :data:`COPRESHEAF_MAPS` selecting the transport parameterisation.
    dropout : float
        Dropout applied to the aggregated message and to the update MLP.
    """

    def __init__(self, channels, routes, heads, copresheaf_map, dropout):
        super().__init__()
        self.routes = routes
        self.attentions = nn.ModuleList(
            CopresheafAttention(channels, heads, copresheaf_map)
            for _ in routes
        )
        self.dropout = nn.Dropout(dropout)

        receiver_ranks = sorted({receiver for _, receiver in routes})
        self.norm_message = nn.ModuleDict(
            {str(rank): nn.LayerNorm(channels) for rank in receiver_ranks}
        )
        self.norm_update = nn.ModuleDict(
            {str(rank): nn.LayerNorm(channels) for rank in receiver_ranks}
        )
        self.update = nn.ModuleDict(
            {
                str(rank): nn.Sequential(
                    nn.Linear(channels, 2 * channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * channels, channels),
                )
                for rank in receiver_ranks
            }
        )

    def forward(self, features, neighborhoods):
        r"""Update the cell features of every rank a neighborhood targets.

        Parameters
        ----------
        features : dict
            Cell features keyed by rank, each of shape ``[num_cells, channels]``.
        neighborhoods : list of torch.Tensor
            Sparse neighborhood matrices, aligned with ``routes``.

        Returns
        -------
        dict
            Updated cell features, keyed by rank.
        """
        messages = {}
        for attention, (sender, receiver), neighborhood in zip(
            self.attentions, self.routes, neighborhoods, strict=True
        ):
            message = attention(
                features[receiver], features[sender], neighborhood
            )
            # otimes: accumulate across the neighborhoods sharing a receiver.
            messages[receiver] = messages.get(receiver, 0) + message

        updated = dict(features)
        for rank, message in messages.items():
            key = str(rank)
            hidden = self.norm_message[key](
                features[rank] + self.dropout(message)
            )
            updated[rank] = self.norm_update[key](
                hidden + self.dropout(self.update[key](hidden))
            )
        return updated


class CopresheafTNN(nn.Module):
    r"""Copresheaf Topological Neural Network.

    Stacks ``layers`` copies of :class:`CopresheafTNNLayer` over the
    neighborhood collection ``N = {N_k}`` of [1], Definition 10. ``neighborhoods``
    *is* that collection: each entry names one neighborhood matrix that the
    lifting materialises, and TopoBench's route parser turns it into the sender
    and receiver ranks. So the choice of neighborhoods, not the code, decides
    which of the copresheaf adjacency and incidence matrices of [1],
    Definition 8 the model transports along.

    Parameters
    ----------
    channels : int
        Feature width shared by all ranks. Must be divisible by ``heads``.
    neighborhoods : list of str
        Neighborhood names, e.g. ``up_adjacency-0`` or ``down_incidence-2``.
    layers : int, optional
        Number of message-passing layers. Default is 2.
    heads : int, optional
        Number of attention heads per neighborhood. Default is 4.
    copresheaf_map : str, optional
        Key of :data:`COPRESHEAF_MAPS` selecting the transport parameterisation.
        Default is ``'sheaf_fc'``, the map of [1], Appendix H.5.
    dropout : float, optional
        Dropout applied inside every layer. Default is 0.0.

    Raises
    ------
    ValueError
        If ``neighborhoods`` is empty, if ``copresheaf_map`` is unknown, or if
        ``heads`` does not divide ``channels``.
    """

    def __init__(
        self,
        channels,
        neighborhoods,
        layers=2,
        heads=4,
        copresheaf_map="sheaf_fc",
        dropout=0.0,
    ):
        super().__init__()

        if not neighborhoods:
            raise ValueError(
                "neighborhoods must name at least one neighborhood function; "
                "it is the collection N of Definition 10"
            )
        if copresheaf_map not in COPRESHEAF_MAPS:
            raise ValueError(
                f"Unknown copresheaf_map {copresheaf_map!r}, "
                f"expected one of {sorted(COPRESHEAF_MAPS)}"
            )
        if heads < 1 or channels % heads != 0:
            raise ValueError(
                f"heads={heads} must be positive and divide channels="
                f"{channels}, otherwise the per-head stalk width truncates"
            )

        self.neighborhoods = list(neighborhoods)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        self.max_rank = max(max(route) for route in self.routes)
        self.channels = channels
        self.copresheaf_map = copresheaf_map
        self.layers = nn.ModuleList(
            CopresheafTNNLayer(
                channels, self.routes, heads, copresheaf_map, dropout
            )
            for _ in range(layers)
        )

    def forward(self, batch):
        r"""Propagate the features of every rank through the complex.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batched complex carrying ``x_{rank}`` features and one sparse matrix
            per entry of ``neighborhoods``. Batching keeps every neighborhood
            matrix block diagonal, so attention never crosses complexes.

        Returns
        -------
        dict
            Cell features keyed by rank, from ``0`` to ``max_rank``.
        """
        features = {
            rank: batch[f"x_{rank}"] for rank in range(self.max_rank + 1)
        }
        neighborhoods = [batch[name] for name in self.neighborhoods]

        for layer in self.layers:
            features = layer(features, neighborhoods)
        return features
