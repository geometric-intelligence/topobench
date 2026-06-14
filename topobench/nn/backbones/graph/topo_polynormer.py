r"""TopoPolynormer: Polynormer with a random-walk structural encoding.

This backbone augments the Polynormer graph transformer ([1]_) with a small
per-node **structural encoding** that gives a message-passing model access to the
local higher-order structure it is otherwise blind to.

**Why.** Standard message passing is bounded by the 1-Weisfeiler-Leman test,
which *cannot count triangles* (or any non-star substructure) — see Chen et al.
([2]_). Intuitively, a message that only travels along edges can never reveal
whether *two of a node's neighbours are themselves connected* — i.e. whether the
node sits on a **triangle** (a 2-simplex). Giving the model structural features
that encode this local cohesion provably lifts it beyond 1-WL (Graph
Substructure Networks, [3]_). A triangle is a 2-simplex, so this is the minimal,
literal bridge from a graph (1-simplices / edges) to topology (2-simplices) — the
theme of the challenge.

**What (default).** For every node we compute, from ``edge_index`` alone, a
*scale-free* "structural fingerprint" — none of whose channels is the raw answer
to any task:

* ``log1p(degree)`` — connectivity (GraphUniverse graphs come from a
  *degree-corrected* SBM, so degree is part of the generative regime);
* the **local clustering coefficient** ``c_i ∈ [0, 1]`` — the *fraction* of a
  node's neighbour-pairs that are linked (a normalised measure of cohesion);
* **random-walk return probabilities** at steps :math:`k = 2, 3, 4` — drop a
  token on the node and ask how likely a random walk is to be *back home* after
  exactly ``k`` steps. A 3-step return is only possible around a **triangle**, a
  4-step return reflects 4-cycles, so these are a multi-scale, degree-normalised
  fingerprint of the local cycle structure (this is GraphGPS's RWSE, [4]_).

These signals are degree-normalised, so their *meaning* is stable across the
GraphUniverse degree/density grid, and — crucially — their sum is **not** the
graph-level triangle-counting target: the model must *learn* to estimate higher-
order structure from them rather than being handed it.

**Optional ablation.** Setting ``use_triangle_count=True`` additionally injects
the raw per-node triangle count (``t_i`` and ``log1p(t_i)``). Because the
triangle-counting label is ``Σ_i t_i / 3`` and the readout sum-pools, this makes
that task an almost-linear readout — useful as an *expressivity demonstration*
of what perfect 2-simplex information buys, but it trivialises the counting task,
so it is **off by default**.

The structural encoding is injected *inside* the backbone (after the input
projection), which bypasses the feature encoder's per-graph ``GraphNorm``, and is
**batch-aware**: it is computed per graph segment (using the ``batch`` vector) so
structure never mixes across the disjoint graphs of a mini-batch.

References
----------
.. [1] Deng, Yue, Zhang. "Polynormer: Polynomial-Expressive Graph Transformer in
   Linear Time." ICLR 2024. https://arxiv.org/abs/2403.01232
.. [2] Chen, Chen, Villar, Bruna. "Can Graph Neural Networks Count
   Substructures?" NeurIPS 2020. https://arxiv.org/abs/2002.04025
.. [3] Bouritsas, Frasca, Zafeiriou, Bronstein. "Improving Graph Neural Network
   Expressivity via Subgraph Isomorphism Counting" (Graph Substructure
   Networks). https://arxiv.org/abs/2006.09252
.. [4] Rampášek et al. "Recipe for a General, Powerful, Scalable Graph
   Transformer" (GraphGPS / RWSE). NeurIPS 2022. https://arxiv.org/abs/2205.12454
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter

DEFAULT_RW_STEPS: tuple[int, ...] = (2, 3, 4)
TRIANGLE_SCALE = 5.0


def num_struct_channels(
    rw_steps: tuple[int, ...] = DEFAULT_RW_STEPS,
    use_triangle_count: bool = False,
) -> int:
    """Return the number of channels produced by :func:`structural_encoding`.

    Parameters
    ----------
    rw_steps : tuple of int, optional
        Random-walk return-probability steps included in the encoding.
    use_triangle_count : bool, optional
        Whether the two raw triangle-count channels are included.

    Returns
    -------
    int
        The number of structural channels: ``log1p(degree)`` and clustering
        coefficient always, plus two triangle channels if ``use_triangle_count``,
        plus one per random-walk step.
    """
    return 2 + (2 if use_triangle_count else 0) + len(rw_steps)


@torch.no_grad()
def structural_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch: torch.Tensor | None = None,
    rw_steps: tuple[int, ...] = DEFAULT_RW_STEPS,
    use_triangle_count: bool = False,
) -> torch.Tensor:
    r"""Compute the per-node random-walk structural encoding.

    For each node the channels are, in order: ``log1p(degree)``; (if
    ``use_triangle_count``) ``triangles / TRIANGLE_SCALE`` and
    ``log1p(triangles)``; the local clustering coefficient; then the random-walk
    return probability for each step in ``rw_steps``.

    The triangle count of node :math:`i` is
    :math:`t_i = \tfrac{1}{2}\sum_j A_{ij} (A^2)_{ij}` (half the number of edges
    among its neighbours); the clustering coefficient is
    :math:`c_i = 2 t_i / (d_i (d_i - 1)) \in [0, 1]`; and the step-:math:`k`
    return probability is :math:`(P^k)_{ii}` for the row-stochastic random walk
    :math:`P = D^{-1} A`. Everything is computed per graph segment so structure
    does not leak across the disjoint graphs of a batch.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity of shape ``[2, num_edges]``.
    num_nodes : int
        Total number of nodes across the batch.
    batch : torch.Tensor, optional
        Batch assignment of shape ``[num_nodes]``. If ``None`` all nodes are
        treated as a single graph.
    rw_steps : tuple of int, optional
        Random-walk return-probability steps to include. Default ``(2, 3, 4)``.
    use_triangle_count : bool, optional
        If ``True``, also include the raw per-node triangle count (and its
        ``log1p``). Off by default. Default ``False``.

    Returns
    -------
    torch.Tensor
        Structural features of shape
        ``[num_nodes, num_struct_channels(rw_steps, use_triangle_count)]``.
    """
    device = edge_index.device
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    n_channels = num_struct_channels(rw_steps, use_triangle_count)
    feats = torch.zeros(num_nodes, n_channels, device=device)
    if num_nodes == 0:
        return feats

    num_graphs = int(batch.max().item()) + 1
    max_k = max(rw_steps) if rw_steps else 0

    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=False).view(-1)
        ng = int(idx.numel())
        if ng == 0:
            continue

        # Restrict edges to this graph and remap node ids to a local 0..ng-1.
        node_in_g = batch == g
        edge_in_g = node_in_g[edge_index[0]] & node_in_g[edge_index[1]]
        e = edge_index[:, edge_in_g]
        remap = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        remap[idx] = torch.arange(ng, device=device)
        e_local = remap[e]

        adj = torch.zeros(ng, ng, device=device)
        if e_local.numel() > 0:
            adj[e_local[0], e_local[1]] = 1.0
        adj = ((adj + adj.t()) > 0).float()
        adj.fill_diagonal_(0.0)

        deg = adj.sum(dim=1)
        adj_sq = adj @ adj
        triangles = 0.5 * (adj_sq * adj).sum(dim=1)
        denom = (deg * (deg - 1.0)).clamp(min=1.0)
        clustering = (2.0 * triangles / denom).clamp(0.0, 1.0)

        channels = [torch.log1p(deg)]
        if use_triangle_count:
            channels += [triangles / TRIANGLE_SCALE, torch.log1p(triangles)]
        channels.append(clustering)

        if max_k > 0:
            walk = adj * (1.0 / deg.clamp(min=1.0)).unsqueeze(1)  # P = D^-1 A
            power = torch.eye(ng, device=device)
            returns: dict[int, torch.Tensor] = {}
            for k in range(1, max_k + 1):
                power = power @ walk
                if k in rw_steps:
                    returns[k] = torch.diagonal(power)
            channels.extend(returns[k] for k in rw_steps)

        feats[idx] = torch.stack(channels, dim=1)

    return feats


class _GlobalLinearAttention(torch.nn.Module):
    r"""Batch-aware global linear (kernel) attention of Polynormer (Eq. 6/8).

    Linear attention with a sigmoid feature map :math:`\sigma`, computed per
    graph segment so it never attends across the disjoint graphs of a batch:
    :math:`\mathrm{num} = \sigma(Q)(\sigma(K)^\top V)`,
    :math:`\mathrm{den} = \sigma(Q)\sum_i \sigma(K_{i})`, costing
    :math:`O(N d^2)` rather than :math:`O(N^2 d)`.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension per attention head.
    heads : int
        Number of attention heads.
    num_layers : int
        Number of stacked global-attention layers.
    beta : float
        Gating initialisation; learnable (sigmoid-squashed) if ``beta < 0``.
    dropout : float
        Dropout probability applied to each layer's output.
    qk_shared : bool, optional
        Whether query and key projections are shared. Default ``True``.
    """

    def __init__(
        self,
        hidden_channels: int,
        heads: int,
        num_layers: int,
        beta: float,
        dropout: float,
        qk_shared: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared
        self.eps = 1e-6

        width = heads * hidden_channels
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, width))
        else:
            self.betas = torch.nn.Parameter(
                torch.ones(num_layers, width) * self.beta
            )

        self.h_lins = torch.nn.ModuleList()
        self.q_lins = torch.nn.ModuleList() if not qk_shared else None
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.h_lins.append(torch.nn.Linear(width, width))
            if not qk_shared:
                self.q_lins.append(torch.nn.Linear(width, width))
            self.k_lins.append(torch.nn.Linear(width, width))
            self.v_lins.append(torch.nn.Linear(width, width))
            self.lns.append(torch.nn.LayerNorm(width))
        self.lin_out = torch.nn.Linear(width, width)

    def reset_parameters(self):
        r"""Reset all learnable parameters.

        Returns
        -------
        None
            This method updates the module in place.
        """
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Apply the stacked batch-aware global linear-attention layers.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, heads * hidden_channels]``.
        batch : torch.Tensor, optional
            Batch assignment of shape ``[num_nodes]``. If ``None`` all nodes are
            treated as a single graph.

        Returns
        -------
        torch.Tensor
            Updated node features of the same shape as ``x``.
        """
        num_nodes = x.size(0)
        if batch is None:
            batch = x.new_zeros(num_nodes, dtype=torch.long)
        num_graphs = int(batch.max().item()) + 1

        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = torch.sigmoid(self.k_lins[i](x)).view(
                num_nodes, self.hidden_channels, self.heads
            )
            if self.qk_shared:
                q = k
            else:
                q = torch.sigmoid(self.q_lins[i](x)).view(
                    num_nodes, self.hidden_channels, self.heads
                )
            v = self.v_lins[i](x).view(
                num_nodes, self.hidden_channels, self.heads
            )

            kv_node = k.unsqueeze(2) * v.unsqueeze(1)
            kv = scatter(
                kv_node, batch, dim=0, dim_size=num_graphs, reduce="sum"
            )
            kv_per_node = kv.index_select(0, batch)
            num = torch.einsum("ndh,ndmh->nmh", q, kv_per_node)

            k_sum = scatter(k, batch, dim=0, dim_size=num_graphs, reduce="sum")
            k_sum_per_node = k_sum.index_select(0, batch)
            den = torch.einsum("ndh,ndh->nh", q, k_sum_per_node).unsqueeze(1)

            if self.beta < 0:
                beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)

            x = (num / (den + self.eps)).reshape(num_nodes, -1)
            x = self.lns[i](x) * (h + beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class TopoPolynormer(torch.nn.Module):
    r"""Polynormer backbone augmented with a random-walk structural encoding.

    The architecture is the Polynormer local-to-global graph transformer ([1]_):
    ``local_layers`` equivariant local attention layers (a GAT message-passing
    term plus a node-wise linear term, combined by a gated degree-2 polynomial
    update) whose outputs are summed, followed (when ``global_layers > 0``) by a
    :class:`_GlobalLinearAttention` global module. On top of this, a per-node
    structural fingerprint (see :func:`structural_encoding`) is projected and
    **added after the input projection** — inside the backbone, bypassing the
    encoder's ``GraphNorm`` — giving the model degree-normalised "sight" of the
    local cycle structure (clustering and random-walk returns) that 1-WL message
    passing is blind to. The raw triangle count is available as an optional
    ablation (``use_triangle_count``) but is off by default, since injecting it
    would trivialise the graph-level triangle-counting task.

    Differences from the reference Polynormer (documented for the correctness
    criterion): the global attention and the structural encoding are batch-aware
    (per graph segment); the task heads are replaced by a single output
    projection to node embeddings (the TopoBench readout produces the final
    logits); and local+global modules are trained jointly (``global_layers = 0``
    recovers a local-only variant; ``use_struct = False`` recovers plain
    Polynormer).

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int
        Hidden dimension per attention head; working width is
        ``hidden_channels * heads``.
    out_channels : int
        Dimension of the returned node embeddings.
    local_layers : int, optional
        Number of local attention layers. Default is 3.
    global_layers : int, optional
        Number of global linear-attention layers (0 disables). Default is 2.
    in_dropout : float, optional
        Dropout on the raw input features. Default is 0.15.
    dropout : float, optional
        Dropout within the layers. Default is 0.3.
    global_dropout : float, optional
        Dropout within the global layers. Default is 0.3.
    heads : int, optional
        Number of attention heads. Default is 1.
    beta : float, optional
        Gating initialisation; learnable if ``beta < 0``. Default is -1.0.
    pre_ln : bool, optional
        Apply a layer norm at the start of each local layer. Default is False.
    qk_shared : bool, optional
        Whether the global module shares query/key projections. Default True.
    use_struct : bool, optional
        Whether to inject the structural encoding. Default True.
    rw_steps : tuple of int, optional
        Random-walk return-probability steps in the encoding. Default (2, 3, 4).
    use_triangle_count : bool, optional
        Whether to additionally inject the raw triangle count (ablation only).
        Default False.

    References
    ----------
    .. [1] Deng, Yue, Zhang. "Polynormer: Polynomial-Expressive Graph
       Transformer in Linear Time." ICLR 2024. https://arxiv.org/abs/2403.01232
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        local_layers: int = 3,
        global_layers: int = 2,
        in_dropout: float = 0.15,
        dropout: float = 0.3,
        global_dropout: float = 0.3,
        heads: int = 1,
        beta: float = -1.0,
        pre_ln: bool = False,
        qk_shared: bool = True,
        use_struct: bool = True,
        rw_steps: tuple[int, ...] = DEFAULT_RW_STEPS,
        use_triangle_count: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.in_dropout = in_dropout
        self.dropout = dropout
        self.heads = heads
        self.beta = beta
        self.pre_ln = pre_ln
        self.use_global = global_layers > 0
        self.use_struct = use_struct
        self.rw_steps = tuple(rw_steps)
        self.use_triangle_count = use_triangle_count
        self.struct_dim = num_struct_channels(
            self.rw_steps, use_triangle_count
        )

        width = heads * hidden_channels

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers, width))
        else:
            self.betas = torch.nn.Parameter(
                torch.ones(local_layers, width) * self.beta
            )

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.pre_lns = torch.nn.ModuleList() if pre_ln else None
        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(width, width))
            self.local_convs.append(
                GATConv(
                    width,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
                )
            )
            self.lins.append(torch.nn.Linear(width, width))
            self.lns.append(torch.nn.LayerNorm(width))
            if pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(width))

        self.lin_in = torch.nn.Linear(in_channels, width)
        self.struct_in = (
            torch.nn.Linear(self.struct_dim, width) if use_struct else None
        )
        self.ln = torch.nn.LayerNorm(width)
        if self.use_global:
            self.global_attn = _GlobalLinearAttention(
                hidden_channels,
                heads,
                global_layers,
                beta,
                global_dropout,
                qk_shared,
            )
        else:
            self.global_attn = None
        self.lin_out = torch.nn.Linear(width, out_channels)

    def reset_parameters(self):
        r"""Reset all learnable parameters.

        Returns
        -------
        None
            This method updates the module in place.
        """
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        if self.use_struct:
            self.struct_in.reset_parameters()
        self.ln.reset_parameters()
        if self.use_global:
            self.global_attn.reset_parameters()
        self.lin_out.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute TopoPolynormer node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch assignment of shape ``[num_nodes]``. Keeps the structural
            encoding and global attention within graph boundaries. If ``None``,
            all nodes are treated as one graph.
        edge_weight : torch.Tensor, optional
            Accepted for ``GNNWrapper`` compatibility but unused.
        **kwargs : dict
            Additional unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        """
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        x = self.lin_in(x)
        if self.use_struct:
            struct = structural_encoding(
                edge_index,
                x.size(0),
                batch,
                self.rw_steps,
                self.use_triangle_count,
            )
            x = x + self.struct_in(struct)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = F.relu(self.h_lins[i](x))
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = torch.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1 - beta) * self.lns[i](h * x) + beta * x
            x_local = x_local + x

        if self.use_global:
            x = self.global_attn(self.ln(x_local), batch)
        else:
            x = x_local

        return self.lin_out(x)
