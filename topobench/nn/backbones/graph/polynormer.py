r"""Polynormer: a polynomial-expressive graph transformer in linear time.

This module re-implements the Polynormer architecture for the TopoBench
framework. Polynormer learns a high-degree *equivariant polynomial* on the node
features whose coefficients are produced by attention. It is built from two
modules that are applied sequentially:

* a **local** equivariant-attention module that mixes a node with its graph
  neighbours through a GAT-style sparse attention, and
* a **global** equivariant-attention module that mixes all nodes through a
  *linear* (kernel) attention, i.e. in :math:`O(N d^2)` time instead of the
  :math:`O(N^2 d)` of vanilla softmax attention.

The released reference implementation operates on a single (large) graph. In
TopoBench the model is fed *mini-batches of disjoint graphs*, so the global
attention here is made **batch-aware**: the kernel sums are computed per graph
segment (using the ``batch`` vector) so that nodes never attend across graph
boundaries. For a single graph this reduces exactly to the original
formulation.

Equation numbers below refer to the paper.

References
----------
.. [1] Chenhui Deng, Zichao Yue, Zhiru Zhang. "Polynormer: Polynomial-Expressive
   Graph Transformer in Linear Time." ICLR 2024. https://arxiv.org/abs/2403.01232
.. [2] Official implementation:
   https://github.com/cornell-zhang/polynormer (``model.py``).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter


class PolynormerAttention(torch.nn.Module):
    r"""Batch-aware global linear (kernel) attention module of Polynormer.

    Implements the equivariant global attention of Polynormer (Eq. 6 and Eq. 8
    of [1]_). Each layer computes, per attention head, a linear attention via
    the kernel trick with a sigmoid feature map :math:`\sigma`:

    .. math::
        \mathrm{num} = \sigma(Q)\,\big(\sigma(K)^\top V\big), \qquad
        \mathrm{den} = \sigma(Q)\,\textstyle\sum_i \sigma(K_{i,:}),

    so the attended features are :math:`\mathrm{num} / \mathrm{den}`. This costs
    :math:`O(N d^2)` instead of :math:`O(N^2 d)` because the :math:`N\times N`
    attention matrix is never materialised. The result is normalised, gated by
    a learnable per-layer vector :math:`\beta` as
    :math:`\mathrm{LN}(\mathrm{num}/\mathrm{den})\odot(H+\beta)`, passed through
    a ReLU and a linear layer, and dropped out.

    Unlike the single-graph reference implementation, the kernel sums
    :math:`\sigma(K)^\top V` and :math:`\sum_i \sigma(K_{i,:})` are accumulated
    **per graph** in the batch (segment-wise over the ``batch`` vector), so the
    attention is restricted to each graph and never leaks across the disjoint
    graphs of a mini-batch. For a single graph this is identical to Eq. 6/8.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension *per attention head*. The working width is
        ``hidden_channels * heads``.
    heads : int
        Number of attention heads.
    num_layers : int
        Number of stacked global-attention layers.
    beta : float
        Gating initialisation. If ``beta < 0`` the per-layer gates are learnable
        and squashed by a sigmoid at runtime; otherwise they are held constant
        at ``beta``.
    dropout : float
        Dropout probability applied to each layer's output.
    qk_shared : bool, optional
        If ``True`` (default) the query and key projections are shared (a single
        projection is used for both), as in the reference implementation.

    References
    ----------
    .. [1] Deng, Yue, Zhang. "Polynormer: Polynomial-Expressive Graph
       Transformer in Linear Time." ICLR 2024. https://arxiv.org/abs/2403.01232
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
        # Small constant guarding the (strictly positive) denominator against
        # floating-point underflow; the reference uses no epsilon.
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
        r"""Reset all learnable parameters to their initial values.

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
            Batch assignment vector of shape ``[num_nodes]`` mapping each node to
            its graph. If ``None``, all nodes are treated as a single graph,
            recovering the original (single-graph) formulation.

        Returns
        -------
        torch.Tensor
            Updated node features of shape
            ``[num_nodes, heads * hidden_channels]``.
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

            # Numerator (Eq. 6): per node n, q_n . sum_{m in g(n)} k_m (x) v_m.
            # The per-node outer products k (x) v are summed *within each graph*
            # (segment-sum over ``batch``), then gathered back to every node.
            kv_node = k.unsqueeze(2) * v.unsqueeze(1)  # [N, d, d, h]
            kv = scatter(
                kv_node, batch, dim=0, dim_size=num_graphs, reduce="sum"
            )  # [num_graphs, d, d, h]
            kv_per_node = kv.index_select(0, batch)  # [N, d, d, h]
            num = torch.einsum("ndh,ndmh->nmh", q, kv_per_node)

            # Denominator (Eq. 6): per node n, q_n . sum_{m in g(n)} k_m.
            k_sum = scatter(
                k, batch, dim=0, dim_size=num_graphs, reduce="sum"
            )  # [num_graphs, d, h]
            k_sum_per_node = k_sum.index_select(0, batch)  # [N, d, h]
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


class Polynormer(torch.nn.Module):
    r"""Polynormer backbone: local-to-global equivariant polynomial attention.

    A faithful re-implementation of the Polynormer encoder of [1]_ adapted to
    the TopoBench pipeline. The forward pass runs ``local_layers`` equivariant
    *local* attention layers followed (when ``global_layers > 0``) by a
    :class:`PolynormerAttention` *global* module, returning node embeddings.

    **Local module (Eq. 7).** Each local layer combines a GAT message-passing
    term with a node-wise linear term and a ReLU feature map :math:`H`, then
    applies the Polynormer gated polynomial update

    .. math::
        X \leftarrow (1-\beta)\,\mathrm{LN}\!\big(H \odot X\big) + \beta\, X,

    where :math:`X = \mathrm{GAT}(X, E) + WX`, :math:`H = \mathrm{ReLU}(W_h X)`
    and :math:`\beta` is a learnable per-layer gate. The Hadamard product
    :math:`H \odot X` injects a degree-2 interaction per layer, so stacking
    :math:`L` layers yields up to degree-:math:`2^L` polynomial expressivity
    (Thm. 3.3 of [1]_). The per-layer outputs are summed into ``x_local``.

    **Global module (Eq. 8).** When enabled, :class:`PolynormerAttention` is
    applied to :math:`\mathrm{LN}(\text{x\_local})` to mix information across
    the whole graph in linear time.

    Differences from the reference implementation (documented for the TDL
    challenge correctness criterion):

    * The reference toggles a ``_global`` flag during training (a local warm-up
      followed by enabling the global module). TopoBench runs a single training
      loop, so the local and global modules are trained **jointly**: the global
      module is active whenever ``global_layers > 0``. Setting
      ``global_layers = 0`` gives the faithful local-only Polynormer variant.
    * The reference ends with task-specific prediction heads
      (``pred_local`` / ``pred_global``). Here the backbone instead emits node
      embeddings of width ``out_channels`` via a single output projection; the
      final logits are produced by the TopoBench readout.
    * The global attention is batch-aware (see :class:`PolynormerAttention`) so
      that mini-batches of disjoint graphs are handled correctly.

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int
        Hidden dimension *per attention head*. The working width is
        ``hidden_channels * heads``.
    out_channels : int
        Dimension of the returned node embeddings.
    local_layers : int, optional
        Number of local (GAT-based) attention layers. Default is 3.
    global_layers : int, optional
        Number of global linear-attention layers. If 0, the global module is
        disabled (local-only variant). Default is 2.
    in_dropout : float, optional
        Dropout applied to the raw input features. Default is 0.15.
    dropout : float, optional
        Dropout applied within the local layers. Default is 0.5.
    global_dropout : float, optional
        Dropout applied within the global layers. Default is 0.5.
    heads : int, optional
        Number of attention heads. Default is 1.
    beta : float, optional
        Gating initialisation shared by the local and global modules. If
        ``beta < 0`` the gates are learnable (sigmoid-squashed); otherwise they
        are held constant. Default is -1.0.
    pre_ln : bool, optional
        If ``True``, apply a layer norm at the start of each local layer.
        Default is False.
    qk_shared : bool, optional
        Whether the global module shares its query and key projections. Default
        is True.

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
        dropout: float = 0.5,
        global_dropout: float = 0.5,
        heads: int = 1,
        beta: float = -1.0,
        pre_ln: bool = False,
        qk_shared: bool = True,
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
        self.ln = torch.nn.LayerNorm(width)
        if self.use_global:
            self.global_attn = PolynormerAttention(
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
        r"""Reset all learnable parameters to their initial values.

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
        r"""Compute Polynormer node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Batch assignment vector of shape ``[num_nodes]`` mapping each node to
            its graph. Used to keep the global attention within graph
            boundaries. If ``None``, all nodes are treated as one graph.
        edge_weight : torch.Tensor, optional
            Accepted for API compatibility with the TopoBench ``GNNWrapper`` but
            unused: the local GAT attention learns its own edge coefficients.
        **kwargs : dict
            Additional unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        """
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Equivariant local attention (Eq. 7), summed over layers ---
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

        # --- Equivariant global attention (Eq. 8) ---
        if self.use_global:
            x = self.global_attn(self.ln(x_local), batch)
        else:
            x = x_local

        return self.lin_out(x)
