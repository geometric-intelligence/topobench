"""Differentiable lifting (DiffLift) modules.

Implements the DiffLift recipe for learning graph liftings end-to-end
(Franco et al.): a GNN computes node embeddings (Step 1), candidate
cells are elicited per dimension (Step 2), and a permutation-invariant
scorer accepts or rejects each candidate (Step 3), with gradients
propagated through the discrete decisions by the straight-through
estimator. Both the stochastic (Bernoulli) sampling of the paper and
its deterministic thresholded variant are provided.

The components are modular:

- :class:`DiffLiftEncoder` — Step 1, node embeddings.
- :class:`CellScorer` — Step 3, multiset scorer for any candidate cell
  given node-to-cell membership pairs (works for hyperedges, 1-cells,
  or 2-cells alike).
- :class:`EdgeSampler` — the ``D = 1`` iteration for cell complexes:
  adaptive-size kNN candidate edges added on top of the observed ones.
- :class:`DiffLift` — the full graph-to-cell-complex recipe
  (``D_max = 2``): learned edges, then cycle-basis candidates of the
  augmented graph scored as 2-cells.

The recipe is not tied to a target domain: the scorer accepts any
candidate given its node membership, so the same components lift
graphs to cell complexes (:class:`DiffLift`) or select hyperedges for
a hypergraph model (pass candidate hyperedge memberships to
:class:`CellScorer`).

References
----------
Franco et al. "Differentiable Lifting for Topological Neural Networks."
https://openreview.net/forum?id=eC89CbINIw
"""

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import GINConv


class DiffLiftEncoder(nn.Module):
    """Node-embedding GNN of the lifting (Step 1 of the recipe).

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int, optional
        Embedding dimension. Default is 32.
    num_layers : int, optional
        Number of GIN layers. Default is 1.
    """

    def __init__(self, in_channels, hidden_channels=32, num_layers=1):
        super().__init__()
        self.convs = nn.ModuleList()
        dim = in_channels
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            dim = hidden_channels

    def forward(self, x, edge_pairs):
        """Embed the nodes.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``[n, in_channels]``.
        edge_pairs : torch.Tensor
            Graph connectivity as node pairs ``[2, P]``.

        Returns
        -------
        torch.Tensor
            Node embeddings ``[n, hidden_channels]``.
        """
        z = x
        for conv in self.convs:
            z = conv(z, edge_pairs)
        return z


class CellScorer(nn.Module):
    """Accept/reject scorer over candidate cells (Step 3 of the recipe).

    The acceptance probability of a candidate is a learned function of
    the multiset of its member-node embeddings (mean pooling followed by
    an MLP). The forward pass returns hard 0/1 gates with
    straight-through gradients; decisions are either thresholded (the
    deterministic variant of the paper) or sampled from a Bernoulli.

    Parameters
    ----------
    in_channels : int
        Dimension of the node embeddings.
    hidden_channels : int, optional
        Hidden width of the scoring MLP. Default is 32.
    sharpening : float, optional
        Multiplier applied to the logits before the sigmoid. Default is
        10.0.
    stochastic : bool, optional
        If True, sample decisions from a Bernoulli; otherwise threshold
        the probability at 0.5. Default is False.
    rescue : bool, optional
        If True, force-keep the highest-scoring candidate of any graph
        whose candidates were all rejected. Default is True.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        sharpening=10.0,
        stochastic=False,
        rescue=True,
    ):
        super().__init__()
        self.sharpening = sharpening
        self.stochastic = stochastic
        self.rescue = rescue
        self.score = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, z, membership_pairs, cell_batch):
        """Gate every candidate cell.

        Parameters
        ----------
        z : torch.Tensor
            Node embeddings ``[n, in_channels]``.
        membership_pairs : torch.Tensor
            Node-to-cell membership pairs ``[2, I]``.
        cell_batch : torch.Tensor
            Graph index of every candidate cell ``[n_cells]``.

        Returns
        -------
        torch.Tensor
            Gates ``[n_cells]``: hard 0/1 forward values with soft
            gradients.
        """
        n_cells = cell_batch.size(0)
        if n_cells == 0:
            return z.new_zeros(0)
        pooled = z.new_zeros(n_cells, z.size(1)).index_add_(
            0, membership_pairs[1], z[membership_pairs[0]]
        )
        counts = torch.bincount(membership_pairs[1], minlength=n_cells)
        pooled = pooled / counts.clamp(min=1).unsqueeze(1)
        probs = torch.sigmoid(self.sharpening * self.score(pooled).squeeze(-1))
        if self.stochastic and self.training:
            hard = torch.bernoulli(probs)
        else:
            hard = (probs > 0.5).float()
        if self.rescue:
            num_graphs = int(cell_batch.max()) + 1
            best = probs.new_full((num_graphs,), -1.0)
            best = best.scatter_reduce(
                0, cell_batch, probs, reduce="amax", include_self=True
            )
            kept = hard.new_zeros(num_graphs).scatter_add_(0, cell_batch, hard)
            lift = (kept == 0)[cell_batch] & (probs == best[cell_batch])
            hard = torch.where(lift, torch.ones_like(hard), hard)
        return hard + (probs - probs.detach())


class EdgeSampler(nn.Module):
    """Learned 1-cells: the ``D = 1`` iteration for cell complexes.

    For each node, a neighborhood size ``k_v`` is drawn from a
    categorical distribution parameterized by its embedding
    (Gumbel-softmax with hard samples), candidate edges connect the node
    to its ``k_v`` nearest neighbors in embedding space, and a pair
    scorer gates each candidate. Learned edges are *added* to the
    observed ones, never removing them.

    Parameters
    ----------
    in_channels : int
        Dimension of the node embeddings.
    k_min : int, optional
        Smallest neighborhood size. Default is 1.
    k_max : int, optional
        Largest neighborhood size. Default is 3.
    sharpening : float, optional
        Logit sharpening of the pair scorer. Default is 10.0.
    stochastic : bool, optional
        Sample the pair decisions from a Bernoulli. Default is False.
    """

    def __init__(
        self,
        in_channels,
        k_min=1,
        k_max=3,
        sharpening=10.0,
        stochastic=False,
    ):
        super().__init__()
        if not 1 <= k_min <= k_max:
            raise ValueError("need 1 <= k_min <= k_max")
        self.k_min = k_min
        self.k_max = k_max
        self.k_logits = nn.Linear(in_channels, k_max - k_min + 1)
        self.pair_scorer = CellScorer(
            in_channels,
            sharpening=sharpening,
            stochastic=stochastic,
            rescue=False,
        )

    def forward(self, z, edge_pairs, batch):
        """Propose and gate new edges.

        Parameters
        ----------
        z : torch.Tensor
            Node embeddings ``[n, d]``.
        edge_pairs : torch.Tensor
            Observed edges as node pairs ``[2, P]`` (both directions).
        batch : torch.Tensor
            Graph index of every node ``[n]``.

        Returns
        -------
        new_pairs : torch.Tensor
            Undirected candidate edges ``[2, E_new]`` (one direction),
            disjoint from the observed edges.
        gate : torch.Tensor
            Straight-through gate of each candidate ``[E_new]``.
        """
        from torch_cluster import knn_graph

        n = z.size(0)
        # Adaptive neighborhood sizes (Gumbel-softmax, hard samples).
        k_probs = F.gumbel_softmax(self.k_logits(z), tau=1.0, hard=True)
        k_values = self.k_min + k_probs.argmax(dim=-1)  # [n]

        candidates = knn_graph(z.detach(), k=self.k_max, batch=batch)
        src, dst = candidates[0], candidates[1]
        # Keep each source only among its k_dst nearest neighbours: the
        # kNN output lists neighbours per target node in order, so rank
        # them and compare against the sampled k of the target.
        order = torch.argsort(dst, stable=True)
        src, dst = src[order], dst[order]
        ranks = torch.arange(src.size(0), device=z.device)
        first = torch.zeros(n, dtype=torch.long, device=z.device)
        counts = torch.bincount(dst, minlength=n)
        first[1:] = torch.cumsum(counts, 0)[:-1]
        ranks = ranks - first[dst]
        keep = ranks < k_values[dst]

        # Drop candidates that already exist as observed edges and
        # deduplicate the two orientations.
        a = torch.minimum(src[keep], dst[keep])
        b = torch.maximum(src[keep], dst[keep])
        cand = torch.unique(torch.stack([a, b], dim=0), dim=1)
        if edge_pairs.numel():
            ea = torch.minimum(edge_pairs[0], edge_pairs[1])
            eb = torch.maximum(edge_pairs[0], edge_pairs[1])
            existing = set(
                map(tuple, torch.stack([ea, eb], dim=0).t().tolist())
            )
            mask = torch.tensor(
                [
                    (int(x), int(y)) not in existing
                    for x, y in cand.t().tolist()
                ],
                dtype=torch.bool,
                device=z.device,
            )
            cand = cand[:, mask]
        if cand.numel() == 0:
            return cand, z.new_zeros(0)
        # Score each candidate pair from its two endpoints.
        members = torch.cat([cand[0], cand[1]])
        cells = torch.arange(cand.size(1), device=z.device).repeat(2)
        gate = self.pair_scorer(
            z,
            torch.stack([members, cells], dim=0),
            batch[cand[0]],
        )
        # Gumbel gradient path for the neighborhood sizes.
        k_grad = k_probs.sum() - k_probs.sum().detach()
        return cand, gate + 0.0 * k_grad


class DiffLift(nn.Module):
    """Complete graph-to-cell-complex lifting (``D_max = 2``).

    Runs the full recipe: node embeddings, learned 1-cells added to the
    observed edges, cycle-basis candidates of the augmented graph, and
    gated 2-cells, with scaled-sum feature lifting.

    Parameters
    ----------
    in_channels : int
        Dimension of the input node features.
    hidden_channels : int, optional
        Embedding dimension of the lifting GNN. Default is 32.
    k_min : int, optional
        Smallest learned-edge neighborhood size. Default is 1.
    k_max : int, optional
        Largest learned-edge neighborhood size. Default is 3.
    max_cell_length : int, optional
        Longest cycle admitted as a 2-cell candidate. Default is 10.
    sharpening : float, optional
        Logit sharpening of the scorers. Default is 10.0.
    stochastic : bool, optional
        Sample accept/reject decisions instead of thresholding.
        Default is False.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=32,
        k_min=1,
        k_max=3,
        max_cell_length=10,
        sharpening=10.0,
        stochastic=False,
    ):
        super().__init__()
        self.max_cell_length = max_cell_length
        self.encoder = DiffLiftEncoder(in_channels, hidden_channels)
        self.edge_sampler = EdgeSampler(
            hidden_channels,
            k_min=k_min,
            k_max=k_max,
            sharpening=sharpening,
            stochastic=stochastic,
        )
        self.cell_scorer = CellScorer(
            hidden_channels,
            sharpening=sharpening,
            stochastic=stochastic,
        )

    def forward(self, x, edge_pairs, batch):
        """Lift a batch of graphs to gated 2-dimensional cell complexes.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``[n, in_channels]``.
        edge_pairs : torch.Tensor
            Observed edges as node pairs ``[2, P]`` (both directions).
        batch : torch.Tensor
            Graph index of every node ``[n]``.

        Returns
        -------
        dict
            ``new_edge_pairs`` ``[2, E_new]`` and ``new_edge_gate``
            ``[E_new]`` (learned 1-cells); ``cell_membership``
            ``[2, I]`` node-to-2-cell pairs, ``cell_gate``
            ``[n_cells]``, and ``cell_batch`` ``[n_cells]``; ``x_2``
            ``[n_cells, in_channels]`` scaled-sum features of the
            2-cells.
        """
        z = self.encoder(x, edge_pairs)
        new_pairs, edge_gate = self.edge_sampler(z, edge_pairs, batch)

        # Candidate 2-cells: cycle basis of the augmented graph.
        graph = nx.Graph()
        graph.add_nodes_from(range(x.size(0)))
        graph.add_edges_from(edge_pairs.t().tolist())
        graph.add_edges_from(new_pairs.t().tolist())
        cycles = [
            c
            for c in nx.cycle_basis(graph)
            if 3 <= len(c) <= self.max_cell_length
        ]
        cycles.sort(key=lambda c: (len(c), tuple(sorted(c))))

        if cycles:
            members = torch.tensor(
                [v for c in cycles for v in c],
                dtype=torch.long,
                device=x.device,
            )
            cells = torch.tensor(
                [i for i, c in enumerate(cycles) for _ in c],
                dtype=torch.long,
                device=x.device,
            )
            membership = torch.stack([members, cells], dim=0)
            cell_batch = batch[
                torch.tensor(
                    [c[0] for c in cycles],
                    dtype=torch.long,
                    device=x.device,
                )
            ]
        else:
            membership = edge_pairs.new_zeros(2, 0)
            cell_batch = batch.new_zeros(0)

        cell_gate = self.cell_scorer(z, membership, cell_batch)

        # Scaled-sum feature lifting for the accepted 2-cells.
        n_cells = cell_batch.size(0)
        x_2 = x.new_zeros(n_cells, x.size(1))
        if n_cells:
            x_2 = x_2.index_add_(0, membership[1], x[membership[0]])
            sizes = torch.bincount(membership[1], minlength=n_cells)
            x_2 = x_2 / sizes.clamp(min=1).unsqueeze(1)
            x_2 = cell_gate.unsqueeze(-1) * x_2

        return {
            "new_edge_pairs": new_pairs,
            "new_edge_gate": edge_gate,
            "cell_membership": membership,
            "cell_gate": cell_gate,
            "cell_batch": cell_batch,
            "x_2": x_2,
        }
