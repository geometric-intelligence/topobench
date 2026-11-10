"""Relative Random Walk Probabilities (RRWP) positional encodings.

This transform implements the RRWP positional encodings introduced in
"Graph Inductive Biases in Transformers without Message Passing"
(GRIT) [1]_, Eq. (1):

.. math::
    \\mathbf{P}_{i,j} = [\\mathbf{I}, \\mathbf{M}, \\mathbf{M}^2, \\ldots,
    \\mathbf{M}^{K-1}]_{i,j} \\in \\mathbb{R}^{K},
    \\quad \\mathbf{M} = \\mathbf{D}^{-1}\\mathbf{A},

where :math:`\\mathbf{M}_{i,j}` is the probability that a simple random
walk starting at node :math:`i` reaches node :math:`j` in one step. The
diagonal :math:`\\mathbf{P}_{i,i}` coincides with the Random Walk
Structural Encodings (RWSE) and is used as a node-level (absolute)
positional encoding, while the off-diagonal entries provide relative
node-pair encodings.

References
----------
.. [1] Ma, Lin, Lim, Romero-Soriano, Dokania, Coates, Torr, Lim.
    "Graph Inductive Biases in Transformers without Message Passing."
    ICML 2023. https://arxiv.org/abs/2305.17589
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


def compute_rrwp(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute RRWP encodings for a single graph.

    The relative encodings are returned in sparse (COO) form, keeping only
    node pairs :math:`(i, j)` for which at least one of the ``walk_length``
    random walk probabilities is non-zero. Following the reference GRIT
    implementation, the returned pair index is oriented as
    ``(source=j, target=i)`` with value :math:`\mathbf{P}_{i,j}`, so that
    during attention node :math:`i` attends to node :math:`j` conditioned
    on the probabilities of walks from :math:`i` to :math:`j`.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape ``[2, num_edges]``.
    num_nodes : int
        Number of nodes in the graph.
    walk_length : int
        Number of random walk channels :math:`K` (including the identity,
        i.e. powers :math:`\mathbf{M}^0, \ldots, \mathbf{M}^{K-1}`).

    Returns
    -------
    abs_pe : torch.Tensor
        Node-level encodings :math:`\mathbf{P}_{i,i}` of shape
        ``[num_nodes, walk_length]`` (equivalent to RWSE).
    rel_pe_index : torch.Tensor
        Indices of the non-zero node-pair encodings, shape ``[2, num_pairs]``.
    rel_pe_val : torch.Tensor
        Values of the non-zero node-pair encodings, shape
        ``[num_pairs, walk_length]``.
    deg : torch.Tensor
        Node out-degrees of shape ``[num_nodes]``.
    """
    device = edge_index.device
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)

    # Random walk transition matrix M = D^{-1} A (dense; graphs handled by
    # this transform are small enough for an [n, n, K] tensor).
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    transition = adj / deg.clamp(min=1.0).unsqueeze(-1)

    pe_list = [torch.eye(num_nodes, device=device)]
    walk_matrix = torch.eye(num_nodes, device=device)
    for _ in range(walk_length - 1):
        walk_matrix = walk_matrix @ transition
        pe_list.append(walk_matrix)
    pe = torch.stack(pe_list, dim=-1)  # [n, n, K]

    abs_pe = pe.diagonal().transpose(0, 1)  # [n, K]

    # Sparsify: keep node pairs with at least one non-zero probability.
    # P_{i, j} describes walks from i to j; the attention edge is oriented
    # (source=j, target=i) as in the reference implementation.
    target_idx, source_idx = pe.abs().sum(dim=-1).nonzero(as_tuple=True)
    rel_pe_index = torch.stack([source_idx, target_idx], dim=0)
    rel_pe_val = pe[target_idx, source_idx]

    return abs_pe, rel_pe_index, rel_pe_val, deg


class AddRRWP(BaseTransform):
    r"""Add RRWP positional encodings (GRIT) to a graph.

    The transform attaches to each ``Data`` object:

    - ``rrwp``: node-level encodings of shape ``[num_nodes, walk_length]``
      (the diagonal of the RRWP tensor, equivalent to RWSE),
    - ``rrwp_index`` / ``rrwp_val``: sparse relative encodings for node
      pairs connected by walks of length ``< walk_length``,
    - ``log_deg``: :math:`\log(1 + d_i)` used by GRIT's degree scaler.

    Since ``rrwp_index`` contains the substring ``"index"``, PyTorch
    Geometric automatically offsets it by the number of nodes when
    batching, exactly like ``edge_index``.

    Parameters
    ----------
    walk_length : int, optional
        Number of random walk channels :math:`K`, including the identity
        (default: 8).
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(self, walk_length: int = 8, **kwargs):
        super().__init__()
        if walk_length < 2:
            raise ValueError(
                f"walk_length must be at least 2, got {walk_length}."
            )
        self.walk_length = walk_length

    def __repr__(self):
        return f"{self.__class__.__name__}(walk_length={self.walk_length})"

    def forward(self, data: Data) -> Data:
        r"""Compute and attach RRWP encodings to the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object.

        Returns
        -------
        torch_geometric.data.Data
            Graph data object with ``rrwp``, ``rrwp_index``, ``rrwp_val``
            and ``log_deg`` attributes.
        """
        abs_pe, rel_pe_index, rel_pe_val, deg = compute_rrwp(
            data.edge_index, data.num_nodes, self.walk_length
        )

        data.rrwp = abs_pe
        data.rrwp_index = rel_pe_index
        data.rrwp_val = rel_pe_val
        data.log_deg = torch.log1p(deg)

        return data
