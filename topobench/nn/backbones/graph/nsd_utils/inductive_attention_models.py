"""
Inductive Sheaf Attention Network models.

Three variants matching the restriction-map families of NSD:

- Diag: per-edge diagonal restriction maps (d scalars each).
- Bundle: per-edge orthogonal restriction maps in O(d), the setting
  used by the SheafAN paper.
- General: full d x d restriction maps.

Each variant follows the SheafAN forward update (Barbero et al., 2022,
equation 5)::

    X_{t+1} = sigma( (Lambda_hat * A_F) (I_n kron W1) X_t W2 )

or the Res-SheafAN variant (equation 6) when ``residual`` is enabled::

    X_{t+1} = X_t + sigma( ((Lambda_hat * A_F) - I) (I_n kron W1) X_t W2 )

The attention-weighted sheaf adjacency ``Lambda_hat * A_F`` is built by
the adjacency builders; ``Lambda`` is the GAT-style attention computed
on the augmented edge index (original edges plus self-loops).
"""

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn

from .adjacency_builders import (
    DiagSheafAdjacencyBuilder,
    GeneralSheafAdjacencyBuilder,
    NormConnectionSheafAdjacencyBuilder,
)
from .san_attention import SheafGATAttention
from .sheaf_base import SheafDiffusion
from .sheaf_models import LocalConcatSheafLearner


def _augment_with_self_loops(edge_index, num_nodes):
    """
    Append self-loop edges to a directed edge index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge indices of shape [2, num_edges].
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Augmented edge indices of shape [2, num_edges + num_nodes].
    """
    loop = torch.arange(num_nodes, device=edge_index.device)
    loop = loop.unsqueeze(0).expand(2, -1)
    return torch.cat([edge_index, loop], dim=1)


class _InductiveSheafAttentionBase(SheafDiffusion):
    """
    Common machinery for the three SheafAN variants.

    Stores the per-layer linear weights, sheaf learners, and attention
    modules. Subclasses provide the restriction-map dimensionality
    (``get_param_size``, ``_map_out_shape``) and the adjacency builder.

    Parameters
    ----------
    config : dict
        See ``SheafDiffusion``. Two extra keys are honoured:
        ``residual`` (bool, default False) selects the Res-SheafAN
        update; ``num_heads`` (int, default 1) controls multi-head
        attention.
    """

    def __init__(self, config):
        super().__init__(None, config)
        self.config = config
        self.residual = config.get("residual", False)
        self.num_heads = config.get("num_heads", 1)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        for _ in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _ in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for _ in range(self.layers):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=self._map_out_shape(),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.sheaf_attentions = nn.ModuleList()
        for _ in range(self.layers):
            self.sheaf_attentions.append(
                SheafGATAttention(
                    in_channels=self.hidden_dim,
                    num_heads=self.num_heads,
                )
            )

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def _map_out_shape(self):
        """
        Return the per-edge restriction map shape.

        Returns
        -------
        tuple of int
            The shape used by ``LocalConcatSheafLearner.out_shape``.
        """
        raise NotImplementedError

    def _build_adjacency(self, num_nodes, edge_index):
        """
        Construct a fresh adjacency builder for the current graph.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edge_index : torch.Tensor
            Directed edge indices of shape [2, num_edges].

        Returns
        -------
        nn.Module
            Adjacency builder configured for ``edge_index``.
        """
        raise NotImplementedError

    def left_right_linear(self, x, left, right, actual_num_nodes):
        """
        Apply (I_n kron W1) on the stalk axis and W2 on the channels.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [num_nodes * d, hidden_channels].
        left : nn.Linear
            Stalk-side transform W1.
        right : nn.Linear
            Channel-side transform W2.
        actual_num_nodes : int
            Number of nodes in the current graph.

        Returns
        -------
        torch.Tensor
            Transformed features, same shape as the input.
        """
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, actual_num_nodes * self.d).t()
        x = right(x)
        return x

    def forward(self, x, edge_index):
        """
        Run the SheafAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Directed edge indices of shape [2, num_edges]. Must be
            bidirectional and contain no self-loops; self-loops are
            added internally for the attention softmax.

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        actual_num_nodes = x.size(0)
        adjacency_builder = self._build_adjacency(actual_num_nodes, edge_index)
        edge_index_aug = _augment_with_self_loops(edge_index, actual_num_nodes)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(actual_num_nodes * self.d, -1)

        for layer in range(self.layers):
            x_maps = F.dropout(
                x,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            x_maps = x_maps.reshape(actual_num_nodes, -1)
            maps = self.sheaf_learners[layer](x_maps, edge_index)
            alpha = self.sheaf_attentions[layer](x_maps, edge_index_aug)
            (a_idx, a_val), trans_maps = adjacency_builder(maps, alpha)
            self.sheaf_learners[layer].set_L(trans_maps)

            x_prev = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_t = self.left_right_linear(
                x,
                self.lin_left_weights[layer],
                self.lin_right_weights[layer],
                actual_num_nodes,
            )
            mx = torch_sparse.spmm(a_idx, a_val, x_t.size(0), x_t.size(0), x_t)

            x = x_prev + F.elu(mx - x_t) if self.residual else F.elu(mx)

        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveSheafAttentionDiag(_InductiveSheafAttentionBase):
    """
    SheafAN variant with diagonal restriction maps.

    Each restriction map is a diagonal d x d matrix, so the transport
    blocks P_ij = F_i diag-times F_j remain diagonal.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see ``SheafDiffusion``). Requires
        ``d > 0``. Honours ``residual`` and ``num_heads``.

    Raises
    ------
    AssertionError
        If ``d`` is not positive.
    """

    def __init__(self, config):
        assert config["d"] > 0
        super().__init__(config)

    def _map_out_shape(self):
        """Return the diagonal restriction map shape ``(d,)``.

        Returns
        -------
        tuple of int
            One-element tuple containing ``d``.
        """
        return (self.d,)

    def _build_adjacency(self, num_nodes, edge_index):
        """Instantiate a diagonal sheaf adjacency builder.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edge_index : torch.Tensor
            Directed edge indices of shape [2, num_edges].

        Returns
        -------
        DiagSheafAdjacencyBuilder
            The configured adjacency builder.
        """
        return DiagSheafAdjacencyBuilder(num_nodes, edge_index, d=self.d)


class InductiveSheafAttentionBundle(_InductiveSheafAttentionBase):
    """
    SheafAN variant with orthogonal restriction maps (paper setting).

    Restriction maps live in O(d) via a Cayley or matrix-exponential
    parameterization. This is the configuration used in the empirical
    evaluation of Barbero et al. (2022).

    Parameters
    ----------
    config : dict
        Configuration dictionary (see ``SheafDiffusion``). Requires
        ``d > 1`` and ``hidden_dim`` divisible by ``d``.

    Raises
    ------
    AssertionError
        If ``d <= 1`` or ``hidden_dim`` is not divisible by ``d``.
    """

    def __init__(self, config):
        assert config["d"] > 1
        super().__init__(config)
        assert self.hidden_dim % self.d == 0

    def _map_out_shape(self):
        """Return the orthogonal-map parameter shape.

        Returns
        -------
        tuple of int
            One-element tuple containing ``d * (d + 1) // 2``, the
            number of lower-triangular parameters needed to recover an
            orthogonal d x d matrix.
        """
        return (self.d * (self.d + 1) // 2,)

    def _build_adjacency(self, num_nodes, edge_index):
        """Instantiate an orthogonal sheaf adjacency builder.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edge_index : torch.Tensor
            Directed edge indices of shape [2, num_edges].

        Returns
        -------
        NormConnectionSheafAdjacencyBuilder
            The configured adjacency builder.
        """
        return NormConnectionSheafAdjacencyBuilder(
            num_nodes, edge_index, d=self.d, orth_map=self.orth_trans
        )


class InductiveSheafAttentionGeneral(_InductiveSheafAttentionBase):
    """
    SheafAN variant with full (unrestricted) d x d restriction maps.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see ``SheafDiffusion``). Requires
        ``d > 1``.

    Raises
    ------
    AssertionError
        If ``d <= 1``.
    """

    def __init__(self, config):
        assert config["d"] > 1
        super().__init__(config)

    def _map_out_shape(self):
        """Return the full-matrix restriction map shape ``(d, d)``.

        Returns
        -------
        tuple of int
            Two-element tuple ``(d, d)`` for arbitrary linear maps.
        """
        return (self.d, self.d)

    def _build_adjacency(self, num_nodes, edge_index):
        """Instantiate a general sheaf adjacency builder.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        edge_index : torch.Tensor
            Directed edge indices of shape [2, num_edges].

        Returns
        -------
        GeneralSheafAdjacencyBuilder
            The configured adjacency builder.
        """
        return GeneralSheafAdjacencyBuilder(num_nodes, edge_index, d=self.d)
