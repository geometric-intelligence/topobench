"""This module implements the Sheaf Attention Network (SheafAN) model[1] for use with the training framework.

Sheaf Attention Networks generalise Graph Attention Networks to
cellular sheaves: an additional learned d x d transport matrix
(restriction map) is attached to each edge alongside the usual GAT
attention coefficient. The transport matrices live in O(d) by default,
yielding a norm-preserving message-passing operator that mitigates both
oversmoothing and heterophily.

[1] Barbero et al. "Sheaf Attention Networks" (Extended Abstract Track,
NeurIPS 2022 Workshop on Symmetry and Geometry in Neural
Representations).
https://openreview.net/forum?id=LIDvgVjpkZr
"""

from torch.nn import Module
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.nsd_utils.inductive_attention_models import (
    InductiveSheafAttentionBundle,
    InductiveSheafAttentionDiag,
    InductiveSheafAttentionGeneral,
)


class SANEncoder(Module):
    """
    Sheaf Attention Network encoder that plugs into the TBModel pipeline.

    The encoder learns per-edge restriction maps together with GAT-style
    attention coefficients and aggregates the resulting attention-weighted
    sheaf adjacency through the layers. Three restriction-map families are
    supported in parity with NSD: diagonal, orthogonal bundle, and general
    full matrices. Each variant can be run either with the plain SheafAN
    update (Barbero et al., 2022, equation 5) or with the residual
    Res-SheafAN update (equation 6) via the ``residual`` flag.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers. Must be divisible by ``d``.
    num_layers : int, optional
        Number of SheafAN layers. Default is 2.
    sheaf_type : str, optional
        Type of restriction maps. One of ``'diag'``, ``'bundle'``,
        ``'general'``. Default is ``'bundle'`` (paper setting).
    d : int, optional
        Dimension of the stalk space. For ``'diag'``, ``d >= 1``; for
        ``'bundle'`` and ``'general'``, ``d > 1``. Default is 2.
    dropout : float, optional
        Dropout rate applied between layers. Default is 0.1.
    input_dropout : float, optional
        Dropout rate applied to input features. Default is 0.1.
    device : str, optional
        Device on which to instantiate the model. Default is ``'cpu'``.
    sheaf_act : str, optional
        Activation used inside the sheaf learner. One of ``'tanh'``,
        ``'elu'``, ``'id'``. Default is ``'tanh'``.
    orth : str, optional
        Orthogonalization method for the bundle variant.
        ``'cayley'`` or ``'matrix_exp'``. Default is ``'cayley'``.
    num_heads : int, optional
        Number of attention heads. Per-head scores are averaged before
        scaling the sheaf adjacency. Default is 1.
    residual : bool, optional
        If True, use the Res-SheafAN update (equation 6); otherwise the
        plain SheafAN update (equation 5). Default is False.
    **kwargs : dict
        Additional keyword arguments (ignored, kept for Hydra
        flexibility).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        sheaf_type="bundle",
        d=2,
        dropout=0.1,
        input_dropout=0.1,
        device="cpu",
        sheaf_act="tanh",
        orth="cayley",
        num_heads=1,
        residual=False,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sheaf_type = sheaf_type
        self.d = d
        self.num_layers = num_layers
        self.device = device

        if sheaf_type == "diag":
            assert d >= 1
            self.sheaf_class = InductiveSheafAttentionDiag
        elif sheaf_type == "bundle":
            assert d > 1
            self.sheaf_class = InductiveSheafAttentionBundle
        elif sheaf_type == "general":
            assert d > 1
            self.sheaf_class = InductiveSheafAttentionGeneral
        else:
            raise ValueError(f"Unknown sheaf type: {sheaf_type}")

        self.sheaf_config = {
            "d": d,
            "layers": num_layers,
            "hidden_channels": hidden_dim // d,
            "input_dim": input_dim,
            "output_dim": hidden_dim,
            "device": device,
            "input_dropout": input_dropout,
            "dropout": dropout,
            "sheaf_act": sheaf_act,
            "orth": orth,
            "num_heads": num_heads,
            "residual": residual,
        }

        self.san_model = self.sheaf_class(self.sheaf_config)

    def forward(
        self,
        x,
        edge_index,
        edge_attr=None,
        edge_weight=None,
        batch=None,
        **kwargs,
    ):
        """
        Forward pass of the Sheaf Attention Network encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges]. Automatically
            symmetrized to undirected.
        edge_attr : torch.Tensor, optional
            Edge feature matrix (unused). Default is None.
        edge_weight : torch.Tensor, optional
            Edge weights (unused). Default is None.
        batch : torch.Tensor, optional
            Batch vector assigning each node to a graph (unused).
            Default is None.
        **kwargs : dict
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        edge_index = to_undirected(edge_index)
        return self.san_model(x, edge_index)

    def get_sheaf_model(self):
        """
        Return the underlying SheafAN model.

        Returns
        -------
        torch.nn.Module
            The wrapped inductive SheafAN model instance.
        """
        return self.san_model
