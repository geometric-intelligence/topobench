"""This module implements a Discrete Neural Sheaf Propagation-based model[1] that can be used with the training framework.

Neural Sheaf Propagation is a method for learning representations of graphs using sheaf structure: node and edge stalks communicating via transport maps / restriction maps.
Adapted and simplified from Suk et al. [2] and Bodnar et al. [1].

[1] Bodnar et al. "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs" https://arxiv.org/abs/2202.04579
[2] Suk et al. "Surfing on the Neural Sheaf" https://openreview.net/pdf?id=xOXFkyRzTlu

"""

from torch.nn import Module
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.graph.nsd_utils.inductive_discrete_propagation_models import (
    InductiveDiscreteBundleSheafPropagation,
    InductiveDiscreteDiagSheafPropagation,
    InductiveDiscreteGeneralSheafPropagation,
)


class NSPEncoder(Module):
    """
    Neural Sheaf Propagation Encoder that can be used with the training framework.

    This encoder learns representations using sheaf structure with node and edge stalks
    communicating via transport maps / restriction maps. Supports three types of sheaf
    structures: diagonal, bundle, and general.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features.
    hidden_dim : int
        Dimension of hidden layers. Must be divisible by d.
    num_layers : int, optional
        Number of sheaf propagation layers. Default is 2.
    sheaf_type : str, optional
        Type of sheaf structure. Options are 'diag', 'bundle', or 'general'.
        Default is 'diag'.
    d : int, optional
        Dimension of the stalk space. For 'diag', d >= 1. For 'bundle' and 'general', d > 1.
        Default is 2.
    dropout : float, optional
        Dropout rate for hidden layers. Default is 0.1.
    input_dropout : float, optional
        Dropout rate for input layer. Default is 0.1.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    sheaf_act : str, optional
        Activation function for sheaf learning. Options are 'tanh', 'elu', 'id'.
        Default is 'tanh'.
    orth : str, optional
        Orthogonalization method for bundle sheaf type. Options are 'cayley' or 'matrix_exp'.
        Default is 'cayley'.
    **kwargs : dict
        Additional keyword arguments (not used).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        sheaf_type="diag",
        d=2,
        dropout=0.1,
        input_dropout=0.1,
        device="cpu",
        sheaf_act="tanh",
        orth="cayley",  # cayley or matrix_exp
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
            self.sheaf_class = InductiveDiscreteDiagSheafPropagation
        elif sheaf_type == "bundle":
            assert d > 1
            self.sheaf_class = InductiveDiscreteBundleSheafPropagation
        elif sheaf_type == "general":
            assert d > 1
            self.sheaf_class = InductiveDiscreteGeneralSheafPropagation
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
        }

        # Create the sheaf model
        self.sheaf_propagation_model = self.sheaf_class(self.sheaf_config)

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
        Forward pass of Neural Sheaf Propagation encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges]. Will be automatically converted to undirected.
        edge_attr : torch.Tensor, optional
            Edge feature matrix (not used). Default is None.
        edge_weight : torch.Tensor, optional
            Edge weights (not used). Default is None.
        batch : torch.Tensor, optional
            Batch vector assigning each node to a specific graph (not used). Default is None.
        **kwargs : dict
            Additional arguments (not used).

        Returns
        -------
        torch.Tensor
            Output node feature matrix of shape [num_nodes, hidden_dim].
        """
        # Neural Sheaf Propagation requires undirected graphs (bidirectional edges)
        # Convert to undirected if not already
        edge_index = to_undirected(edge_index)

        # Run through the sheaf model (no edge attributes)
        return self.sheaf_propagation_model(x, edge_index)

    def get_sheaf_propagation_model(self):
        """
        Get the underlying sheaf model.

        Returns
        -------
        SheafDiffusion
            The sheaf propagation model instance.
        """
        return self.sheaf_propagation_model
