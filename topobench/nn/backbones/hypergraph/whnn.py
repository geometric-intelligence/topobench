"""Wasserstein Hypergraph Neural Network backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlicedWassersteinPooling(nn.Module):
    r"""Pool sets with fixed-size sliced-Wasserstein signatures.

    The layer projects member features onto several directions, sorts each
    one-dimensional projection inside every group, samples a fixed number of
    quantiles, and maps the resulting signature back to feature space.

    Parameters
    ----------
    in_channels : int
        Number of input features per item.
    out_channels : int
        Number of output features per group.
    num_projections : int, optional
        Number of random 1D projections. Defaults to 8.
    num_reference_points : int, optional
        Number of quantile reference points. Defaults to 4.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_projections=8,
        num_reference_points=4,
    ):
        super().__init__()
        if num_projections <= 0:
            raise ValueError("num_projections must be positive")
        if num_reference_points <= 0:
            raise ValueError("num_reference_points must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_projections = num_projections
        self.num_reference_points = num_reference_points

        projections = torch.randn(num_projections, in_channels)
        projections = F.normalize(projections, p=2, dim=-1)
        self.register_buffer("projections", projections)
        self.register_buffer(
            "reference_positions",
            torch.linspace(0.0, 1.0, num_reference_points),
        )
        self.signature_to_features = nn.Linear(
            num_projections * num_reference_points,
            out_channels,
        )

    def reset_parameters(self):
        """Reset trainable parameters."""
        self.signature_to_features.reset_parameters()

    def forward(self, x, group_index, num_groups):
        r"""Pool item features into group features.

        Parameters
        ----------
        x : torch.Tensor
            Item features of shape ``[num_items, in_channels]``.
        group_index : torch.Tensor
            Group assignment for each item.
        num_groups : int
            Number of output groups.

        Returns
        -------
        torch.Tensor
            Group features of shape ``[num_groups, out_channels]``.
        """
        if x.numel() == 0 or group_index.numel() == 0 or num_groups == 0:
            signature = x.new_zeros(
                num_groups,
                self.num_projections * self.num_reference_points,
            )
            return self.signature_to_features(signature)

        projected = x @ self.projections.to(device=x.device, dtype=x.dtype).T
        counts = torch.bincount(group_index, minlength=num_groups)
        max_count = int(counts.max().item())

        if max_count == 0:
            signature = x.new_zeros(
                num_groups,
                self.num_projections * self.num_reference_points,
            )
            return self.signature_to_features(signature)

        perm = torch.argsort(group_index, stable=True)
        sorted_group_index = group_index[perm]
        sorted_projected = projected[perm]

        starts = torch.zeros(num_groups + 1, dtype=torch.long, device=x.device)
        torch.cumsum(counts, dim=0, out=starts[1:])
        item_starts = starts[sorted_group_index]
        intra_index = torch.arange(x.shape[0], device=x.device) - item_starts

        padded = torch.full(
            (num_groups, max_count, self.num_projections),
            fill_value=float("inf"),
            device=x.device,
            dtype=x.dtype,
        )
        padded[sorted_group_index, intra_index] = sorted_projected

        sorted_padded, _ = padded.sort(dim=1)

        r_grid = self.reference_positions.to(device=x.device, dtype=x.dtype)
        n_minus_1 = (counts - 1).clamp(min=0).to(dtype=x.dtype)
        positions = n_minus_1.unsqueeze(1) * r_grid.unsqueeze(0)
        lower = positions.floor().long().clamp(max=max_count - 1)
        upper = positions.ceil().long().clamp(max=max_count - 1)
        weight = (positions - lower.to(dtype=x.dtype)).unsqueeze(-1)

        P = self.num_projections
        R = self.num_reference_points
        lower_expanded = lower.unsqueeze(-1).expand(num_groups, R, P)
        upper_expanded = upper.unsqueeze(-1).expand(num_groups, R, P)

        val_lower = torch.gather(sorted_padded, dim=1, index=lower_expanded)
        val_upper = torch.gather(sorted_padded, dim=1, index=upper_expanded)
        sampled = val_lower * (1.0 - weight) + val_upper * weight

        signatures = sampled.transpose(1, 2)
        mask = (counts > 0).unsqueeze(-1).unsqueeze(-1)
        signatures = torch.where(
            mask, signatures, torch.zeros_like(signatures)
        )

        signatures = signatures.flatten(start_dim=1)
        return self.signature_to_features(signatures)


class WHNNLayer(nn.Module):
    r"""One WHNN node-hyperedge-node message passing layer.

    Parameters
    ----------
    channels : int
        Number of feature channels for nodes and hyperedges.
    num_projections : int, optional
        Number of 1D Wasserstein projections. Defaults to 8.
    num_reference_points : int, optional
        Number of quantile samples per projection. Defaults to 4.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    activation : str, optional
        Activation function. Defaults to ``relu``.
    residual : bool, optional
        Whether to use residual connection. Defaults to True.
    """

    def __init__(
        self,
        channels,
        num_projections=8,
        num_reference_points=4,
        dropout=0.0,
        activation="relu",
        residual=True,
    ):
        super().__init__()
        self.node_to_hyperedge = SlicedWassersteinPooling(
            channels,
            channels,
            num_projections=num_projections,
            num_reference_points=num_reference_points,
        )
        self.hyperedge_to_node = SlicedWassersteinPooling(
            channels,
            channels,
            num_projections=num_projections,
            num_reference_points=num_reference_points,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.norm = nn.LayerNorm(channels)
        self.activation = _activation(activation)

    def reset_parameters(self):
        """Reset trainable parameters."""
        self.node_to_hyperedge.reset_parameters()
        self.hyperedge_to_node.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, node_index, hyperedge_index, num_hyperedges):
        r"""Run one WHNN layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        node_index : torch.Tensor
            Indices of nodes belonging to hyperedges.
        hyperedge_index : torch.Tensor
            Indices of hyperedges corresponding to node_index.
        num_hyperedges : int
            Total number of hyperedges.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated node features and hyperedge features.
        """
        num_nodes = x.shape[0]
        hyperedge_features = self.node_to_hyperedge(
            x[node_index],
            hyperedge_index,
            num_hyperedges,
        )
        node_messages = self.hyperedge_to_node(
            hyperedge_features[hyperedge_index],
            node_index,
            num_nodes,
        )
        node_messages = self.dropout(self.activation(node_messages))
        if self.residual:
            node_messages = node_messages + x
        return self.norm(node_messages), hyperedge_features


class WHNN(nn.Module):
    r"""Wasserstein Hypergraph Neural Network.

    Parameters
    ----------
    num_features : int
        Number of input and output node features.
    num_layers : int, optional
        Number of WHNN layers. Defaults to 2.
    num_projections : int, optional
        Number of one-dimensional Wasserstein slices. Defaults to 8.
    num_reference_points : int, optional
        Number of quantile samples per slice. Defaults to 4.
    input_dropout : float, optional
        Dropout applied to input node features. Defaults to 0.0.
    dropout : float, optional
        Dropout applied in WHNN layers. Defaults to 0.0.
    activation : str, optional
        Activation function. Defaults to ``relu``.
    residual : bool, optional
        Whether to use residual node updates. Defaults to True.
    **kwargs : dict, optional
        Additional keyword arguments for Hydra configuration compatibility.
    """

    def __init__(
        self,
        num_features,
        num_layers=2,
        num_projections=8,
        num_reference_points=4,
        input_dropout=0.0,
        dropout=0.0,
        activation="relu",
        residual=True,
        **kwargs,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.num_features = num_features
        self.input_dropout = nn.Dropout(input_dropout)
        self.layers = nn.ModuleList(
            [
                WHNNLayer(
                    num_features,
                    num_projections=num_projections,
                    num_reference_points=num_reference_points,
                    dropout=dropout,
                    activation=activation,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        )

    def reset_parameters(self):
        """Reset trainable parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, incidence):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        incidence : torch.Tensor
            Node-hyperedge incidence, as sparse/dense matrix or 2-row indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated node features and hyperedge features.
        """
        node_index, hyperedge_index, num_hyperedges = _incidence_to_indices(
            incidence
        )
        node_index = node_index.to(device=x.device)
        hyperedge_index = hyperedge_index.to(device=x.device)

        x = self.input_dropout(x)
        hyperedge_features = x.new_zeros(num_hyperedges, self.num_features)
        for layer in self.layers:
            x, hyperedge_features = layer(
                x,
                node_index,
                hyperedge_index,
                num_hyperedges,
            )
        return x, hyperedge_features


def _incidence_to_indices(incidence):
    r"""Convert supported incidence formats to membership indices.

    Parameters
    ----------
    incidence : torch.Tensor
        Node-hyperedge incidence matrix or index.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Node indices, hyperedge indices, and total number of hyperedges.
    """
    if incidence.layout == torch.sparse_coo:
        incidence = incidence.coalesce()
        indices = incidence.indices()
        return indices[0], indices[1], incidence.shape[1]

    if incidence.dim() != 2:
        raise ValueError("incidence must be a matrix or a 2-row index tensor")

    if incidence.shape[0] == 2 and incidence.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ):
        hyperedge_index = incidence[1]
        num_hyperedges = (
            int(hyperedge_index.max().item()) + 1
            if hyperedge_index.numel() > 0
            else 0
        )
        return incidence[0], hyperedge_index, num_hyperedges

    indices = incidence.nonzero(as_tuple=False).T
    return indices[0], indices[1], incidence.shape[1]


def _activation(name):
    r"""Return activation module by name.

    Parameters
    ----------
    name : str
        Activation function name.

    Returns
    -------
    nn.Module
        Activation module instance.
    """
    activations = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]
