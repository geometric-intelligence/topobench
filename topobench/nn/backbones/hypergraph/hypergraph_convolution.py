"""Define the hypergraph convolution neural network layer."""

import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from topobench.transforms.liftings.graph2hypergraph.hypergraph_laplacian import (
    Laplacian,
)


class SparseMM(torch.autograd.Function):
    """Provide sparse times dense matrix multiplication with autograd support."""

    @staticmethod
    def forward(ctx, M1, M2):
        """Compute the forward pass for sparse matrix multiplication.

        Parameters
        ----------
        ctx : object
            The context object.
        M1 : torch.Tensor
            The sparse matrix.
        M2 : torch.Tensor
            The dense matrix.

        Returns
        -------
        torch.Tensor
            The resulting multiplied matrix.
        """
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        """Compute the backward pass for sparse matrix multiplication.

        Parameters
        ----------
        ctx : object
            The context object.
        g : torch.Tensor
            The gradient tensor.

        Returns
        -------
        tuple
            The gradients for M1 and M2.
        """
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


def incidence_to_hyperedges(incidence, min_size=2):
    """Convert a sparse node-hyperedge incidence matrix to a hyperedge dict.

    ``Laplacian`` expects hyperedges as a mapping from hyperedge id to the list
    of node ids it contains, whereas ``HypergraphWrapper`` hands the backbone
    the sparse ``[num_nodes, num_hyperedges]`` incidence matrix.

    Hyperedges with fewer than ``min_size`` nodes are dropped: for a singleton
    the supremum and the infimum coincide and the normalisation constant
    ``2 * len(e) - 3`` becomes negative, which would inject negative weights
    into the adjacency and break the symmetric normalisation. Singletons do
    occur here, since the k-hop lifting gives every isolated node a hyperedge
    containing only itself.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse incidence matrix of shape ``[num_nodes, num_hyperedges]``.
    min_size : int, optional
        Minimum number of nodes for a hyperedge to be kept, by default 2.

    Returns
    -------
    dict
        Mapping from hyperedge index to the list of its node indices.
    """
    indices = incidence.coalesce().indices().cpu()
    nodes = indices[0].tolist()
    edges = indices[1].tolist()

    hyperedges = {}
    for node, edge in zip(nodes, edges, strict=True):
        hyperedges.setdefault(edge, []).append(node)

    return {
        edge: members
        for edge, members in hyperedges.items()
        if len(members) >= min_size
    }


class HyperGraphConvolution(Module):
    """Define a simple GCN layer.

    Parameters
    ----------
    a : int
        The input feature dimension.
    b : int
        The output feature dimension.
    reapproximate : bool, optional
        Whether to reapproximate the Laplacian, by default True.
    cuda : int or None, optional
        The CUDA device index, by default None.
    **kwargs : dict, optional
        Required for TopoBench to do evaluation.
    """

    def __init__(self, a, b, reapproximate=True, cuda=None, **kwargs):
        super().__init__()
        self.a, self.b = a, b
        self.reapproximate = reapproximate
        self.device = torch.device(
            "cuda:" + str(cuda) if cuda is not None else "cpu"
        )

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer parameters."""
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, H, structure, m=True):
        """Compute the forward pass of the HyperGraph Convolution layer.

        Parameters
        ----------
        H : torch.Tensor
            The hidden node features.
        structure : torch.Tensor
            The sparse node-hyperedge incidence matrix, of shape
            ``[num_nodes, num_hyperedges]``.
        m : bool, optional
            Whether to use mediators, by default True.

        Returns
        -------
        tuple
            A tuple containing the updated node features and hyperedge features.
        """
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        n = H.shape[0]
        num_hyperedges = structure.shape[1]

        if self.reapproximate:
            X = HW.cpu().detach().numpy()
            hyperedges = incidence_to_hyperedges(structure)

            if len(hyperedges) > 0:
                A = Laplacian(n, hyperedges, X, m)
            else:
                # Every hyperedge was a singleton: fall back to the identity,
                # i.e. self-loops only, which is what the normalised Laplacian
                # would reduce to anyway.
                A = torch.sparse_coo_tensor(
                    torch.arange(n).repeat(2, 1),
                    torch.ones(n),
                    (n, n),
                )
        else:
            A = structure

        A = A.to(H.device)

        AHW = SparseMM.apply(A, HW)

        x_1 = torch.zeros((num_hyperedges, self.b), device=H.device)
        return AHW + b, x_1

    def __repr__(self):
        """Return the string representation of the module.

        Returns
        -------
        str
            The module string representation.
        """
        return (
            self.__class__.__name__
            + " ("
            + str(self.a)
            + " -> "
            + str(self.b)
            + ")"
        )
