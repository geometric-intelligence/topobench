"""This module implements the HypergraphKernelLifting class."""

import copy
import typing

import torch
import torch_geometric
import torch_geometric.utils
from scipy.linalg import fractional_matrix_power as fmp

from topobench.transforms.liftings.graph2hypergraph.base import (
    Graph2HypergraphLifting,
)


def graph_heat_kernel(laplacian: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    """Return graph heat kernel $$K = exp(-t L)$$.

    Parameters
    ----------
    laplacian : torch.Tensor
        The graph Laplacian (alternatively can be the normalized graph Laplacian).

    t : float
        The temperature parameter for the heat kernel.

    Returns
    -------
    torch.Tensor
        The heat kernel.
    """
    return torch.linalg.matrix_exp(-t * laplacian)


def graph_matern_kernel(
    laplacian: torch.Tensor, nu: int = 1, kappa: int = 1
) -> torch.Tensor:
    """
    Return graph Matérn kernel.

    Parameters
    ----------
    laplacian : torch.Tensor
        The graph Laplacian (alternatively can be the normalized graph Laplacian).
    nu : float
        Smoothness parameter of the kernel.
    kappa : int
        Lengthscale parameter of the kernel.

    Returns
    -------
    torch.Tensor
        The Matérn kernel matrix K = (2*nu / kappa^2 * I + L)^(-nu).

    Notes
    -----
    I represents the identity matrix and L is the graph Laplacian.
    """

    id_matrix = torch.eye(laplacian.shape[0])
    return torch.tensor(fmp((2 * nu / kappa**2) * id_matrix + laplacian, -nu))


def get_graph_kernel(
    laplacian, kernel: str | typing.Callable = "heat", **kwargs
):
    """
    Return a graph kernel.

    Parameters
    ----------
    laplacian : torch.Tensor
        The graph Laplacian (alternatively can be the normalized graph Laplacian).
    kernel : str or callable
        Either the name of a kernel or a callable kernel function.
    **kwargs : dict
        Additional keyword arguments representing the hyperparameters of the kernel.
        These should be passed to the kernel function.

    Returns
    -------
    torch.Tensor
        A graph kernel for the provided Laplacian matrix.
    """

    if callable(kernel):
        return kernel(laplacian, **kwargs)
    if kernel == "heat":
        t = kwargs["t"]
        return graph_heat_kernel(laplacian, t=t)
    if kernel == "matern":
        nu, kappa = kwargs["nu"], kwargs["kappa"]
        return graph_matern_kernel(laplacian, nu=nu, kappa=kappa)
    if kernel == "identity":
        return torch.ones_like(laplacian)
    raise ValueError(f"Unknown graph kernel type {kernel}")


def get_feat_kernel(
    features, kernel: str | typing.Callable = "identity", **kwargs
):
    """
    Compute a kernel matrix for the given features based on the specified kernel type.

    Parameters
    ----------
    features : torch.Tensor
        A 2D tensor representing the features for which the kernel matrix is to be computed.
        Each row corresponds to a feature vector.

    kernel : str or callable, optional
        Specifies the type of kernel to apply or a custom kernel function.
        - If a string, it specifies a predefined kernel type. Currently, only "identity" is supported.
        The "identity" kernel returns an identity matrix of size `(N, N)`, where `N` is the number of features.
        - If a callable, it should be a function that takes `features` and additional keyword arguments (`**kwargs`)
        as input and returns a kernel matrix.
        Default is "identity".

    **kwargs : dict, optional
        Additional keyword arguments required by the custom kernel function if `kernel` is a callable.

    Returns
    -------
    torch.Tensor
        The computed kernel matrix. If `kernel="identity"`, the result is an identity matrix of size `(N, N)`.
        If `kernel` is a callable, the result is determined by the custom kernel function.

    Raises
    ------
    ValueError
        If `kernel` is a string but not one of the supported kernel types (currently only "identity").

    Examples
    --------
    Example with the "identity" kernel:

    >>> import torch
    >>> features = torch.randn(5, 3)  # 5 features with 3 dimensions each
    >>> kernel_matrix = get_feat_kernel(features, "identity")
    >>> print(kernel_matrix)

    Example with a custom kernel function:

    >>> def custom_kernel_fn(features, **kwargs):
    ...     # Example: return a random kernel matrix of appropriate size
    ...     return torch.rand(features.shape[0], features.shape[0])
    >>> kernel_matrix = get_feat_kernel(features, custom_kernel_fn)
    >>> print(kernel_matrix)
    """

    if callable(kernel):
        return kernel(features)
    if kernel == "identity":
        return torch.ones((features.shape[0], features.shape[0]))
    raise ValueError(f"Unknown feature kernel type: {kernel}")


def get_combination(c_name_or_func: typing.Callable | str) -> typing.Callable:
    """Return a combination function based on the specified type or function.

    Parameters
    ----------
    c_name_or_func : str or callable
        The combination method to use. This can be:
        - A string specifying a predefined combination type:
          - "prod": Returns a function that computes the element-wise product of two inputs.
          - "sum": Returns a function that computes the element-wise sum of two inputs.
        - A callable: A custom combination function that takes two arguments (A and B) and combines them.

    Returns
    -------
    callable
        A function that combines two inputs based on the specified combination type or custom function.
        The returned function takes two parameters, A and B, which can be scalars, tensors, or other compatible types,
        and returns their combined result.

    Raises
    ------
    ValueError
        If `c_name_or_func` is a string that does not match any supported predefined combination type.

    Examples
    --------
    Example with the "prod" combination:
        >>> prod_fn = get_combination("prod")
        >>> result = prod_fn(2, 3)
        >>> print(result)
        6

    Example with the "sum" combination:
        >>> sum_fn = get_combination("sum")
        >>> result = sum_fn(2, 3)
        >>> print(result)
        5

    Example with a custom combination function:
        >>> def custom_combination(A, B):
        ...     return A - B
        >>> custom_fn = get_combination(custom_combination)
        >>> result = custom_fn(7, 4)
        >>> print(result)
        3
    """
    if callable(c_name_or_func):
        return c_name_or_func
    if c_name_or_func == "prod":
        return lambda A, B: A * B
    if c_name_or_func == "sum":
        return lambda A, B: A + B
    raise ValueError(f"Unknown name for kernel combination {c_name_or_func}")


class HypergraphKernelLifting(Graph2HypergraphLifting):
    r"""Lift graphs to hypergraph domain by the kernel over graphs (features can be included).

    Parameters
    ----------
    graph_kernel : str or callable
        The kernel function to be applied to the graph topology, if a string, it specifies a predefined kernel type. Currently, only "heat" is supported. If a callable, it should be a function that takes the graph Laplacian and additional kwargs as input and returns a kernel matrix.
    feat_kernel : str or callable
        The kernel function to be applied to the features, if a string, it specifies a predefined kernel type. Currently, only "identity" is supported. If a callable, it should be a function that takes the features and additional kwargs as input and returns a kernel matrix.
    C : str or callable
        Default is "heat".
    fraction : float
        The fraction of the kernel to be considered for the hypergraph construction. Default is 0.5.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        graph_kernel="heat",
        feat_kernel="identity",
        C="prod",
        fraction=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph_kernel = graph_kernel
        self.feat_kernel = feat_kernel
        self.fraction = fraction
        self.graph_kernel = graph_kernel
        self.feat_kernel = feat_kernel
        self.C = C
        self.kwargs = kwargs

    def _remove_empty_edges(self, incidence: torch.Tensor) -> torch.Tensor:
        r"""Remove hyperedges with fewer than two incident vertices from the incidence matrix.

        Parameters
        ----------
        incidence : torch.Tensor
            A 2D tensor (vertices x hyperedges) representing the incidence matrix of a hypergraph.

        Returns
        -------
        torch.Tensor
            A filtered incidence matrix with only hyperedges having more than one incident vertex.
        """
        keep_columns = torch.where(torch.sum(incidence != 0, dim=0) > 1)[0]
        return incidence[:, keep_columns]

    def _deduplicate_hyperedges(self, incidence: torch.Tensor) -> torch.Tensor:
        r"""Remove duplicate hyperedges from the incidence matrix.

        Parameters
        ----------
        incidence : torch.Tensor
            A 2D tensor (vertices x hyperedges) representing the incidence matrix of a hypergraph.

        Returns
        -------
        torch.Tensor
            A deduplicated incidence matrix with unique hyperedges.
        """
        transposed_tensor = incidence.T
        return torch.unique(transposed_tensor, dim=0).T

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> dict[str, torch.Tensor]:
        r"""Lift the topology of a graph to hypergraph domain by considering the kernel over vertices or alternatively features.

        Extended Summary
        ----------------
        In a most generic form the kernel looks like:
        $$K =  C(K_v(v, v^{\prime}) K_x(x, x^{\prime})),$$
        where $K_v$ is a kernel over the graph (graph_kernel), $K_x$ is a kernel over the features (feat_kernel), and
        C is the function to combine those (for instance sum or prod).

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        typing.Dict[str, torch.Tensor]
            The lifted topology.

        Raises
        ------
        ValueError: if the input is incomplete or in incorrect format.
        """
        if not torch_geometric.utils.is_undirected(data.edge_index):
            raise ValueError(
                "HypergraphKernelLifting is applicable only to undirected graphs"
            )

        num_nodes = data.x.shape[0]
        data.pos = data.x
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_hyperedges)

        # constructing the Laplacian
        edge_index, edge_weight = torch_geometric.utils.get_laplacian(
            data.edge_index, data.edge_weight, normalization="sym"
        )
        laplacian = torch.zeros(
            (num_nodes, num_nodes), dtype=edge_weight.dtype
        )
        laplacian[edge_index[0], edge_index[1]] = edge_weight

        # obtaining the kernels (K_v is the kernel from graph topology and K_x is the kernel from features)
        K_v = get_graph_kernel(
            laplacian=laplacian, kernel=self.graph_kernel, **self.kwargs
        )
        K_x = get_feat_kernel(
            features=data.x, kernel=self.feat_kernel, **self.kwargs
        )

        # combine the kernels
        C = get_combination(c_name_or_func=self.C)
        K = C(K_v, K_x)

        # build new graph
        threshold = torch.quantile(K, 1 - self.fraction)
        indices = torch.nonzero(threshold <= K, as_tuple=True)
        data_lifted = copy.deepcopy(data)
        data_lifted.edge_index = indices

        # save and postprocess hyperedge incidence
        incidence_1[data_lifted.edge_index[1], data_lifted.edge_index[0]] = 1
        incidence_1 = self._remove_empty_edges(incidence_1)
        incidence_1 = self._deduplicate_hyperedges(incidence_1)
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": incidence_1.shape[1],
            "x_0": data.x,
        }
