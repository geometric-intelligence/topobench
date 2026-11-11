"""Provide functions to compute the hypergraph Laplacian."""

import numpy as np
import scipy.sparse as sp
import torch


def Laplacian(V, E, X, m):
    """Approximate the hypergraph Laplacian with or without mediators.

    Parameters
    ----------
    V : int
        The number of vertices.
    E : dict
        The dictionary of hyperedges.
    X : numpy.ndarray
        The node feature matrix.
    m : bool
        Whether to use mediators.

    Returns
    -------
    torch.sparse.FloatTensor
        The approximate hypergraph Laplacian matrix.
    """
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E:
        hyperedge = list(E[k])

        p = np.dot(X[hyperedge], rv)  # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2 * len(hyperedge) - 3  # normalisation constant
        if m:
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / c)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / c)

            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend(
                        [
                            [Se, mediator],
                            [Ie, mediator],
                            [mediator, Se],
                            [mediator, Ie],
                        ]
                    )
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se, Ie], [Ie, Se]])
            e = len(hyperedge)

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / e)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / e)

    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """Update the weights on edges connecting extremes to the mediator.

    Parameters
    ----------
    Se : int
        The supremum node index.
    Ie : int
        The infimum node index.
    mediator : int
        The mediator node index.
    weights : dict
        The dictionary tracking edge weights.
    c : float
        The normalization constant.

    Returns
    -------
    dict
        The updated edge weights dictionary.
    """
    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights


def adjacency(edges, weights, n):
    """Compute a sparse adjacency matrix from given edges and weights.

    Parameters
    ----------
    edges : list
        The list of edges.
    weights : dict
        The dictionary of weights for each edge.
    n : int
        The number of nodes in the graph.

    Returns
    -------
    torch.sparse.FloatTensor
        The normalized sparse PyTorch tensor.
    """
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary]
    organised = []

    for e in edges:
        i, j = e[0], e[1]
        w = weights[(i, j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32
    )
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M):
    """Symmetrically normalize a sparse matrix.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetrically normalized sparse matrix.
    """
    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.0
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)


def ssm2tst(M):
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    M : scipy.sparse.coo_matrix
        The input scipy sparse matrix.

    Returns
    -------
    torch.sparse.FloatTensor
        The converted PyTorch sparse tensor.
    """
    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def normalise(M):
    """Row-normalize a sparse matrix.

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        The input sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The row-normalized sparse matrix.
    """
    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.0
    di = np.nan_to_num(di)
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
