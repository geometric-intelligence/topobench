"""Adapt lifting outputs to relation-labelled walk graphs without losing ranks.

All connectivity stored by this transform uses graph-local indices. PyG must
concatenate, not increment, these fields; their names deliberately avoid
``index`` and ``x_`` (TopoBench's collator assigns meaning to those strings).
"""

import hashlib
import re

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import BaseTransform

from topobench.data.utils.trawl.encodings import positional_encodings


def as_csr(value):
    """Convert dense or any PyTorch sparse layout to unsigned CPU CSR.

    Parameters
    ----------
    value : torch.Tensor or array_like or scipy.sparse.spmatrix
        Dense or sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Float64 CSR matrix of absolute values.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_sparse_coo().coalesce()
            row, col = value.indices().numpy()
            return sp.csr_matrix(
                (np.abs(value.values().numpy()), (row, col)), value.shape
            )
        value = value.numpy()
    return abs(sp.csr_matrix(value, dtype=np.float64))


def relation_matrix(name, counts, incidences, data):
    """Resolve TopoTune hop/direction/relation/rank syntax.

    Adjacency means shared upper cells; coadjacency means shared lower cells.
    Multi-hop incidence is the product along consecutive ranks. Explicit
    lifting-provided relation matrices take precedence over derived ones.

    Parameters
    ----------
    name : str
        Neighborhood name such as ``"up_incidence-0"`` or
        ``"2-down_adjacency-2"``.
    counts : list of int
        Number of cells per rank.
    incidences : dict
        CSR incidence matrix per rank ``r >= 1``, shaped
        ``(counts[r - 1], counts[r])``.
    data : torch_geometric.data.Data
        Lifted data, checked for an explicit matrix stored under ``name``.

    Returns
    -------
    tuple
        ``(rank, target, matrix)``: source rank, target rank and the CSR
        relation matrix of shape ``(counts[rank], counts[target])``.
    """
    match = re.fullmatch(
        r"(?:(\d+)-)?(up|down)_(adjacency|incidence)-(\d+)", name
    )
    if match is None:
        raise ValueError(f"Invalid TRAWL neighborhood: {name!r}")
    hops, direction, kind, rank = match.groups()
    hops, rank = int(hops or 1), int(rank)
    if hops < 1 or rank >= len(counts):
        raise ValueError(f"Neighborhood outside configured ranks: {name}")
    other = rank + (hops if direction == "up" else -hops)
    target = other if kind == "incidence" else rank
    shape = (counts[rank], counts[target] if 0 <= target < len(counts) else 0)
    if not 0 <= other < len(counts):
        return rank, target, sp.csr_matrix(shape)
    if name in data:
        matrix = as_csr(data[name])
        # TopoBench stores incidence as target x source, unlike adjacency.
        if kind == "incidence":
            matrix = matrix.T.tocsr()
    else:
        matrix = sp.eye(counts[min(rank, other)], format="csr")
        for level in range(min(rank, other) + 1, max(rank, other) + 1):
            matrix = matrix @ incidences[level]
        if direction == "down":
            matrix = matrix.T.tocsr()
        if kind == "adjacency":
            matrix = (matrix @ matrix.T).tocsr()
            matrix.setdiag(0)
            matrix.eliminate_zeros()
    if matrix.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {matrix.shape}")
    return rank, target, matrix


class TRAWLTransform(BaseTransform):
    """Prepare any rank/incidence lifting for TRAWL.

    Plain graphs become rank 0 states; ``incidence_hyperedges`` is adapted to
    rank 1. Missing higher ranks are empty, not fabricated. Featureless cells
    receive a constant feature, while existing lifted features are preserved.

    Parameters
    ----------
    max_rank : int, optional
        Highest cell rank to include (default: 2).
    graph : str, optional
        Walk graph: ``"hasse"``, ``"augmented_hasse"`` or ``"cell_overlap"``
        (default: "augmented_hasse").
    neighborhoods : list of str, optional
        Relation names resolved by ``relation_matrix``; defaults to
        ``up_incidence-r`` for every rank below ``max_rank`` (default: None).
    bidirectional : bool, optional
        If True, symmetrize every relation (default: True).
    overlap_ranks : list of int, optional
        Ranks included in the ``"cell_overlap"`` graph; defaults to
        ``1..max_rank`` (default: None).
    encodings : dict, optional
        Keyword arguments for ``positional_encodings`` (default: None,
        meaning ``{"local": True, "rw_steps": 8}``).
    encoding_scope : str, optional
        Compute encodings per relation (``"separate"``) or on their union
        (``"union"``) (default: "separate").
    color_key : str, optional
        Data attribute holding one nonnegative color ID per cell; defaults
        to colouring cells by rank (default: None).
    color_refinement : bool, optional
        If True, split encoding graphs by unordered color pair
        (default: False).
    num_colors : int, optional
        Number of color IDs; defaults to ``max_rank + 1`` (default: None).
    seed : int, optional
        Seed passed to ``positional_encodings`` (default: 0).
    **kwargs : dict
        Ignored extra options.
    """

    def __init__(
        self,
        max_rank=2,
        graph="augmented_hasse",
        neighborhoods=None,
        bidirectional=True,
        overlap_ranks=None,
        encodings=None,
        encoding_scope="separate",
        color_key=None,
        color_refinement=False,
        num_colors=None,
        seed=0,
        **kwargs,
    ):
        if graph not in {"hasse", "augmented_hasse", "cell_overlap"}:
            raise ValueError(f"Unknown walk graph {graph}")
        if encoding_scope not in {"separate", "union"}:
            raise ValueError("encoding_scope must be separate or union")
        self.max_rank = int(max_rank)
        if self.max_rank < 0:
            raise ValueError("max_rank must be nonnegative")
        self.graph = graph
        self.neighborhoods = list(
            neighborhoods
            or [f"up_incidence-{r}" for r in range(self.max_rank)]
            or ["up_adjacency-0"]
        )
        self.bidirectional = bidirectional
        self.overlap_ranks = list(overlap_ranks or range(1, self.max_rank + 1))
        self.encodings = dict(encodings or {"local": True, "rw_steps": 8})
        self.encoding_scope = encoding_scope
        self.color_key = color_key
        self.color_refinement = color_refinement
        self.num_colors = int(num_colors or (self.max_rank + 1))
        self.seed = int(seed)

    def finalize_dataset(self, data_list):
        """Give absent ranks the same empty feature shape as populated ranks.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed graphs, updated in place.

        Returns
        -------
        list of torch_geometric.data.Data
            The same graphs with consistent per-rank feature widths.
        """
        for rank in range(self.max_rank + 1):
            key = f"trawl_signal_{rank}"
            widths = {
                data[key].shape[1] for data in data_list if len(data[key])
            }
            if len(widths) > 1:
                raise ValueError(
                    f"Rank-{rank} feature widths vary across graphs: {widths}"
                )
            if widths:
                width = next(iter(widths))
                for data in data_list:
                    if not len(data[key]):
                        data[key] = data[key].new_empty(0, width)
                        if f"x_{rank}" in data and not len(data[f"x_{rank}"]):
                            data[f"x_{rank}"] = data[key].clone()
        return data_list

    def forward(self, data):
        """Build the TRAWL walk graph, features, colors and encodings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Lifted (or plain) graph.

        Returns
        -------
        torch_geometric.data.Data
            The input data with ``trawl_*`` fields added.
        """
        counts = [int(data.num_nodes)] + [0] * self.max_rank
        if counts[0] <= 0:
            raise ValueError("TRAWL requires at least one rank-0 cell")
        incidences = {}
        for rank in range(1, self.max_rank + 1):
            key = f"incidence_{rank}"
            if (
                rank == 1
                and key not in data
                and "incidence_hyperedges" in data
            ):
                key = "incidence_hyperedges"
            if key in data:
                matrix = as_csr(data[key])
                if matrix.shape[0] != counts[rank - 1]:
                    raise ValueError(f"{key} has inconsistent lower-rank size")
                counts[rank] = matrix.shape[1]
            else:
                features = data.get(f"x_{rank}")
                counts[rank] = 0 if features is None else len(features)
                matrix = sp.csr_matrix((counts[rank - 1], counts[rank]))
            incidences[rank] = matrix
        offsets = np.cumsum([0] + counts)
        total = int(offsets[-1])
        memberships = {0: sp.eye(counts[0], format="csr")}
        for rank in range(1, self.max_rank + 1):
            memberships[rank] = memberships[rank - 1] @ incidences[rank]
            memberships[rank].data[:] = 1
        for rank, count in enumerate(counts):
            features = data.get(f"x_{rank}")
            if features is None and rank == 0:
                features = data.get("x")
            if features is None and rank == 1:
                features = data.get("x_hyperedges")
            if features is None:
                features = torch.ones(count, 1)
            if features.ndim == 1:
                features = features[:, None]
            if len(features) != count:
                raise ValueError(f"Rank {rank} feature/count mismatch")
            data[f"trawl_signal_{rank}"] = features.float()
        colors = np.repeat(np.arange(len(counts)), counts)
        if self.color_key is not None:
            colors = np.asarray(data[self.color_key].cpu()).reshape(-1)
            if len(colors) != total or np.any(colors < 0):
                raise ValueError(
                    "Colors must contain one nonnegative ID per cell"
                )
        matrices = []
        if self.graph == "cell_overlap":
            selected = [
                memberships[r]
                if r in self.overlap_ranks
                else sp.csr_matrix((counts[0], counts[r]))
                for r in range(len(counts))
            ]
            membership = sp.hstack(selected, format="csr")
            matrix = (membership.T @ membership).tocsr()
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            matrices.append(matrix)
        else:
            for name in self.neighborhoods:
                if self.graph == "hasse" and (
                    "adjacency" in name or name[0].isdigit()
                ):
                    raise ValueError(
                        "Strict Hasse graphs only accept immediate incidence neighborhoods"
                    )
                src, dst, matrix = relation_matrix(
                    name, counts, incidences, data
                )
                # Native graphs have no rank-1 incidence: use their edges.
                if (
                    name == "up_adjacency-0"
                    and counts[1:2] in ([], [0])
                    and "edge_index" in data
                ):
                    edges = data.edge_index.cpu().numpy()
                    matrix = sp.csr_matrix(
                        (np.ones(edges.shape[1]), edges),
                        shape=(counts[0], counts[0]),
                    )
                coo = matrix.tocoo()
                if coo.nnz:
                    expanded = sp.csr_matrix(
                        (
                            coo.data,
                            (coo.row + offsets[src], coo.col + offsets[dst]),
                        ),
                        shape=(total, total),
                    )
                else:
                    expanded = sp.csr_matrix((total, total))
                if self.bidirectional:
                    expanded = expanded.maximum(expanded.T)
                matrices.append(expanded)
        entries, weights = [], []
        for relation, matrix in enumerate(matrices):
            coo = matrix.tocoo()
            entries.extend(
                zip(coo.row, coo.col, np.full(coo.nnz, relation), strict=True)
            )
            weights.extend(coo.data)
        data.trawl_edges = torch.tensor(
            np.asarray(entries).reshape(-1, 3), dtype=torch.long
        )
        data.trawl_weights = torch.tensor(weights, dtype=torch.float32)
        data.trawl_counts = torch.tensor([counts], dtype=torch.long)
        data.trawl_colors = torch.tensor(colors, dtype=torch.long)
        data.trawl_relations = torch.tensor([len(matrices)])
        union = sum(matrices, sp.csr_matrix((total, total)))
        enc_matrices = (
            matrices if self.encoding_scope == "separate" else [union]
        )
        if self.color_refinement:
            if np.any(colors >= self.num_colors):
                raise ValueError("num_colors must cover every color ID")
            refined = []
            # Fixed slots for each unordered color pair, including same-color
            # relations. Refine PSE graphs without changing walk connectivity.
            for matrix in enc_matrices:
                coo = matrix.tocoo()
                for a in range(self.num_colors):
                    for b in range(a, self.num_colors):
                        keep = (
                            (colors[coo.row] == a) & (colors[coo.col] == b)
                        ) | ((colors[coo.row] == b) & (colors[coo.col] == a))
                        refined.append(
                            sp.csr_matrix(
                                (
                                    coo.data[keep],
                                    (coo.row[keep], coo.col[keep]),
                                ),
                                shape=matrix.shape,
                            )
                        )
            enc_matrices = refined
        data.trawl_pe = torch.from_numpy(
            np.concatenate(
                [
                    positional_encodings(
                        matrix, seed=self.seed, **self.encodings
                    )
                    for matrix in enc_matrices
                ],
                axis=1,
            )
        )
        digest = hashlib.sha256()
        for tensor in [
            data.trawl_counts,
            data.trawl_edges,
            data.trawl_weights,
        ]:
            digest.update(tensor.numpy().tobytes())
        data.trawl_identity = torch.tensor(
            [int.from_bytes(digest.digest()[:7], "little")]
        )
        return data
