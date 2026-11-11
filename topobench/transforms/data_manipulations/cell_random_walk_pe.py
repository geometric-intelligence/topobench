"""Precompute random walk positional encodings for cell complexes."""

import torch
import torch_geometric

from topobench.nn.backbones.cell.cellular_transformer import random_walk_pe


class CellRandomWalkPE(torch_geometric.transforms.BaseTransform):
    r"""Precompute random walk positional encodings on a cell complex.

    Computes the per-rank random walk encodings (RWPe, Appendix C.1 of
    `Ballester et al., 2024 <https://arxiv.org/abs/2405.14094>`_) used
    by the Cellular Transformer once per complex at preprocessing time,
    attaching them as ``rwpe_0``, ``rwpe_1`` and ``rwpe_2``. This moves
    the ``pe_steps`` sequential sparse matrix products off the training
    hot path: the backbone reads the cached tensors instead of
    recomputing them on every forward pass. Since the encodings are
    per-cell dense features, standard batching concatenates them in
    exact alignment with ``x_0``/``x_1``/``x_2``, and the diagonal of a
    block-diagonal random walk operator equals the per-block diagonals,
    so precomputed and on-the-fly encodings are numerically identical.

    The transform is a no-op on data without the cell-complex
    neighborhood matrices, so it composes safely in generic pipelines.

    Parameters
    ----------
    pe_steps : int, optional
        Number of random walk steps (must match the backbone's
        ``pe_steps``; default: 8).
    **kwargs : optional
        Additional (ignored) arguments for the class.
    """

    def __init__(self, pe_steps: int = 8, **kwargs) -> None:
        super().__init__()
        self.type = "cell_random_walk_pe"
        self.pe_steps = pe_steps

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"pe_steps={self.pe_steps})"
        )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Attach per-rank random walk encodings to the data object.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data, expected to carry the within-rank
            neighborhood matrices of a 2-dimensional cell complex
            (``adjacency_0``, ``coadjacency_1``, ``coadjacency_2``).

        Returns
        -------
        torch_geometric.data.Data
            The data with ``rwpe_0``, ``rwpe_1``, ``rwpe_2`` attached
            (unchanged if the neighborhood matrices are absent).
        """
        neighborhoods = {
            "rwpe_0": "adjacency_0",
            "rwpe_1": "coadjacency_1",
            "rwpe_2": "coadjacency_2",
        }
        for pe_key, matrix_key in neighborhoods.items():
            matrix = data.get(matrix_key, None)
            if matrix is not None:
                with torch.no_grad():
                    data[pe_key] = random_walk_pe(matrix, self.pe_steps)
        return data
