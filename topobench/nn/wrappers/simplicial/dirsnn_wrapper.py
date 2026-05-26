"""Wrapper for the Dir-SNN model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper

# Stable ordering of the 10 directed edge adjacencies fed to the Dir-SNN
# backbone. Mirrors the order produced by
# :class:`DirectedSimplicialLifting` (which itself follows
# ``compute_lower_adj`` / ``compute_upper_adj`` of the upstream
# reference repository).
DIRSNN_ADJ_KEYS: tuple[str, ...] = (
    "dir_lower_adj_100",
    "dir_lower_adj_101",
    "dir_lower_adj_110",
    "dir_lower_adj_111",
    "dir_upper_adj_101",
    "dir_upper_adj_102",
    "dir_upper_adj_112",
    "dir_upper_adj_110",
    "dir_upper_adj_120",
    "dir_upper_adj_121",
)

# The first four entries of ``DIRSNN_ADJ_KEYS`` are the lower (Eq. 3)
# adjacencies and the remaining six are the upper (Eq. 4) adjacencies;
# the official experimental setup of the upstream Dir-SNN repository
# uses only the lower four. See "official experimental mode" below.
_DIRSNN_LOWER_KEYS: tuple[str, ...] = DIRSNN_ADJ_KEYS[:4]
_DIRSNN_UPPER_KEYS: tuple[str, ...] = DIRSNN_ADJ_KEYS[4:]
_VALID_ADJ_SUBSETS = (None, "lower", "upper")


class DirSNNWrapper(AbstractWrapper):
    r"""Wrapper for the Dir-SNN backbone.

    Implements the data-marshalling expected by :class:`DirSNN`
    (Lecha et al. 2024, arXiv:2409.08389): edge features ``x_1`` plus a
    tuple of edge-level *directed* adjacency matrices produced by the
    paired :class:`DirectedSimplicialLifting` transform.

    Two intended operating modes are supported via ``adj_subset``:

    * **Paper-compatible / challenge mode** (``adj_subset=None``,
      default). The wrapper forwards all ten adjacency tensors named
      in :data:`DIRSNN_ADJ_KEYS` to the backbone, in the order
      ``(A^{0,0}_{down,1}, A^{0,1}_{down,1}, A^{1,0}_{down,1},
      A^{1,1}_{down,1},
      A^{0,1}_{up,1}, A^{0,2}_{up,1}, A^{1,2}_{up,1},
      A^{1,0}_{up,1}, A^{2,0}_{up,1}, A^{2,1}_{up,1})``
      (paper Sec. III, Eqs. (3)-(4)). This is the general formulation
      of the paper.

    * **Official experimental mode** (``adj_subset="lower"``). The
      wrapper forwards only the four *lower* directed adjacencies
      (``dir_lower_adj_100``, ``dir_lower_adj_101``,
      ``dir_lower_adj_110``, ``dir_lower_adj_111``). This matches the
      narrower experimental setup actually used in the upstream
      reference repository (``compute_lower_adj`` of
      ``repos/DirSNN/compute_adj.py`` returns exactly these four; the
      upper adjacencies live in ``compute_upper_adj`` but are not used
      in the paper's published experiments).

    A symmetric ``adj_subset="upper"`` knob is also provided for the
    six upper directed adjacencies (used for ablations).

    Parameters
    ----------
    backbone : torch.nn.Module
        The Dir-SNN backbone. Its ``n_adjs`` must equal the number of
        adjacencies implied by ``adj_subset``.
    adj_subset : {None, "lower", "upper"}, optional
        Which directed adjacencies to forward to the backbone. ``None``
        (default) forwards all 10. ``"lower"`` forwards the 4 lower
        adjacencies (Eq. (3)) only. ``"upper"`` forwards the 6 upper
        adjacencies (Eq. (4)) only.
    **kwargs : dict
        Forwarded to :class:`AbstractWrapper`.

    Notes
    -----
    The Hodge-Laplacian fallback used in earlier revisions of this
    wrapper has been removed: it did not exercise the paper's central
    novelty (higher-order directionality). Use
    :class:`DirectedSimplicialLifting` to produce the directed
    adjacencies on the batch -- the default
    ``configs/model/simplicial/dirsnn.yaml`` does so automatically.
    """

    def __init__(self, backbone, adj_subset: str | None = None, **kwargs):
        super().__init__(backbone, **kwargs)
        if adj_subset not in _VALID_ADJ_SUBSETS:
            raise ValueError(
                f"Unknown adj_subset={adj_subset!r}; expected one of "
                f"{_VALID_ADJ_SUBSETS}."
            )
        self.adj_subset = adj_subset
        if adj_subset is None:
            self._adj_keys = DIRSNN_ADJ_KEYS
        elif adj_subset == "lower":
            self._adj_keys = _DIRSNN_LOWER_KEYS
        else:  # "upper"
            self._adj_keys = _DIRSNN_UPPER_KEYS

        expected_n_adjs = len(self._adj_keys)
        actual_n_adjs = getattr(backbone, "n_adjs", None)
        if actual_n_adjs != expected_n_adjs:
            raise ValueError(
                f"DirSNNWrapper expected backbone.n_adjs={expected_n_adjs} "
                f"for adj_subset={adj_subset!r}, got {actual_n_adjs!r}."
            )

    def forward(self, batch):
        r"""Forward pass for the Dir-SNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing batched simplicial data with the ten
            directed adjacency attributes named in
            :data:`DIRSNN_ADJ_KEYS`.

        Returns
        -------
        dict
            Dictionary with updated edge embeddings under ``x_1`` and
            node embeddings (obtained via the boundary map) under
            ``x_0``.
        """
        adjs = tuple(getattr(batch, key) for key in self._adj_keys)
        x_1 = self.backbone(batch.x_1, adjs)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        # Project edge embeddings back to nodes via the boundary
        # operator so that downstream readouts have access to a
        # 0-cell signal as well.
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        model_out["x_1"] = x_1
        return model_out
