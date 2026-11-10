"""Wrapper that samples walks for the TRAWL backbone and batches its output."""

import numpy as np
import scipy.sparse as sp
import torch

from topobench.nn.backbones.combinatorial.trawl_utils.sampling import (
    WalkSampler,
    upload,
    walk_seed,
)
from topobench.nn.wrappers.base import AbstractWrapper


class TRAWLWrapper(AbstractWrapper):
    r"""Sample walks over ``TRAWLTransform`` outputs and run TRAWL per complex.

    Reads the ``trawl_*`` fields of a batch, samples seeded walks for every
    complex on the host, calls the backbone and assembles the model output
    (``x_r``/``batch_r``, cell, walk and graph embeddings, and labels).

    Parameters
    ----------
    backbone : torch.nn.Module
        A ``TRAWL`` backbone.
    walks : dict, optional
        Options for ``sample_neighbors`` and ``guidance`` for
        ``guided_transition`` (default: ``{"k": 32, "length": 32}``).
    seed : int, optional
        Base seed for walk sampling (default: 42).
    eval_views : int, optional
        Walk views averaged at evaluation (default: 1).
    walk_refresh : str, optional
        ``"train"`` resamples walks every training step, ``"fixed"`` never
        does (default: "train").
    transition_cache_size : int, optional
        Transition distributions cached by the sampler (default: 1024).
    **kwargs : dict
        Arguments of ``AbstractWrapper`` (``out_channels``,
        ``num_cell_dimensions``). TRAWL already normalizes its outputs, so
        the residual connections of the base wrapper are not used.
    """

    # Walk metadata read on the host (see ``TBModel.on_before_batch_transfer``).
    host_fields = (
        "trawl_counts",
        "trawl_edges",
        "trawl_weights",
        "trawl_identity",
        "trawl_relations",
        "trawl_colors",
    )

    def __init__(
        self,
        backbone,
        walks=None,
        seed=42,
        eval_views=1,
        walk_refresh="train",
        transition_cache_size=1024,
        **kwargs,
    ):
        kwargs.update(num_cell_dimensions=0, residual_connections=False)
        kwargs.setdefault("out_channels", backbone.hidden_dim)
        super().__init__(backbone, **kwargs)
        if walk_refresh not in {"train", "fixed"}:
            raise ValueError("walk_refresh must be one of ['fixed', 'train']")
        if eval_views < 1:
            raise ValueError("eval_views must be positive")
        self.walks = dict(walks or {"k": 32, "length": 32})
        self.seed, self.eval_views = seed, eval_views
        self.walk_refresh = walk_refresh
        self.walk_sampler = WalkSampler(transition_cache_size)
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.long)
        )
        # Host mirror of ``sampling_step``; reading the buffer would wait for
        # the device on every forward pass.
        self._host_sampling_step = None

    def __call__(self, batch):
        r"""Run the forward pass without the base residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch produced by ``TRAWLTransform``.

        Returns
        -------
        dict
            Model output of ``forward``.
        """
        return self.forward(batch)

    def initialize(self, data_list):
        """Infer the backbone's input widths from transformed training data.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed training graphs.
        """
        if not data_list:
            raise ValueError(
                "Cannot infer TRAWL input widths from an empty dataset"
            )
        self._check_ranks(data_list[0].trawl_counts.shape[-1])
        signal_widths = []
        for rank in range(self.backbone.max_rank + 1):
            key = f"trawl_signal_{rank}"
            widths = {
                data[key].shape[1] for data in data_list if len(data[key])
            }
            if len(widths) > 1:
                raise ValueError(
                    f"Inconsistent rank-{rank} feature widths: {widths}"
                )
            signal_widths.append(
                next(iter(widths), data_list[0][key].shape[1])
            )
        self.backbone.initialize(signal_widths, data_list[0].trawl_pe.shape[1])

    def _check_ranks(self, num_ranks):
        """Check that the transform built the ranks the backbone expects.

        Parameters
        ----------
        num_ranks : int
            Number of ranks built by ``TRAWLTransform``.
        """
        if num_ranks != self.backbone.max_rank + 1:
            raise ValueError(
                f"Transform built ranks 0..{num_ranks - 1}, but the backbone "
                f"expects max_rank={self.backbone.max_rank}"
            )

    def compilable_layers(self):
        """Return the backbone layers that ``torch.compile`` may compile.

        Returns
        -------
        list of torch.nn.Module
            Sequence layers of the backbone.
        """
        return self.backbone.compilable_layers()

    def _host(self, batch):
        """Return CPU copies of ``host_fields`` without waiting for the device.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch, optionally carrying precomputed ``host_copies`` arrays.

        Returns
        -------
        dict
            Mapping from field name to NumPy array.
        """
        host = getattr(batch, "host_copies", None)
        if host is None:
            host = {
                key: batch[key].detach().cpu().numpy()
                for key in self.host_fields
                if key in batch
            }
        return host

    def next_sampling_step(self):
        """Return the training sampling step and advance it.

        Returns
        -------
        int
            Sampling step before the increment.
        """
        if self._host_sampling_step is None:
            self._host_sampling_step = int(self.sampling_step)
        step = self._host_sampling_step
        self._host_sampling_step += 1
        self.sampling_step.add_(1)
        return step

    def reset_sampling_step(self):
        """Restart walk refreshes, e.g. before supervised fitting."""
        self.sampling_step.zero_()
        self._host_sampling_step = 0

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state and invalidate the host sampling-step mirror.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``nn.Module``.
        **kwargs : dict
            Keyword arguments forwarded to ``nn.Module``.
        """
        self._host_sampling_step = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, batch):
        """Sample and encode walks for every complex in the batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by ``TRAWLTransform``.

        Returns
        -------
        dict
            Contextual cell features ``x_r`` and ``batch_r``, cell, walk and
            graph embeddings with their batch indices, labels and readout
            metadata.
        """
        backbone = self.backbone
        max_rank = backbone.max_rank
        host = self._host(batch)
        num_graphs = len(host["trawl_relations"])
        counts = host["trawl_counts"].reshape(num_graphs, -1)
        self._check_ranks(counts.shape[1])
        edge_slices = getattr(batch, "_slice_dict", {}).get("trawl_edges")
        node_offsets = np.zeros(max_rank + 1, dtype=int)
        state_offset = 0
        (
            graph_embeddings,
            nodes,
            node_batches,
            walk_embeddings,
            walk_batches,
        ) = [], [], [], [], []
        all_cells, all_ranks, walk_views = [], [], []
        rank_cells = [[] for _ in range(max_rank + 1)]
        rank_batches = [[] for _ in range(max_rank + 1)]
        views = 1 if self.training else self.eval_views
        step = self.next_sampling_step() if self.training else 0
        if self.walk_refresh != "train":
            step = 0
        device = batch.trawl_pe.device

        def index(size, value):
            """Constant index vector.

            Parameters
            ----------
            size : int
                Length.
            value : int
                Fill value.

            Returns
            -------
            torch.Tensor
                Long tensor of ``size`` copies of ``value``.
            """
            return torch.full((size,), value, dtype=torch.long, device=device)

        for graph_id, sizes in enumerate(counts):
            if (
                int(host["trawl_relations"][graph_id])
                != backbone.num_neighborhoods
            ):
                raise ValueError("Transform/model neighborhood counts differ")
            signals = []
            for rank, size in enumerate(sizes):
                signals.append(
                    batch[f"trawl_signal_{rank}"][
                        node_offsets[rank] : node_offsets[rank] + size
                    ]
                )
                node_offsets[rank] += size
            total = int(sizes.sum())
            ranks = upload(np.repeat(np.arange(len(sizes)), sizes), device)
            colors = None
            if backbone.color is not None:
                host_colors = host["trawl_colors"][
                    state_offset : state_offset + total
                ]
                if (
                    host_colors.size
                    and int(host_colors.max()) >= backbone.color.num_embeddings
                ):
                    raise ValueError(
                        "num_colors is smaller than a transformed color ID"
                    )
                colors = batch.trawl_colors[
                    state_offset : state_offset + total
                ]
            if edge_slices is None:
                edges, weights = host["trawl_edges"], host["trawl_weights"]
            else:
                start, stop = (
                    int(edge_slices[graph_id]),
                    int(edge_slices[graph_id + 1]),
                )
                edges, weights = (
                    host["trawl_edges"][start:stop],
                    host["trawl_weights"][start:stop],
                )
            groups = (
                [None]
                if backbone.walk_scope == "union"
                else list(range(backbone.num_neighborhoods))
            )
            walk_groups = []
            for group_id, relation in enumerate(groups):
                select = (
                    np.ones(len(edges), dtype=bool)
                    if relation is None
                    else edges[:, 2] == relation
                )
                chosen = edges[select]
                matrix = sp.csr_matrix(
                    (weights[select], (chosen[:, 0], chosen[:, 1])),
                    shape=(total, total),
                )
                sampled = []
                for view in range(views):
                    seed = walk_seed(
                        self.seed,
                        int(host["trawl_identity"][graph_id]),
                        step,
                        group_id,
                        view,
                    )
                    paths = self.walk_sampler(matrix, seed=seed, **self.walks)
                    if len(paths):
                        sampled.append((view, upload(paths, device)))
                walk_groups.append(sampled)
            out = backbone(
                signals,
                batch.trawl_pe[state_offset : state_offset + total],
                ranks,
                walk_groups,
                colors=colors,
            )
            contextual = out["contextual"]
            for view, pooled in out["walks"]:
                walk_embeddings.append(pooled)
                walk_views.append(index(len(pooled), view))
                walk_batches.append(index(len(pooled), graph_id))
            graph_embeddings.append(out["graph_embedding"])
            rank_offset = 0
            for rank, size in enumerate(sizes):
                rank_cells[rank].append(
                    contextual[rank_offset : rank_offset + size]
                )
                rank_batches[rank].append(index(size, graph_id))
                rank_offset += size
            all_cells.append(contextual)
            all_ranks.append(ranks)
            nodes.append(contextual[: sizes[0]])
            node_batches.append(index(sizes[0], graph_id))
            state_offset += total
        result = {
            "x_0": torch.cat(nodes),
            "batch_0": torch.cat(node_batches),
            "cell_embeddings": torch.cat(all_cells),
            "cell_ranks": torch.cat(all_ranks),
            "graph_embedding": torch.stack(graph_embeddings),
            "labels": batch.get("y"),
            "walk_readout_compatible": backbone.walk_scope == "union"
            and backbone.graph_readout == "walks",
            "num_views": views,
        }
        result["walk_embedding"] = (
            torch.cat(walk_embeddings)
            if walk_embeddings
            else result["graph_embedding"].new_empty((0, backbone.output_dim))
        )
        result["walk_batch"] = (
            torch.cat(walk_batches)
            if walk_batches
            else result["batch_0"].new_empty(0)
        )
        result["walk_view"] = (
            torch.cat(walk_views)
            if walk_views
            else result["batch_0"].new_empty(0)
        )
        for rank in range(1, max_rank + 1):
            result[f"x_{rank}"] = torch.cat(rank_cells[rank])
            result[f"batch_{rank}"] = torch.cat(rank_batches[rank])
        return result
