"""Wrapper for the loopy (r-neighbourhood) models."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class LoopyWrapper(AbstractWrapper):
    r"""Wrapper for the loopy r-neighbourhood models.

    The :class:`RNeighbourhood` transform stores, for every order ``L``, the
    path tensors ``loopyN{L}`` and ``loopyA{L}`` with **graph-local** node
    indices, plus ``loopyNcount{L}`` (the number of paths per graph). This
    wrapper shifts the local indices to their global positions in the
    batched graph — using the per-graph node offsets derived from
    ``batch_0`` — and hands the assembled paths to the backbone. The
    backbone returns the embeddings of the rank-0 cells.
    """

    def forward(self, batch):
        r"""Forward pass for the loopy wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data. Expected to carry
            ``loopyN{L}``, ``loopyA{L}`` and ``loopyNcount{L}`` for every
            ``L`` in ``0 .. r``.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        loopy_n, loopy_a = self._assemble_paths(batch)

        x_0 = self.backbone(
            batch.x_0,
            batch.edge_index,
            batch=batch.batch_0,
            loopy_n=loopy_n,
            loopy_a=loopy_a,
            edge_weight=batch.get("edge_weight", None),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out

    @staticmethod
    def _assemble_paths(batch):
        r"""Shift path node indices from graph-local to batch-global.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        loopy_n : dict[int, torch.Tensor]
            Per-order node-index paths of shape ``(L + 2, num_paths)`` with
            batch-global indices.
        loopy_a : dict[int, torch.Tensor]
            Per-order hop distances of shape ``(L + 2, num_paths)``, aligned
            with ``loopy_n``.
        """
        batch_0 = batch.batch_0
        num_graphs = int(batch_0.max().item()) + 1 if batch_0.numel() else 0
        node_counts = torch.bincount(batch_0, minlength=num_graphs)
        node_offset = torch.cat(
            [node_counts.new_zeros(1), node_counts.cumsum(0)]
        )

        loopy_n = {}
        loopy_a = {}
        order = 0
        while f"loopyN{order}" in batch:
            paths = batch[f"loopyN{order}"]
            if paths.numel() > 0:
                counts = batch[f"loopyNcount{order}"]
                graph_of_path = torch.repeat_interleave(
                    torch.arange(num_graphs, device=paths.device), counts
                )
                paths = paths + node_offset[graph_of_path].unsqueeze(1)
            loopy_n[order] = paths.t()
            loopy_a[order] = batch[f"loopyA{order}"].t()
            order += 1

        return loopy_n, loopy_a
