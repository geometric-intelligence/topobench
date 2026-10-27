"""Shared fixtures for TRAWL tests."""

import torch
from torch_geometric.data import Data, InMemoryDataset

from topobench.dataloader import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.nn.wrappers.combinatorial.trawl_wrapper import TRAWLWrapper
from topobench.transforms.data_manipulations.trawl import TRAWLTransform

WRAPPER_OPTIONS = {
    "walks",
    "seed",
    "eval_views",
    "walk_refresh",
    "transition_cache_size",
}


def complex_data(empty=False):
    """Small lifted complex with three nodes, two edges and one 2-cell."""
    incidence = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Data(
        x_0=torch.randn(3, 2),
        x_1=torch.randn(2, 3),
        x_2=torch.empty(0, 1) if empty else torch.randn(1, 1),
        incidence_1=incidence.to_sparse(),
        incidence_2=(
            torch.empty(2, 0) if empty else torch.ones(2, 1)
        ).to_sparse(),
        num_nodes=3,
        y=torch.tensor([1]),
        train_mask=torch.tensor([True]),
        val_mask=torch.tensor([True]),
        test_mask=torch.tensor([True]),
    )


def prepare(data=None, **kwargs):
    """Apply ``TRAWLTransform`` with cheap encodings."""
    return TRAWLTransform(
        encodings={"local": True, "RWSE": {"max_pe_dim": 2}}, **kwargs
    )(data or complex_data())


def collate(graphs):
    """Batch transformed graphs with TopoBench's collator."""
    dataset = DataloadDataset(graphs)
    return collate_fn([dataset[i] for i in range(len(dataset))])


def network(graphs, **kwargs):
    """Small ``TRAWLWrapper(TRAWL)`` initialized on ``graphs``.

    Keyword arguments go to the wrapper when they configure walk sampling
    and to the backbone otherwise.
    """
    wrapper_options = {"walks": {"k": 3, "length": 5}}
    backbone_options = {"hidden_dim": 16, "depth": 2, "architecture": "mlp"}
    for key, value in kwargs.items():
        target = wrapper_options if key in WRAPPER_OPTIONS else backbone_options
        target[key] = value
    net = TRAWLWrapper(TRAWL(**backbone_options), **wrapper_options)
    net.initialize(graphs)
    return net


class TinyDataset(InMemoryDataset):
    """Sixteen labeled triangles."""

    def __init__(self):
        super().__init__()
        rows = []
        for i in range(16):
            edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
            rows.append(
                Data(
                    x=torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0]]),
                    edge_index=edges,
                    num_nodes=3,
                    y=torch.tensor([i % 2]),
                )
            )
        self.data, self.slices = self.collate(rows)


class TinyLoader:
    """Dataset loader returning ``TinyDataset`` for runner tests."""

    def __init__(self, parameters):
        self.directory = parameters.data_dir

    def load(self):
        """Return the dataset and its directory."""
        return TinyDataset(), self.directory
