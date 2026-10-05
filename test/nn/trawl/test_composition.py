"""Regression checks for interactions between independently configured pieces."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from topobench.data.utils.trawl.sampling import WalkSampler
from topobench.model.trawl_pretraining import TRAWLPretrainer
from topobench.nn.encoders.trawl import TRAWLFeatureEncoder
from topobench.nn.readouts.trawl import TRAWLReadout
from topobench.transforms.data_manipulations.trawl import TRAWLTransform

from .test_trawl import collate, complex_data, model, prepare


def test_native_component_registration():
    """Native discovery and direct Hydra imports share the backbone classes."""
    from topobench.loss import TBLoss
    from topobench.loss.loss import TBLoss as DirectTBLoss
    from topobench.nn.backbones import MODEL_CLASSES
    from topobench.nn.backbones.combinatorial import BACKBONE_CLASSES
    from topobench.nn.backbones.combinatorial.trawl import TRAWL
    from topobench.nn.encoders import FEATURE_ENCODERS
    from topobench.nn.readouts import READOUT_CLASSES
    from topobench.transforms import TRANSFORMS
    from topobench.transforms.data_manipulations import DATA_MANIPULATIONS

    assert MODEL_CLASSES["TRAWL"] is TRAWL
    assert TBLoss is DirectTBLoss
    assert TBLoss.__module__ == "topobench.loss.loss"
    assert BACKBONE_CLASSES["TRAWL"] is TRAWL
    assert "NeighborhoodFusion" not in MODEL_CLASSES
    assert "PureTorchMambaBlock" not in MODEL_CLASSES
    assert FEATURE_ENCODERS["TRAWLFeatureEncoder"] is TRAWLFeatureEncoder
    assert "TRAWLReadout" in READOUT_CLASSES
    assert TRANSFORMS["TRAWLTransform"] is DATA_MANIPULATIONS["TRAWLTransform"]


def test_separate_readout_does_not_silently_discard_fusion():
    data = prepare()
    output = model([data], walk_scope="separate", fusion="attention")(
        collate([data])
    )
    with pytest.raises(ValueError, match="fusion"):
        TRAWLReadout(16, 2, graph_dim=32, aggregation="walk_logits")(
            output, None
        )


def test_attention_occurrences_and_absent_rank_widths():
    first = prepare(complex_data(empty=True))
    second = complex_data()
    second.x_2 = torch.randn(1, 4)
    second = prepare(second)
    TRAWLTransform().finalize_dataset([first, second])
    assert first.trawl_signal_2.shape == (0, 4)
    net = model(
        [first, second], occurrence_pooling="attention", graph_readout="cells"
    )
    result = net(collate([first, second]))
    result["graph_embedding"].square().sum().backward()
    assert net.occurrence_attention.weight.grad is not None
    assert torch.isfinite(result["graph_embedding"]).all()


def test_transition_cache_respects_masked_edges():
    full = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    masked = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    sampler = WalkSampler(cache_size=1)
    sampler(full, k=8, length=8, seed=4)
    paths = sampler(masked, k=8, length=8, seed=4)
    assert set(paths.flatten()) == {0, 1}
    assert len(sampler.cache) == 1


def test_pretraining_updates_native_feature_encoder():
    class RankEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleList(
                [nn.Linear(width, 5) for width in [2, 3, 1]]
            )

        def forward(self, batch):
            for rank, layer in enumerate(self.projections):
                batch[f"x_{rank}"] = layer(batch[f"x_{rank}"])
            return batch

    data = prepare()
    net = model([data], in_channels=[5, 5, 5])
    encoder = TRAWLFeatureEncoder(RankEncoder(), [0, 1, 2])
    trainer = TRAWLPretrainer(
        net, feature_encoder=encoder, target_widths=[2, 3, 1]
    )
    trainer.log = lambda *args, **kwargs: None
    trainer._step(collate([data])).backward()
    assert encoder.encoder.projections[0].weight.grad is not None
