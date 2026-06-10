"""Pipeline smoke checks for the combinatorial ETNN config.

The fast test composes Hydra config only. The actual one-epoch training smoke
test is kept in this file but skipped by default because it may download and
preprocess MUTAG. That gives us a ready manual/CI check without making normal
unit-test runs depend on network/data availability.
"""

from __future__ import annotations

import hydra
import pytest

from test._utils.simplified_pipeline import run


class TestETNNPipeline:
    """End-to-end checks for ``model=combinatorial/etnn``."""

    def setup_method(self):
        """Reset Hydra between tests so overrides stay isolated."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_etnn_config_composes_with_graph_to_combinatorial_lifting(self):
        """ETNN config should select the expected TopoBench lifting path."""
        # This is the cheapest integration check: it verifies Hydra can resolve
        # the graph dataset, combinatorial ETNN model, and required lifting
        # without instantiating datasets or trainers.
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn",
                    "dataset=graph/MUTAG",
                ],
                return_hydra_config=False,
            )

        # The model config should resolve to the new combinatorial ETNN
        # backbone.
        assert cfg.model.model_name == "etnn"
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.combinatorial.etnn.ETNN"
        )

        # Graph datasets need a graph-to-combinatorial lifting before ETNN can
        # consume rank-wise cell features and neighborhoods.
        assert "graph2combinatorial_lifting" in cfg.transforms
        lifting = cfg.transforms.graph2combinatorial_lifting
        assert lifting.transform_name == "GraphTriangleInducedCC"

        # Hydra resolves the lifting interpolation to the ETNN neighborhood
        # list. This ensures preprocessing creates exactly the sparse relations
        # the backbone will consume.
        assert list(lifting.neighborhoods) == list(
            cfg.model.backbone.neighborhoods
        )

    @pytest.mark.skip(
        reason=(
            "One-epoch ETNN pipeline run may download/process MUTAG. "
            "Run manually when network/data setup is available."
        )
    )
    def test_etnn_one_epoch_pipeline_smoke(self):
        """Exercise lifting -> encoder -> ETNN -> wrapper -> readout -> loss."""
        # This is the real end-to-end check to run when data/network access is
        # available. It intentionally stays here as executable documentation for
        # the exact command path ETNN must support.
        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=combinatorial/etnn",
                    "dataset=graph/MUTAG",
                    "trainer.max_epochs=1",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "paths=test",
                    "callbacks=model_checkpoint",
                ],
                return_hydra_config=True,
            )

            # The simplified pipeline instantiates the dataset loader,
            # preprocessing/lifting, datamodule, TBModel, trainer, and test
            # loop. Keeping this as an explicit smoke test protects the ETNN
            # config from drifting away from TopoBench's normal execution path.
            run(cfg)
