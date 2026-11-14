"""Test pipeline for a particular dataset and model."""

import hydra
import pytest

from test._utils.simplified_pipeline import run


DATASET = "graph/MUTAG"  # ADD YOUR DATASET HERE
MODELS = [
    "graph/gcn",
    "cell/topotune",
    "simplicial/topotune",
    # DSNN, both operator paths. `dsnn` is the faithful config: MUTAG is
    # undirected, so its Laplacian is provably real and q is inert (Thm 3).
    # `dsnn_degree` induces an orientation, so the complex terms are non-zero
    # and the Hermitian assembly is exercised end to end. The `dsnn_ortho` and
    # `dsnn_general` variants are left out on purpose: they run the same real
    # path as `dsnn` here, and their builders are covered by the unit tests.
    "graph/dsnn",
    "graph/dsnn_degree",
]  # ADD ONE OR SEVERAL MODELS


class TestPipeline:
    """Test pipeline for a particular dataset and model."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Test pipeline."""
        with hydra.initialize(config_path="../../configs", job_name="job"):
            for MODEL in MODELS:
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={DATASET}",
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "trainer.accelerator=cpu",
                        "trainer.devices=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True
                )
                run(cfg)
