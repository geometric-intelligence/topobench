"""Test pipeline for a particular dataset and model."""

import hydra
import pytest

from test._utils.simplified_pipeline import run


DATASET = "graph/MUTAG"  # ADD YOUR DATASET HERE
MODELS = [  # simplicial TNN variants (MUTAG auto-lifts to a clique complex)
    "simplicial/poly_filter_tnn",
    "simplicial/poly_filter_tnn_monomial",
    "simplicial/poly_filter_tnn_jacobi",
    "simplicial/filter_bank_tnn",
]


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
