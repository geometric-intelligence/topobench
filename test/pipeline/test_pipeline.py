"""Test pipeline for a particular dataset and model."""

import hydra
from test._utils.simplified_pipeline import run


DATASET = "graph/ba2motif"  # ADD YOUR DATASET HERE
MODELS = ["graph/gcn"]  # ADD ONE OR SEVERAL MODELS OF YOUR CHOICE HERE


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
                        f"dataset={DATASET}",  # IF YOU IMPLEMENT A LARGE DATASET WITH AN OPTION TO USE A SLICE OF IT, ADD BELOW THE CORRESPONDING OPTION
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True,
                )
                run(cfg)
