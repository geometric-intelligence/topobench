"""Test pipeline for on-disk preprocessing with MUTAG dataset."""

import hydra
from test._utils.simplified_pipeline import run


DATASET = "graph/MUTAG_ondisk"                                          # On-disk preprocessing dataset
MODELS   = ["graph/gcn", "cell/topotune", "simplicial/topotune"]        # Test multiple models


class TestOnDiskPipeline:
    """Test pipeline for on-disk preprocessing."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    def test_pipeline(self):
        """Test pipeline with on-disk preprocessing."""
        with hydra.initialize(config_path="../../configs", job_name="test_on_disk"):
            for MODEL in MODELS:
                cfg = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        f"model={MODEL}",
                        f"dataset={DATASET}",
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True
                )
                run(cfg)