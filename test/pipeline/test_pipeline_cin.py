"""Pipeline integration test for CIN.

Verifies that the full TopoBench training pipeline runs end-to-end
with the corrected CIN model on a compatible cell-complex dataset
(MUTAG lifted to a cell complex via the dataset's default transforms).

Follows the canonical pattern in ``test/pipeline/test_pipeline.py``:
compose the Hydra config which have a short training run through the
shared ``simplified_pipeline.run`` helper (no subprocess).

"""

import hydra

from test._utils.simplified_pipeline import run

DATASET = "graph/MUTAG"
MODELS = ["cell/cin"]


class TestPipelineCIN:
    """End-to-end pipeline smoke test for CIN."""

    def setup_method(self):
        """Clear any global Hydra state before composing a config."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline_cin(self):
        """CIN on MUTAG (lifted to a cell complex) must train end-to-end."""
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
                    return_hydra_config=True,
                )
                run(cfg)
