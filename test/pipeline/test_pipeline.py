"""Required challenge pipeline test for coordinate-policy ETNN.

The challenge asks each submission to fill this file with a compatible dataset
and model config. This test exercises the normal TopoBench training path for
the submitted GraphUniverse-compatible coordinate-policy ETNN-LapPE config on
MUTAG: Hydra composition, LapPE preprocessing, graph-to-combinatorial lifting,
feature encoding, ETNN message passing, readout, loss, and trainer.
"""

import hydra

from test._utils.simplified_pipeline import run

DATASET = "graph/MUTAG"
MODELS = ["combinatorial/etnn_coordinate_policy_lappe"]


class TestPipeline:
    """Run the submitted ETNN config through the standard TopoBench pipeline."""

    def setup_method(self):
        """Reset Hydra so this test is isolated from previous compositions."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
        """Verify the submitted ETNN policy trains end-to-end."""
        with hydra.initialize(
            version_base="1.3",
            config_path="../../configs",
            job_name="job",
        ):
            for MODEL in MODELS:
                # MUTAG is intentionally small, so this is a practical CI smoke
                # test while still covering the full LapPE preprocessing ->
                # lifting -> ETNN -> readout execution path required by the
                # challenge. Other coordinate policies are covered in
                # test_etnn_pipeline.py; physical mode is validated separately
                # on QM9 because MUTAG/GraphUniverse do not provide physical
                # positions.
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
