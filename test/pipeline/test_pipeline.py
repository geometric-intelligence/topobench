import hydra

from test._utils.simplified_pipeline import run

DATASET = "graph/MUTAG"  # handy spot to swap dataset
MODELS = [
    "graph/gcn",
    "graph/esc_gnn",
    "cell/topotune",
    "simplicial/topotune",
]  # one model or a few


class TestPipeline:
    def setup_method(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_pipeline(self):
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
