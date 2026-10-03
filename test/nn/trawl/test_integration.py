"""Run the real TopoBench training/preprocessing path on local synthetic data."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset

from topobench.model.trawl_pretraining import run_pretraining
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.run import run


class TinyDataset(InMemoryDataset):
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
    def __init__(self, parameters):
        self.directory = parameters.data_dir

    def load(self):
        return TinyDataset(), self.directory


@pytest.mark.parametrize(
    "recipe",
    [
        "proteins_mamba",
        "proteins_hybrid",
        "nci1_hybrid",
        "nci1_sisa",
        "nci1_mamba",
        "zinc",
        "custom_layers",
        "adjacency",
        "mixed",
    ],
)
def test_recipe_compiles(recipe):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="run",
            overrides=[f"experiment=trawl/{recipe}", "logger=[]"],
        )
        model = instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )
        assert type(model.backbone) is TRAWL
        assert len(model.backbone.encoders[0]) == (
            4 if recipe == "zinc" else 5
        )
        if recipe.startswith(("proteins", "nci1")):
            assert cfg.dataset.split_params.split_type == "seeded_stratified"
            assert cfg.transforms.lifting.max_cell_length == 6


@pytest.mark.parametrize(
    "pretrain,profile,strategy,check_val",
    [
        (False, "base", "best", 1),
        (True, "base", "best", 1),
        (True, "recipe", "weight_average", 1),
        (False, "recipe", "logit_ensemble", 1),
        # Sparse validation: plateau schedulers must step only when validated.
        (True, "base", "best", 2),
    ],
)
@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "gpu_ddp"])
def test_native_runner(
    tmp_path, pretrain, profile, strategy, check_val, accelerator
):
    if accelerator != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU runner validation")
    if accelerator == "gpu_ddp" and torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required for DDP validation")
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[3] / "configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="run",
            overrides=[
                "model=combinatorial/trawl"
                if profile == "base"
                else "experiment=trawl/proteins_mamba",
                "dataset=graph/PROTEINS",
                "logger=csv",
                "dataset.loader._target_=test.nn.trawl.test_integration.TinyLoader",
                f"dataset.loader.parameters.data_dir={tmp_path.as_posix()}/data",
                f"dataset.split_params.data_split_dir={tmp_path.as_posix()}/splits",
                f"paths.output_dir={tmp_path.as_posix()}/output",
                f"paths.log_dir={tmp_path.as_posix()}/logs",
                f"paths.work_dir={tmp_path.as_posix()}",
                "callbacks.model_summary=null",
                "model.backbone.hidden_dim=16",
                f"model.backbone.depth={1 if accelerator == 'cpu' else 2}",
                f"model.backbone.architecture={'mlp' if accelerator == 'cpu' else 'hybrid'}",
                "model.backbone.walks.k=2",
                "model.backbone.walks.length=4",
                "model.backbone.walks.guidance=null",
                "transforms.trawl.encodings.rw_samples=2",
                f"trainer={'gpu' if accelerator == 'gpu_ddp' else accelerator}",
                "trainer.max_epochs=2",
                f"trainer.check_val_every_n_epoch={check_val}",
                "+trainer.enable_progress_bar=false",
                "+trainer.enable_model_summary=false",
                "dataset.dataloader_params.batch_size=4",
                f"pretraining.enabled={str(pretrain).lower()}",
                f"pretraining.max_epochs={2 * check_val - 1}",
                f"evaluation.checkpoint={strategy}",
                "evaluation.top_k=2",
                "evaluation.walk_views=2",
                "callbacks.model_checkpoint.save_top_k=2",
            ],
        )
        if accelerator == "gpu_ddp":
            cfg.trainer.devices = 2
            OmegaConf.update(
                cfg,
                "trainer.strategy",
                "ddp_spawn_find_unused_parameters_true",
                force_add=True,
            )
        metrics, objects = run(cfg)
        assert "val/accuracy" in metrics
        assert "test_best_rerun/accuracy" in metrics
        # Late rerun rows must be flushed to the finalized CSV logger.
        header = (
            (tmp_path / "output/csv/version_0/metrics.csv")
            .read_text()
            .splitlines()[0]
        )
        assert "test_best_rerun/accuracy" in header
        assert objects["callbacks"][0] is not None
        assert (tmp_path / "output/trawl_manifest.json").is_file()
        if pretrain:
            assert (tmp_path / "output/pretraining/last.ckpt").is_file()
            assert list(
                (tmp_path / "output/pretraining/metrics").glob("*/metrics.csv")
            )
        if pretrain and profile == "recipe" and accelerator == "cpu":
            datamodule = objects["datamodule"]
            original_seed = datamodule.order_seed
            cfg.pretraining.ckpt_path = str(
                tmp_path / "output/pretraining/last.ckpt"
            )
            cfg.pretraining.max_epochs = 2
            run_pretraining(
                objects["model"],
                datamodule,
                cfg.pretraining,
                cfg.trainer,
                output_dir=tmp_path / "output",
            )
            resumed = torch.load(cfg.pretraining.ckpt_path, weights_only=False)
            assert resumed["epoch"] == 1
            assert datamodule.order_seed == original_seed
        manifest = json.loads(
            (tmp_path / "output/trawl_manifest.json").read_text()
        )
        assert (
            "nn/backbones/combinatorial/trawl.py"
            in manifest["implementation_sha256"]
        )
        assert (
            "data/utils/trawl/sampling.py" in manifest["implementation_sha256"]
        )
        assert "nn/readouts/trawl.py" in manifest["implementation_sha256"]
        assert (
            "callbacks/model_checkpoint.py"
            in manifest["implementation_sha256"]
        )
        assert "evaluator/evaluator.py" in manifest["implementation_sha256"]
        if accelerator == "cpu":
            checkpoint = next(
                item
                for item in objects["callbacks"]
                if getattr(item, "best_model_path", None)
            )
            source = Path(checkpoint.best_model_path)
            relocated = tmp_path / "relocated_checkpoints"
            shutil.copytree(source.parent, relocated)
            cfg.train = False
            cfg.ckpt_path = str(relocated / source.name)
            cfg.paths.output_dir = str(tmp_path / "replay")
            replay_metrics, _ = run(cfg)
            assert float(
                replay_metrics["test_best_rerun/loss"]
            ) == pytest.approx(
                float(metrics["test_best_rerun/loss"]), abs=1e-6
            )
            replay = json.loads(
                (tmp_path / "replay/trawl_evaluation.json").read_text()
            )
            assert replay["selected_checkpoints"]
            assert all(
                Path(item["path"]).parent == relocated
                for item in replay["selected_checkpoints"]
            )
