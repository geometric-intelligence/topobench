"""TRAWL configs and the real TopoBench training path on synthetic data."""

import json
import os
import shutil
from pathlib import Path

import hydra
import pytest
import torch
import torch_geometric
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.dataloader import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.run import run
from topobench.transforms.data_transform import DataTransform

# cuBLAS reads its workspace configuration when first used in a process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CONFIGS = str(Path(__file__).resolve().parents[2] / "configs")


def _compose(overrides):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        return compose(config_name="run", overrides=overrides)


@pytest.mark.parametrize(
    "model,dataset,transforms,max_rank",
    [
        ("cell/trawl", "graph/MUTAG", ["graph2cell_lifting", "trawl"], 2),
        (
            "cell/trawl",
            "graph/PROTEINS",
            ["data_manipulations", "graph2cell_lifting", "trawl"],
            2,
        ),
        (
            "cell/trawl",
            "graph/IMDB-BINARY",
            [
                "data_manipulations",
                "one_hot_node_degree_features",
                "graph2cell_lifting",
                "trawl",
            ],
            2,
        ),
        (
            "simplicial/trawl",
            "graph/MUTAG",
            ["graph2simplicial_lifting", "trawl"],
            2,
        ),
        (
            "simplicial/trawl",
            "simplicial/mantra_betti_numbers",
            ["trawl"],
            2,
        ),
        (
            "combinatorial/trawl",
            "graph/MUTAG",
            ["graph2combinatorial_lifting", "trawl"],
            2,
        ),
        (
            "hypergraph/trawl",
            "graph/MUTAG",
            ["graph2hypergraph_lifting", "trawl"],
            1,
        ),
        ("hypergraph/trawl", "hypergraph/cocitation_cora", ["trawl"], 1),
        ("graph/trawl", "graph/cocitation_cora", ["trawl"], 0),
    ],
)
def test_default_transforms_per_domain(model, dataset, transforms, max_rank):
    """Each domain config gets the standard lifting plus the TRAWL transform.

    Dataset defaults (e.g. degree features for IMDB-BINARY) are kept.
    """
    cfg = _compose([f"model={model}", f"dataset={dataset}", "logger=[]"])
    assert list(cfg.transforms) == transforms
    assert cfg.transforms.trawl.max_rank == max_rank
    instance = instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    assert type(instance.backbone.backbone) is TRAWL
    assert instance.backbone.backbone.max_rank == max_rank


def test_other_models_keep_core_defaults():
    cfg = _compose(["model=cell/cwn", "dataset=graph/MUTAG", "logger=[]"])
    assert cfg.get("pretraining") is None
    assert not cfg.checkpoint_selection.report
    assert cfg.checkpoint_selection.strategy == "best"
    assert (
        cfg.callbacks.model_checkpoint._target_
        == "lightning.pytorch.callbacks.ModelCheckpoint"
    )


def _molecule(seed, n):
    ring = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n])
    edges = torch.cat([ring, torch.tensor([[0], [2]])], 1)
    edges = torch_geometric.utils.to_undirected(edges, num_nodes=n)
    generator = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(n, 7, generator=generator),
        edge_index=edges,
        y=torch.tensor([seed % 2]),
        num_nodes=n,
    )


@pytest.mark.parametrize(
    "model",
    [
        "cell/trawl",
        "simplicial/trawl",
        "combinatorial/trawl",
        "hypergraph/trawl",
        "graph/trawl",
    ],
)
def test_domain_configs_train_on_lifted_graphs(model):
    cfg = _compose(
        [
            f"model={model}",
            "dataset=graph/MUTAG",
            "logger=[]",
            "model.backbone.hidden_dim=16",
            "model.backbone.architecture=mlp",
            "model.backbone.depth=1",
        ]
    )
    transforms = [DataTransform(**dict(value)) for value in cfg.transforms.values()]
    graphs = []
    for seed in range(3):
        graph = _molecule(seed, 5 + seed)
        for transform in transforms:
            graph = transform(graph)
        graphs.append(graph)
    for transform in transforms:
        finalize = getattr(transform.transform, "finalize_dataset", None)
        if finalize is not None:
            graphs = finalize(graphs)
    net = instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    net.backbone.initialize(graphs)
    dataset = DataloadDataset(graphs)
    batch = collate_fn([dataset[i] for i in range(len(graphs))])
    output = net.readout(model_out=net.backbone(batch), batch=batch)
    assert output["logits"].shape == (3, 2)
    output["logits"].sum().backward()


RECIPES = [
    "proteins_mamba",
    "proteins_hybrid",
    "nci1_hybrid",
    "nci1_sisa",
    "nci1_mamba",
    "custom_layers",
    "adjacency",
    "mixed",
]


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_compiles(recipe):
    cfg = _compose([f"experiment=trawl_{recipe}", "logger=[]"])
    model = instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    assert type(model.backbone.backbone) is TRAWL
    assert len(model.backbone.backbone.encoders[0]) == 5
    if recipe.startswith(("proteins", "nci1")):
        assert cfg.dataset.split_params.split_type == "seeded_stratified"
        assert cfg.transforms.graph2cell_lifting.max_cell_length == 6
        assert cfg.pretraining is not None
        assert cfg.checkpoint_selection.report


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
    overrides = [
        "model=cell/trawl"
        if profile == "base"
        else "experiment=trawl_proteins_mamba",
        "dataset=graph/PROTEINS",
        "logger=csv",
        "dataset.loader._target_=test._utils.trawl.TinyLoader",
        f"dataset.loader.parameters.data_dir={tmp_path.as_posix()}/data",
        f"dataset.split_params.data_split_dir={tmp_path.as_posix()}/splits",
        f"paths.output_dir={tmp_path.as_posix()}/output",
        f"paths.log_dir={tmp_path.as_posix()}/logs",
        f"paths.work_dir={tmp_path.as_posix()}",
        "callbacks.model_summary=null",
        "model.backbone.hidden_dim=16",
        f"model.backbone.depth={1 if accelerator == 'cpu' else 2}",
        f"model.backbone.architecture={'mlp' if accelerator == 'cpu' else 'hybrid'}",
        "model.backbone_wrapper.walks.k=2",
        "model.backbone_wrapper.walks.length=4",
        "model.backbone_wrapper.walks.guidance=null",
        "model.backbone_wrapper.eval_views=2",
        "transforms.trawl.encodings.RWSE.max_pe_dim=2",
        f"trainer={'gpu' if accelerator == 'gpu_ddp' else accelerator}",
        "trainer.max_epochs=2",
        f"trainer.check_val_every_n_epoch={check_val}",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
        "dataset.dataloader_params.batch_size=4",
        f"checkpoint_selection.strategy={strategy}",
        "checkpoint_selection.top_k=2",
        "callbacks.model_checkpoint.save_top_k=2",
        "checkpoint_selection.report=true",
    ]
    if pretrain:
        overrides += [
            "pretraining=trawl",
            f"pretraining.max_epochs={2 * check_val - 1}",
        ]
    else:
        overrides.append("~pretraining")
    cfg = _compose(overrides)
    if accelerator == "gpu_ddp":
        cfg.trainer.devices = 2
        OmegaConf.update(cfg, "trainer.strategy", "ddp_spawn", force_add=True)
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
        instantiate(cfg.pretraining)(
            model=objects["model"],
            datamodule=datamodule,
            trainer_config=cfg.trainer,
            output_dir=tmp_path / "output",
        )
        resumed = torch.load(cfg.pretraining.ckpt_path, weights_only=False)
        assert resumed["epoch"] == 1
        assert datamodule.order_seed == original_seed
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
        # A supplied checkpoint is evaluated but never deleted.
        cfg.delete_checkpoint_after_test = True
        replay_metrics, _ = run(cfg)
        assert Path(cfg.ckpt_path).is_file()
        assert float(replay_metrics["test_best_rerun/loss"]) == pytest.approx(
            float(metrics["test_best_rerun/loss"]), abs=1e-6
        )
        replay = json.loads(
            (tmp_path / "replay/evaluation_report.json").read_text()
        )
        assert replay["checkpoint_strategy"] == strategy
        assert replay["selected_checkpoints"]
        assert all(
            Path(item["path"]).parent == relocated
            for item in replay["selected_checkpoints"]
        )
