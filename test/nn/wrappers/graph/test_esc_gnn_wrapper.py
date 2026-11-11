from pathlib import Path

import hydra
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from torch_geometric.data import Data

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.dataloader.utils import DomainData
from topobench.nn.wrappers.graph.esc_gnn_wrapper import ESCGNNWrapper
from topobench.utils.config_resolvers import register_all_resolvers


class _RecordingBackbone(torch.nn.Module):
    num_structural_codes = 387

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.calls: list[tuple[torch.Tensor, ...]] = []

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        self.calls.append(args)
        return self.output


def _esc_batch() -> Data:
    return Data(
        x_0=torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        esc_code_id=torch.tensor([0, 300, 1], dtype=torch.long),
        esc_code_count=torch.tensor([2.0, 1.0, 3.0], dtype=torch.float32),
        esc_nnz_per_edge=torch.tensor([2, 1], dtype=torch.long),
        edge_attr=torch.tensor([[5.0], [7.0]]),
        batch_0=torch.zeros(2, dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
    )


def _wrapper(
    output: torch.Tensor,
    *,
    residual_connections: bool = False,
) -> tuple[ESCGNNWrapper, _RecordingBackbone]:
    backbone = _RecordingBackbone(output)
    wrapper = ESCGNNWrapper(
        backbone,
        out_channels=output.size(-1),
        num_cell_dimensions=1,
        residual_connections=residual_connections,
    )
    return wrapper, backbone


def test_wrapper_routes_structure_and_returns_exact_topobench_output():
    batch = _esc_batch()
    expected = torch.tensor([[7.0, 11.0], [13.0, 17.0]])
    wrapper, backbone = _wrapper(expected)

    result = wrapper(batch)

    assert set(result) == {"labels", "batch_0", "x_0"}
    assert result["labels"] is batch.y
    assert result["batch_0"] is batch.batch_0
    assert result["x_0"] is expected
    assert len(backbone.calls) == 1

    routed = backbone.calls[0]
    assert len(routed) == 5
    assert routed[0] is batch.x_0
    assert routed[1] is batch.edge_index
    assert routed[2] is batch.esc_code_id
    assert routed[3] is batch.esc_code_count
    assert routed[4] is batch.esc_nnz_per_edge


@pytest.mark.parametrize(
    ("dataset", "expected_task_level", "expected_out_channels"),
    [
        ("graph/graphuniverse_inductive", "node_inductive", 20),
        ("graph/graphuniverse_inductive_triangle", "graph", 1),
    ],
)
def test_tiny_graphuniverse_preprocesses_batches_and_forwards(
    tmp_path: Path,
    dataset: str,
    expected_task_level: str,
    expected_out_channels: int,
):
    register_all_resolvers()
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).parents[4] / "configs"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    overrides = [
        "model=graph/esc_gnn",
        f"dataset={dataset}",
        f"paths.data_dir={data_dir.as_posix()}",
        f"paths.output_dir={output_dir.as_posix()}",
        (
            "dataset.loader.parameters.generation_parameters."
            "family_parameters.n_graphs=4"
        ),
        (
            "dataset.loader.parameters.generation_parameters."
            "family_parameters.n_nodes_range=[12,14]"
        ),
        (
            "dataset.loader.parameters.generation_parameters."
            "family_parameters.n_communities_range=[2,3]"
        ),
        "dataset.split_params.train_prop=0.5",
        "dataset.dataloader_params.batch_size=2",
    ]
    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)

    assert cfg.model.backbone_wrapper.residual_connections is False
    assert cfg.model.readout.pooling_type == "sum"
    assert cfg.model.readout.task_level == expected_task_level
    assert cfg.model.feature_encoder.structural_codebook_size == 387
    assert cfg.model.backbone.num_structural_codes == 387
    assert cfg.transforms.ESCStructuralEncoding.encoder_version == (
        "esc-paper-induced-v1"
    )

    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    generated, generated_dir = dataset_loader.load()
    preprocessor = PreProcessor(generated, generated_dir, cfg.transforms)

    train, validation, test = preprocessor.load_dataset_splits(
        cfg.dataset.split_params
    )
    datamodule = TBDataloader(
        dataset_train=train,
        dataset_val=validation,
        dataset_test=test,
        **cfg.dataset.dataloader_params,
    )
    batch = next(iter(datamodule.train_dataloader()))

    assert isinstance(batch, DomainData)

    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    ).eval()
    with torch.no_grad():
        result = model(batch)

    num_graphs = int(batch.batch_0.unique().numel())
    expected_rows = (
        batch.num_nodes
        if expected_task_level == "node_inductive"
        else num_graphs
    )
    assert model.task_level == expected_task_level
    assert result["labels"] is batch.y
    assert result["logits"].shape == (expected_rows, expected_out_channels)
    assert torch.isfinite(result["logits"]).all()
