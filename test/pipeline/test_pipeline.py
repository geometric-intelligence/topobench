"""Test pipeline for a particular dataset and model."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from torch_geometric.data import Data

from test._utils.simplified_pipeline import run
from topobench.data.datasets.ocb_dataset import NUM_NODE_TYPES, OCBDataset


DATASET = "graph/OCB101"                                                # ADD YOUR DATASET HERE
MODELS   = ["graph/gcn", "graph/gin", "cell/topotune"]                  # ADD ONE OR SEVERAL MODELS OF YOUR CHOICE HERE
OCB_PROCESSED_FILES = {"OCB101": "data.pt", "OCB301": "data_301.pt"}


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
                        f"dataset={DATASET}", # IF YOU IMPLEMENT A LARGE DATASET WITH AN OPTION TO USE A SLICE OF IT, ADD BELOW THE CORRESPONDING OPTION
                        "trainer.max_epochs=2",
                        "trainer.min_epochs=1",
                        "trainer.check_val_every_n_epoch=1",
                        "paths=test",
                        "callbacks=model_checkpoint",
                    ],
                    return_hydra_config=True
                )
                run(cfg)


def _prepare_dummy_ocb_processed(
    data_root: Path, dataset_name: str, num_graphs: int = 20
) -> None:
    """Create a tiny processed dataset so tests avoid downloading data.

    Parameters
    ----------
    data_root : Path
        Temporary directory to host the fake dataset structure.
    dataset_name : str
        Either ``\"OCB101\"`` or ``\"OCB301\"``.
    num_graphs : int, optional
        Number of synthetic graphs to generate, by default ``20``.
    """
    processed_dir = (
        data_root / "graph" / "circuits" / dataset_name / "processed"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / OCB_PROCESSED_FILES[dataset_name]

    data_list = []
    for idx in range(num_graphs):
        num_nodes = idx % 3 + 1
        x = torch.zeros((num_nodes, NUM_NODE_TYPES + 1), dtype=torch.float)
        x[:, -1] = float(idx)
        if num_nodes > 1:
            edge_index = torch.vstack(
                [
                    torch.arange(0, num_nodes - 1, dtype=torch.long),
                    torch.arange(1, num_nodes, dtype=torch.long),
                ]
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor([float(idx)], dtype=torch.float),
                vid=torch.arange(num_nodes, dtype=torch.long),
                valid=torch.tensor([1], dtype=torch.long),
            )
        )
    data, slices = OCBDataset.collate(data_list)
    torch.save((data, slices), processed_path)


class TestOCBPipelineSmoke:
    """Synthetic pipeline tests for OCB datasets."""

    @pytest.mark.parametrize("dataset_name", ["OCB101", "OCB301"])
    def test_ocb_pipeline_runs(self, tmp_path: Path, dataset_name: str):
        """Ensure the OCB configs run end-to-end on dummy data.

        Parameters
        ----------
        tmp_path : Path
            Pytest temporary directory for synthetic data artifacts.
        dataset_name : str
            Dataset identifier from the parametrization list.
        """
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        _prepare_dummy_ocb_processed(tmp_path, dataset_name)

        with hydra.initialize(
            config_path="../../configs", job_name="ocb", version_base="1.3"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"dataset=graph/{dataset_name}",
                    "model=graph/gcn",
                    "paths=test",
                    f"paths.data_dir={tmp_path}",
                    "paths.output_dir=${paths.data_dir}/outputs",
                    "callbacks=model_checkpoint",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "trainer.max_epochs=1",
                    "trainer.min_epochs=1",
                    "trainer.check_val_every_n_epoch=1",
                    "+trainer.limit_train_batches=2",
                    "+trainer.limit_val_batches=1",
                    "+trainer.limit_test_batches=1",
                ],
                return_hydra_config=True,
            )
            run(cfg)
