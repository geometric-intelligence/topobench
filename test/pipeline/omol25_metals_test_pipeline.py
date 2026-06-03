"""Pipeline test for OMol25 metals dataset loader and training."""

import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from topobench.data.loaders.hypergraph.omol25_metals_dataset_loader import (
    OMol25MetalsDatasetLoader,
)


def _make_omol25_metals_loaders(tmp_path):
    """Create train/val/test loaders for OMol25 metals for a small smoke test.

    This function uses the OMol25MetalsDatasetLoader with simple inductive
    random splits and lightweight dataloader parameters, so that CI runtime
    stays reasonable.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture used as the dataset root directory.

    Returns
    -------
    dict
        Dictionary of dataloaders keyed by ``"train"``, ``"val"``, and
        ``"test"``.
    """
    data_root = tmp_path / "omol25_metals_data"
    data_root.mkdir(parents=True, exist_ok=True)

    loader = OMol25MetalsDatasetLoader(
        parameters=DictConfig({
            "data_domain": "hypergraph",
            "data_type": "omol25_metals",
            "data_name": "omol25_metals",
            "data_dir": str(data_root),
        })
    )

    split_params = {
        "learning_setting": "inductive",
        "split_type": "random_in_train",
        "data_seed": 0,
        "train_prop": 0.8,
        "val_prop": 0.1,
    }
    dataloader_params = {
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
    }

    _, loaders = loader(split_params=split_params, dataloader_params=dataloader_params)
    return loaders


class SimpleGCN(nn.Module):
    """Small GCN-style model for a regression smoke test on OMol25 metals.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int, optional
        Number of hidden channels, by default 32.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        """Initialize SimpleGCN model.

        Parameters
        ----------
        in_channels : int
            Number of input node features.
        hidden_channels : int, optional
            Number of hidden channels, by default 32.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, batch):
        """Forward pass through the GCN model.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch of graph data.

        Returns
        -------
        torch.Tensor
            Predicted output values.
        """
        x = batch.x
        edge_index = batch.edge_index

        if not hasattr(batch, "batch") or batch.batch is None:
            batch.batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch_index = batch.batch

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch_index)
        out = self.lin(x).view(-1)
        return out


def test_omol25_metals_pipeline_training_step(tmp_path):
    """Run a single training step on OMol25 metals to test the full pipeline.

    The test downloads the dataset (if needed), constructs inductive splits,
    wraps them into dataloaders, instantiates a simple GCN, and performs one
    optimization step with an MSE loss. This is only a smoke test; it does
    not measure task performance.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture used as the dataset root directory.

    Returns
    -------
    None
        This function only performs assertions and has no return value.
    """
    loaders = _make_omol25_metals_loaders(tmp_path)
    train_loader = loaders["train"]

    first_batch = next(iter(train_loader))
    in_channels = first_batch.num_node_features

    model = SimpleGCN(in_channels=in_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    batch = next(iter(train_loader))

    pred = model(batch)
    target = batch.y.view_as(pred).float()
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_eval = model(batch)

    assert pred_eval.shape == target.shape
