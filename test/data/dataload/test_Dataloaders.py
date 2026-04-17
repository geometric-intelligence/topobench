"""Test the Dataloader class."""

import hydra
import rootutils
import torch

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.dataloader.utils import to_data_list

from omegaconf import OmegaConf
import os
from topobench.run import initialize_hydra

# rootutils.setup_root("./", indicator=".project-root", pythonpath=True)


class TestCollateFunction:
    """Test collate_fn."""

    def setup_method(self):
        """Setup the test."""

        hydra.initialize(
        version_base="1.3", config_path="../../../configs", job_name="run"
        )
        cfg = hydra.compose(config_name="run.yaml", overrides=["dataset=graph/NCI1"])

        graph_loader = hydra.utils.instantiate(cfg.dataset.loader, _recursive_=False)

        datasets, dataset_dir = graph_loader.load()
        preprocessor = PreProcessor(datasets, dataset_dir, None)
        dataset_train, dataset_val, dataset_test = (
            preprocessor.load_dataset_splits(cfg.dataset.split_params)
        )

        self.batch_size = 2
        datamodule = TBDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            batch_size=self.batch_size,
        )
        self.val_dataloader = datamodule.val_dataloader()
        self.val_dataset = dataset_val

    def test_lift_features(self):
        """Test the collate function.

        To test the collate function we use the TBDataloader class to create a dataloader that uses the collate function.
        We first check that the batched data has the expected shape. We then convert the batched data back to a list and check that the data in the list is the same as the original data.
        """

        def check_shape(batch, elems, key):
            """Check that the batched data has the expected shape.

            Parameters
            ----------
            batch : dict
                The batched data.
            elems : list
                A list of the original data.
            key : str
                The key of the data to check.
            """
            if "x_" in key or key == "x":  # or "edge_attr" in key:
                rows = 0
                for i in range(len(elems)):
                    rows += elems[i][key].shape[0]
                assert batch[key].shape[0] == rows
                assert batch[key].shape[1] == elems[0][key].shape[1]
            elif "edge_index" in key:
                cols = 0
                for i in range(len(elems)):
                    cols += elems[i][key].shape[1]
                assert batch[key].shape[0] == 2
                assert batch[key].shape[1] == cols
            # elif "batch_" in key:
            #    assert 0
            #    rows = 0
            #    n = int(key.split("_")[1])
            #    for i in range(len(elems)):
            #        rows += elems[i][f"x_{n}"].shape[0]
            #    assert batch[key].shape[0] == rows
            elif key in elems[0]:
                for i in range(len(batch[key].shape)):
                    i_elems = 0
                    for j in range(len(elems)):
                        i_elems += elems[j][key].shape[i]
                    assert batch[key].shape[i] == i_elems

        def check_separation(matrix, n_elems_0_row, n_elems_0_col):
            """Check that the matrix is separated into two parts diagonally concatenated.

            Parameters
            ----------
            matrix : torch.Tensor
                The matrix to check.
            n_elems_0_row : int
                The number of elements in the first part of the matrix.
            n_elems_0_col : int
                The number of elements in the first part of the matrix.
            """
            assert torch.all(matrix[:n_elems_0_row, n_elems_0_col:] == 0)
            assert torch.all(matrix[n_elems_0_row:, :n_elems_0_col] == 0)

        def check_values(matrix, m1, m2):
            """Check that the values in the matrix are the same as the values in the original data.

            Parameters
            ----------
            matrix : torch.Tensor
                The matrix to check.
            m1 : torch.Tensor
                The first part of the matrix.
            m2 : torch.Tensor
                The second part of the matrix.
            """
            assert torch.allclose(matrix[: m1.shape[0], : m1.shape[1]], m1)
            assert torch.allclose(matrix[m1.shape[0] :, m1.shape[1] :], m2)

        # assert 0
        batch = next(iter(self.val_dataloader))
        elems = [self.val_dataset.data_lst[i] for i in range(self.batch_size)]

        # Check shape
        for key, val in batch:
            check_shape(batch, elems, key)

        # Check that the batched data is separated correctly and the values are correct
        for key, val in batch:
            if "incidence_" in key:
                i = int(key.split("_")[1])
                if i == 0:
                    n0_row = 1
                else:
                    n0_row = torch.sum(batch[f"batch_{i-1}"] == 0)
                n0_col = torch.sum(batch[f"batch_{i}"] == 0)
                check_separation(batch[key].to_dense(), n0_row, n0_col)
                check_values(
                    batch[key].to_dense(),
                    elems[0][key].to_dense(),
                    elems[1][key].to_dense(),
                )

        # Check that going back to a list of data gives the same data
        batch_list = to_data_list(batch)
        assert len(batch_list) == len(elems)
        for i in range(len(batch_list)):
            for key in elems[i]:
                if key in batch_list[i]:
                    if batch_list[i][key].is_sparse:
                        assert torch.all(
                            batch_list[i][key].coalesce().indices()
                            == elems[i][key].coalesce().indices()
                        )
                        assert torch.allclose(
                            batch_list[i][key].coalesce().values(),
                            elems[i][key].coalesce().values(),
                        )
                        assert batch_list[i][key].shape, elems[i][key].shape
                    else:
                        assert torch.allclose(
                            batch_list[i][key], elems[i][key]
                        )


if __name__ == "__main__":
    t = TestCollateFunction()
    t.setup_method()
    t.test_lift_features()
