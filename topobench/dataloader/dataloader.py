"TBDataloader class."

from typing import Any

try:
    from lightning import LightningDataModule
except ImportError:
    from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from topobench.dataloader.dataload_dataset import DataloadDataset
from topobench.dataloader.utils import collate_fn


class TBDataloader(LightningDataModule):
    r"""This class takes care of returning the dataloaders for the training, validation, and test datasets.

    It also handles the collate function. The class is designed to work with the `torch` dataloaders.

    Parameters
    ----------
    dataset_train : DataloadDataset
        The training dataset.
    dataset_val : DataloadDataset, optional
        The validation dataset (default: None).
    dataset_test : DataloadDataset, optional
        The test dataset (default: None).
    batch_size : int, optional
        The batch size for the dataloader (default: 1).
    num_workers : int, optional
        The number of worker processes to use for data loading (default: 0).
    pin_memory : bool, optional
        If True, the data loader will copy tensors into pinned memory before returning them (default: False).
    **kwargs : optional
        Additional arguments.

    References
    ----------
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_train: DataloadDataset,
        dataset_val: DataloadDataset = None,
        dataset_test: DataloadDataset = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["dataset_train", "dataset_val", "dataset_test"],
        )
        self.dataset_train = dataset_train
        self.batch_size = batch_size

        if dataset_val is None and dataset_test is None:
            # Transductive setting
            self.dataset_val = dataset_train
            self.dataset_test = dataset_train
            assert self.batch_size == 1, (
                "Batch size must be 1 for transductive setting."
            )
        else:
            self.dataset_val = dataset_val
            self.dataset_test = dataset_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = kwargs.get("persistent_workers", False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset_train={self.dataset_train}, dataset_val={self.dataset_val}, dataset_test={self.dataset_test}, batch_size={self.batch_size})"

    def train_dataloader(self) -> DataLoader:
        r"""Create and return the train dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        r"""Create and return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        r"""Create and return the test dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The test dataloader.
        """
        if self.dataset_test is None:
            raise ValueError("There is no test dataloader.")
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def teardown(self, stage: str | None = None) -> None:
        r"""Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        Parameters
        ----------
        stage : str, optional
            The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"` (default: None).
        """

    def state_dict(self) -> dict[Any, Any]:
        r"""Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        dict
            A dictionary containing the datamodule state that you want to save.
        """
        return {}
