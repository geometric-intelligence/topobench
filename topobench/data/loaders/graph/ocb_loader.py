"""OCB Circuit Dataset Loader for TopoBench."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from topobench.data.datasets.ocb_dataset import OCB101Dataset, OCB301Dataset
from topobench.data.loaders.base import AbstractLoader


class OCBDatasetLoader(AbstractLoader):
    """Single loader that dispatches to the correct OCB dataset class.

    Parameters
    ----------
    parameters : DictConfig
        Configuration with ``data_name`` specifying the desired OCB dataset.
    """

    _DATASETS: dict[str, type[Any]] = {
        "OCB101": OCB101Dataset,
        "OCB301": OCB301Dataset,
    }

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Any:
        """Load the requested OCB dataset.

        Returns
        -------
        Any
            Instantiated PyG dataset.

        Raises
        ------
        ValueError
            If ``data_name`` is missing or unsupported.
        RuntimeError
            If dataset initialization fails.
        """
        data_name = getattr(self.parameters, "data_name", None)
        if data_name is None:
            raise ValueError(
                "parameters.data_name must be provided for OCBDatasetLoader"
            )

        try:
            dataset_cls = self._DATASETS[data_name]
        except KeyError as exc:
            supported = ", ".join(sorted(self._DATASETS))
            raise ValueError(
                f"Unsupported OCB dataset '{data_name}'. "
                f"Supported datasets: {supported}"
            ) from exc

        try:
            # Each dataset manages its own processed/raw sub-folders under this root.
            return dataset_cls(
                root=str(self.get_data_dir()), parameters=self.parameters
            )
        except Exception as exc:  # pragma: no cover - rethrow context
            raise RuntimeError(
                f"Error loading OCB dataset '{data_name}': {exc}"
            ) from exc
