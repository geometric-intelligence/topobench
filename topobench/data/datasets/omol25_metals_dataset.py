"""OMol25 metals dataset integration for TopoBench."""

import os
import time
from collections.abc import Callable
from urllib.error import URLError

import torch
from torch_geometric.data import InMemoryDataset, download_url


class OMol25MetalsDataset(InMemoryDataset):
    r"""Metal-complex subset of OMol25 as a PyG dataset.

    The dataset stores the preprocessed file ``processed/data.pt`` under
    ``root``. The file is produced by an external pipeline that converts
    OMol25 molecules into :class:`torch_geometric.data.Data` objects and
    serializes them with :class:`torch_geometric.data.InMemoryDataset`.

    Parameters
    ----------
    root : str
        Root directory for the dataset. The file ``processed/data.pt`` will
        be stored under this directory.
    transform : callable, optional
        Callable applied to each data object on-the-fly.
    pre_transform : callable, optional
        Callable applied before saving data objects to disk.
    """

    url: str = (
        "https://github.com/demiqin/omol25_metals/raw/main/"
        "data/omol25_metals/subset/processed/data.pt"
    )

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root, transform=transform, pre_transform=pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return list of raw file names.

        Returns
        -------
        list of str
            Empty list, because raw OMol25 files are not stored locally.
        """
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Return list of processed file names.

        Returns
        -------
        list of str
            List containing ``"data.pt"``.
        """
        return ["data.pt"]

    def download(self) -> None:
        """Download the preprocessed file into ``processed/data.pt``.

        The file is fetched from the NREL GitHub URL defined in
        :attr:`url`. Includes retry logic for network resilience.
        """
        os.makedirs(self.processed_dir, exist_ok=True)

        target_path = self.processed_paths[0]
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                # download_url returns the path to the downloaded file
                # It extracts the filename from the URL, so we get a file
                # named 'data.pt' in self.processed_dir
                local_path = download_url(self.url, self.processed_dir)

                # Ensure the downloaded file is at the expected location
                if os.path.abspath(local_path) != os.path.abspath(target_path):
                    # If names don't match, rename to expected name
                    os.replace(local_path, target_path)
                return  # Success
            except URLError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(
                        f"Download attempt {attempt + 1} failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(
                        f"Download attempt {attempt + 1} failed with error: {e}, retrying..."
                    )
                    time.sleep(retry_delay)

        # If all retries failed, raise the last error
        if last_error:
            raise RuntimeError(
                f"Failed to download {self.url} after {max_retries} "
                f"attempts: {last_error}"
            ) from last_error

    def process(self) -> None:
        """Convert raw data into processed form.

        All preprocessing is performed externally, so this method is a no-op.
        It is only defined to satisfy the :class:`InMemoryDataset` interface.
        """
        # Nothing to do: ``download`` already creates ``processed/data.pt``.
        return
