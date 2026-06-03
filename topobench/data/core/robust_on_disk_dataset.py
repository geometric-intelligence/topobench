import os
from collections.abc import Callable

import torch
from torch_geometric.data import Data, Dataset


class RobustOnDiskDataset(Dataset):
    """
    A robust Chunk-based on-disk dataset that processes raw samples
    individually and saves them in chunks to avoid memory bottlenecks.
    """

    def __init__(
        self,
        root: str,
        chunk_size: int = 512,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self.chunk_size = chunk_size
        super().__init__(root, transform, pre_transform, pre_filter)

        if not self._is_processed():
            self.process()

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def processed_file_names(self) -> list[str]:
        if not os.path.exists(self.processed_dir):
            return []
        files = [
            f
            for f in os.listdir(self.processed_dir)
            if f.startswith("chunk_") and f.endswith(".pt")
        ]
        return sorted(files)

    def _is_processed(self) -> bool:
        return len(self.processed_file_names) > 0

    def len(self) -> int:
        # Sum of all samples in chunks
        total = 0
        for fname in self.processed_file_names:
            chunk = torch.load(os.path.join(self.processed_dir, fname))
            total += len(chunk)
        return total

    def get(self, idx: int) -> Data:
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Index {idx} out of range")
        chunk_id = idx // self.chunk_size
        within = idx % self.chunk_size
        chunk_path = os.path.join(self.processed_dir, f"chunk_{chunk_id}.pt")
        chunk = torch.load(chunk_path)
        data = chunk[within]
        if self.transform:
            data = self.transform(data)
        return data

    # -------------------------
    # To be implemented by subclass
    # -------------------------
    def prepare_raw_data(self) -> list[str]:
        raise NotImplementedError

    def process_raw_file(self, raw_path: str) -> list[Data]:
        raise NotImplementedError

    # -------------------------
    # Main processing logic
    # -------------------------
    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_paths = self.prepare_raw_data()
        all_data = []

        for raw_path in raw_paths:
            samples = self.process_raw_file(raw_path)
            for data in samples:
                if self.pre_filter and not self.pre_filter(data):
                    continue
                if self.pre_transform:
                    data = self.pre_transform(data)
                all_data.append(data)

            while len(all_data) >= self.chunk_size:
                self._save_chunk(all_data[: self.chunk_size])
                all_data = all_data[self.chunk_size:]

        if len(all_data) > 0:
            self._save_chunk(all_data)

    def _save_chunk(self, chunk: list[Data]):
        chunk_id = len(self.processed_file_names)
        path = os.path.join(self.processed_dir, f"chunk_{chunk_id}.pt")
        torch.save(chunk, path)
