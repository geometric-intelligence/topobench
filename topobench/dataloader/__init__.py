"""This module implements the dataloader for the topobench package."""

from .cluster_aware_sampler import ClusterAwareNodeSampler, HybridNodeSampler
from .dataload_dataset import DataloadDataset
from .dataloader import TBDataloader
from .extended_context_collate import (
    ExtendedContextCollate,
    create_extended_context_dataloader,
)
from .ondisk_transductive_collate import (
    NodeBatchSampler,
    OnDiskTransductiveCollate,
)
from .structure_centric_collate import (
    StructureCentricCollate,
    create_structure_centric_dataloader,
)
from .structure_centric_sampler import (
    StructureCentricBatchSampler,
    StructureCentricSampler,
)

__all__ = [
    "DataloadDataset",
    "TBDataloader",
    "OnDiskTransductiveCollate",
    "NodeBatchSampler",
    "ClusterAwareNodeSampler",
    "HybridNodeSampler",
    # B1 Bonus: Structure-complete transductive learning
    "StructureCentricSampler",
    "StructureCentricBatchSampler",
    "StructureCentricCollate",
    "ExtendedContextCollate",
    "create_structure_centric_dataloader",
    "create_extended_context_dataloader",
]
