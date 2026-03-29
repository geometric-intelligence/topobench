"""Tests for ExtendedContextCollate."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

from topobench.data.preprocessor import OnDiskTransductivePreprocessor
from topobench.dataloader import (
    ClusterAwareNodeSampler,
    ExtendedContextCollate,
    NodeBatchSampler,
    create_extended_context_dataloader,
)


@pytest.fixture
def triangle_graph():
    """Create a graph with known triangle structures."""
    edges = [
        [0, 1],
        [1, 2],
        [2, 0],
        [2, 3],
        [3, 4],
        [4, 2],
        [4, 5],
        [5, 6],
        [6, 4],
        [1, 3],
        [5, 7],
        [7, 8],
    ]

    edge_index = torch.tensor(
        [[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long
    )
    edge_index_full = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return Data(
        x=torch.randn(9, 8),
        edge_index=edge_index_full,
        y=torch.randint(0, 3, (9,)),
        num_nodes=9,
    )


@pytest.fixture
def preprocessor_with_structures(triangle_graph, tmp_path):
    """Create preprocessor with built index."""
    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=triangle_graph,
        data_dir=tmp_path / "context_index",
        max_clique_size=3,
    )
    preprocessor.build_index()
    return preprocessor


@pytest.fixture
def larger_graph(tmp_path):
    """Create a larger graph for testing."""
    edge_index = erdos_renyi_graph(
        num_nodes=100, edge_prob=0.15, directed=False
    )

    data = Data(
        x=torch.randn(100, 16),
        edge_index=edge_index,
        y=torch.randint(0, 5, (100,)),
        num_nodes=100,
    )

    preprocessor = OnDiskTransductivePreprocessor(
        graph_data=data,
        data_dir=tmp_path / "large_context_index",
        max_clique_size=3,
    )
    preprocessor.build_index()

    return data, preprocessor


class TestExtendedContextCollate:
    """Test core functionality and expansion strategies."""

    def test_context_expansion_and_structure_completeness(
        self, preprocessor_with_structures
    ):
        """Test context expansion completes structures with correct metadata."""
        collate_fn = ExtendedContextCollate(
            preprocessor_with_structures,
            max_expansion_ratio=2.0,
            include_structure_metadata=True,
        )

        core_nodes = [0, 1]
        batch = collate_fn([core_nodes])

        # Implicitly tests: Data creation, attribute existence, subgraph extraction
        assert batch.num_nodes >= 3  # Should expand to complete triangle
        assert batch.core_mask.sum().item() == len(core_nodes)
        assert batch.num_structures > 0
        assert (
            abs(batch.expansion_ratio - batch.num_nodes / len(core_nodes))
            < 0.01
        )
        assert len(batch.batch_node_ids) == batch.num_nodes
        assert batch.x.shape[0] == batch.num_nodes
        if batch.edge_index.numel() > 0:
            assert batch.edge_index.max() < batch.num_nodes

    def test_expansion_ratio_enforcement_and_filtering(
        self, preprocessor_with_structures
    ):
        """Test max expansion ratio with structure filtering."""
        max_ratio = 1.3
        collate_fn = ExtendedContextCollate(
            preprocessor_with_structures,
            max_expansion_ratio=max_ratio,
            filter_on_expansion=True,
        )

        # Nodes from different triangles to test filtering
        core_nodes = [0, 1, 2, 3]
        batch = collate_fn([core_nodes])

        assert batch.expansion_ratio <= max_ratio + 0.1
        assert batch.num_structures > 0

        # Test filter_on_expansion=False allows exceeding ratio
        collate_no_filter = ExtendedContextCollate(
            preprocessor_with_structures,
            max_expansion_ratio=1.2,
            filter_on_expansion=False,
        )
        batch_no_filter = collate_no_filter([core_nodes])
        assert batch_no_filter.num_nodes >= len(core_nodes)


class TestSamplerIntegration:
    """Test integration with different samplers."""

    def test_multiple_samplers_and_batches(self, preprocessor_with_structures):
        """Test with random and cluster samplers across multiple batches."""
        collate_fn = ExtendedContextCollate(
            preprocessor_with_structures, max_expansion_ratio=1.5
        )

        # Test random sampler
        random_sampler = NodeBatchSampler(
            num_nodes=preprocessor_with_structures.num_nodes,
            batch_size=3,
            shuffle=False,
        )
        for core_nodes in random_sampler:
            batch = collate_fn([core_nodes])
            assert batch.num_nodes >= len(core_nodes)
            assert batch.core_mask.sum().item() == len(core_nodes)
            break

        # Test cluster sampler
        cluster_sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_with_structures.graph_data,
            batch_size=3,
            clustering_method="louvain",
            seed=42,
        )
        batch_count = 0
        for core_nodes in cluster_sampler:
            batch = collate_fn([core_nodes])
            assert batch.num_nodes > 0
            assert batch.expansion_ratio >= 1.0
            batch_count += 1
            if batch_count >= 2:
                break

        assert batch_count == 2


class TestDataloaderCreation:
    """Test convenience dataloader creation."""

    def test_dataloader_with_transform_and_length(
        self, preprocessor_with_structures
    ):
        """Test dataloader creation, transform application, and iteration."""
        from omegaconf import OmegaConf

        from topobench.transforms.data_transform import DataTransform

        transform = DataTransform(
            **OmegaConf.create(
                {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                }
            )
        )

        node_sampler = NodeBatchSampler(
            num_nodes=preprocessor_with_structures.num_nodes,
            batch_size=3,
            shuffle=False,
        )

        loader = create_extended_context_dataloader(
            preprocessor_with_structures,
            node_sampler=node_sampler,
            max_expansion_ratio=1.5,
            transform=transform,
        )

        assert len(loader) == len(node_sampler)
        batch = next(iter(loader))
        assert hasattr(batch, "x_0")
        assert hasattr(batch, "x_1")
        assert hasattr(batch, "incidence_1")


class TestEndToEndWorkflow:
    """Test complete workflows and edge cases."""

    def test_multi_epoch_training_loop(self, preprocessor_with_structures):
        """Test multi-epoch workflow with cluster-aware sampling."""
        sampler = ClusterAwareNodeSampler(
            graph_data=preprocessor_with_structures.graph_data,
            batch_size=3,
            clustering_method="louvain",
            seed=42,
        )

        loader = create_extended_context_dataloader(
            preprocessor_with_structures,
            node_sampler=sampler,
            max_expansion_ratio=1.5,
        )

        for _epoch in range(2):
            batch_count = 0
            for batch in loader:
                assert isinstance(batch, Data)
                assert batch.num_nodes > 0
                assert hasattr(batch, "core_mask")
                batch_count += 1

            assert batch_count > 0

    def test_structure_completeness_comparison(
        self, preprocessor_with_structures
    ):
        """Compare structure completeness with and without expansion."""
        collate_no_exp = ExtendedContextCollate(
            preprocessor_with_structures, max_expansion_ratio=1.0
        )
        collate_with_exp = ExtendedContextCollate(
            preprocessor_with_structures, max_expansion_ratio=2.0
        )

        core_nodes = [0, 1]
        batch_no_exp = collate_no_exp([core_nodes])
        batch_with_exp = collate_with_exp([core_nodes])

        assert batch_with_exp.num_structures >= batch_no_exp.num_structures

    def test_memory_efficiency_on_large_graph(self, larger_graph):
        """Test memory bounds on larger graph."""
        import gc
        import os

        import psutil

        data, preprocessor = larger_graph
        process = psutil.Process(os.getpid())
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        sampler = NodeBatchSampler(
            num_nodes=data.num_nodes, batch_size=50, shuffle=False
        )
        loader = create_extended_context_dataloader(
            preprocessor, node_sampler=sampler, max_expansion_ratio=1.5
        )

        for batch_idx, batch in enumerate(loader):
            assert batch.num_nodes <= 50 * 1.5 + 5
            if batch_idx >= 5:
                break

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - baseline_memory
        assert memory_increase < 200, (
            f"Memory increase too large: {memory_increase:.1f}MB"
        )


class TestEdgeCases:
    """Test boundary conditions."""

    def test_empty_single_and_full_batches(self, preprocessor_with_structures):
        """Test empty, single node, and all nodes batches."""
        collate_fn = ExtendedContextCollate(preprocessor_with_structures)

        # Empty batch
        batch = collate_fn([[]])
        assert batch.num_nodes == 0
        assert batch.core_mask.numel() == 0

        # Single node
        batch = collate_fn([[0]])
        assert batch.num_nodes >= 1
        assert batch.core_mask.sum().item() == 1

        # All nodes
        all_nodes = list(range(preprocessor_with_structures.num_nodes))
        batch = collate_fn([all_nodes])
        assert abs(batch.expansion_ratio - 1.0) < 0.1

    def test_disconnected_graph_components(self, tmp_path):
        """Test with disconnected components."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [3, 4, 4, 5, 5, 3]], dtype=torch.long
        )
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        data = Data(x=torch.randn(6, 8), edge_index=edge_index, num_nodes=6)

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=data,
            data_dir=tmp_path / "disconnected",
            max_clique_size=3,
        )
        preprocessor.build_index()

        collate_fn = ExtendedContextCollate(preprocessor)
        batch = collate_fn([[0, 1]])

        # Should not leak nodes from other component
        assert batch.num_nodes <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
