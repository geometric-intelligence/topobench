"""Tests for OnDiskTransductivePreprocessor.

This test suite covers:
- Initialization and configuration
- Index building and persistence
- Structure querying operations
- Subgraph extraction with transforms
- High-level API (load_dataset_splits)
- PyTorch Dataset interface
- Edge cases and robustness
- Resource management
"""

import networkx as nx
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from topobench.data.preprocessor.ondisk_transductive import (
    OnDiskTransductivePreprocessor,
)
from topobench.dataloader import TBDataloader


@pytest.fixture
def small_graph_data():
    """Small PyG Data object for quick tests.

    Creates a 10-node graph with:
    - Triangle structures
    - Train/val/test masks
    - Node features and labels
    """
    # Create a graph with triangles: 0-1-2-0, 3-4-5-3, plus connections
    edge_index = torch.tensor( # weird linter rule...
        [
            [
                0,
                1,
                1,
                2,
                2,
                0,
                3,
                4,
                4,
                5,
                5,
                3,
                2,
                3,  # Connect triangles
                6,
                7,
                8,
                9,
            ],  # Chain
            [
                1,
                0,
                2,
                1,
                0,
                2,
                4,
                3,
                5,
                4,
                3,
                5,
                3,
                2,  # Connect triangles
                7,
                8,
                9,
                6,
            ],  # Chain
        ],
        dtype=torch.long,
    )

    x = torch.randn(10, 8)
    y = torch.randint(0, 3, (10,))

    # Create splits
    train_mask = torch.tensor([True] * 6 + [False] * 4)
    val_mask = torch.tensor([False] * 6 + [True] * 2 + [False] * 2)
    test_mask = torch.tensor([False] * 8 + [True] * 2)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=10,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


@pytest.fixture
def karate_graph_data():
    """Real Karate Club graph for integration tests."""
    G = nx.karate_club_graph()

    # Convert to PyG edge_index
    edges = list(G.edges())
    edge_index = torch.tensor(
        [
            [e[0] for e in edges] + [e[1] for e in edges],
            [e[1] for e in edges] + [e[0] for e in edges],
        ],
        dtype=torch.long,
    )

    num_nodes = G.number_of_nodes()
    x = torch.randn(num_nodes, 8)

    # Convert club labels (strings) to integers
    club_map = {"Mr. Hi": 0, "Officer": 1}
    y = torch.tensor(
        [
            club_map.get(G.nodes[i].get("club", "Mr. Hi"), 0)
            for i in range(num_nodes)
        ]
    )

    # Create splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[:20] = True
    val_mask[20:27] = True
    test_mask[27:] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


@pytest.fixture
def temp_index_dir(tmp_path):
    """Temporary directory for index files."""
    return tmp_path / "transductive_index"


@pytest.fixture
def preprocessor(small_graph_data, temp_index_dir):
    """Basic preprocessor instance without index built."""
    return OnDiskTransductivePreprocessor(
        graph_data=small_graph_data,
        data_dir=temp_index_dir,
        max_structure_size=3,
    )


@pytest.fixture
def preprocessor_with_index(preprocessor):
    """Preprocessor with index already built."""
    preprocessor.build_index()
    return preprocessor


class TestTransductivePreprocessorBasics:
    """Test basic functionality of OnDiskTransductivePreprocessor."""

    def test_build_index(self, preprocessor, temp_index_dir):
        """Verify build_index creates necessary files and sets state."""
        preprocessor.build_index()

        assert preprocessor._index_built
        assert preprocessor.num_structures > 0
        assert temp_index_dir.exists()
        # Check SQLite database exists
        assert any(temp_index_dir.glob("*.db"))

    def test_build_index_persistence(self, small_graph_data, temp_index_dir):
        """Verify index persists and can be reloaded."""
        # Build index
        prep1 = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )
        prep1.build_index()
        num_structures = prep1.num_structures
        prep1.close()

        # Reload from disk
        prep2 = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )
        prep2.build_index()
        assert prep2.num_structures == num_structures
        prep2.close()

    def test_get_stats(self, preprocessor):
        """Verify get_stats returns correct information."""
        stats_before = preprocessor.get_stats()
        assert stats_before["num_nodes"] == 10
        assert stats_before["num_edges"] > 0
        assert stats_before["num_structures"] == 0  # Before index build

        preprocessor.build_index()
        stats_after = preprocessor.get_stats()
        assert stats_after["num_nodes"] == 10
        assert stats_after["num_structures"] > 0


class TestStructureQuerying:
    """Test structure query operations."""

    def test_query_batch_modes(self, preprocessor_with_index):
        """Verify fully contained vs partial overlap query modes."""
        batch_nodes = [0, 1, 2]

        # Fully contained: only complete structures
        fully_contained = preprocessor_with_index.query_batch(
            batch_nodes, fully_contained=True
        )
        assert isinstance(fully_contained, list)
        assert len(fully_contained) == 1  # One triangle structure: 0-1-2

        # Check structure format: (structure_id, [node_ids])
        for struct_id, node_ids in fully_contained:
            assert isinstance(struct_id, int)
            assert isinstance(node_ids, list)
            assert all(node in batch_nodes for node in node_ids)

        # Partial overlap: includes touching structures
        partial_overlap = preprocessor_with_index.query_batch(
            batch_nodes, fully_contained=False
        )
        assert len(partial_overlap) >= len(fully_contained)

    def test_query_edge_cases(self, preprocessor_with_index):
        """Verify query handles edge cases correctly."""
        # Empty batch
        assert (
            preprocessor_with_index.query_batch([], fully_contained=True) == []
        )

        # Single node
        structures = preprocessor_with_index.query_batch(
            [0], fully_contained=False
        )
        assert isinstance(structures, list)

        # Invalid nodes
        assert (
            preprocessor_with_index.query_batch([999], fully_contained=True)
            == []
        )


class TestSubgraphExtraction:
    """Test subgraph creation and extraction."""

    def test_get_subgraph(self, preprocessor_with_index):
        """Verify get_subgraph returns valid PyG Data with correct structure."""
        # Test basic subgraph extraction
        subgraph = preprocessor_with_index.get_subgraph([0, 1, 2])

        assert isinstance(subgraph, Data)
        assert subgraph.num_nodes == 3
        assert hasattr(subgraph, "x")
        assert hasattr(subgraph, "edge_index")

        # Test non-contiguous nodes (node mapping)
        subgraph2 = preprocessor_with_index.get_subgraph([2, 5, 8])
        assert subgraph2.num_nodes == 3
        # Edge indices should use local IDs
        if subgraph2.edge_index.numel() > 0:
            assert subgraph2.edge_index.max() < 3


class TestHighLevelSplitAPI:
    """Test load_dataset_splits() integration."""

    def test_load_dataset_splits_structure_centric(
        self, small_graph_data, temp_index_dir
    ):
        """Verify load_dataset_splits returns correct dataset objects."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 10,
                "node_budget": 100,
            }
        )

        train, val, test = preprocessor.load_dataset_splits(split_config)

        # Check types
        from topobench.data.datasets.transductive_split import (
            TransductiveSplitDataset,
        )

        assert isinstance(train, TransductiveSplitDataset)
        assert isinstance(val, TransductiveSplitDataset)
        assert isinstance(test, TransductiveSplitDataset)

        # Check index was built
        assert preprocessor._index_built

        preprocessor.close()

    def test_load_dataset_splits_extended_context(
        self, small_graph_data, temp_index_dir
    ):
        """Verify extended context strategy works."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        split_config = OmegaConf.create(
            {
                "strategy": "extended_context",
                "nodes_per_batch": 5,
                "max_expansion_ratio": 1.5,
                "sampler_method": "louvain",
            }
        )

        train, val, test = preprocessor.load_dataset_splits(split_config)

        from topobench.data.datasets.transductive_split import (
            TransductiveSplitDataset,
        )

        assert all(
            isinstance(ds, TransductiveSplitDataset)
            for ds in [train, val, test]
        )

        preprocessor.close()

    def test_load_dataset_splits_no_masks(self, temp_index_dir):
        """Verify works without train/val/test masks."""
        # Create triangle graph without masks
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 0],
                [1, 0, 2, 1, 0, 2],
            ],
            dtype=torch.long,
        )
        graph_data = Data(
            x=torch.randn(3, 8), edge_index=edge_index, num_nodes=3
        )

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 10,
            }
        )

        train, val, test = preprocessor.load_dataset_splits(split_config)

        # Should not raise error
        assert train is not None
        assert val is not None
        assert test is not None

        preprocessor.close()

    def test_load_dataset_splits_builds_index_automatically(
        self, small_graph_data, temp_index_dir
    ):
        """Verify index is built automatically on first call."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        assert not preprocessor._index_built

        split_config = OmegaConf.create({"strategy": "structure_centric"})
        _ = preprocessor.load_dataset_splits(split_config)

        assert preprocessor._index_built
        assert preprocessor.num_structures > 0

        preprocessor.close()

    def test_dataset_splits_with_tbdataloader(
        self, small_graph_data, temp_index_dir
    ):
        """Verify splits work with TBDataloader."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 5,
            }
        )

        train, val, test = preprocessor.load_dataset_splits(split_config)

        # Create datamodule (batch_size=1 for transductive)
        datamodule = TBDataloader(train, val, test, batch_size=1)

        # Get loaders
        train_loader = datamodule.train_dataloader()
        _val_loader = datamodule.val_dataloader()
        _test_loader = datamodule.test_dataloader()

        # Verify we can iterate (just check first batch)
        train_batch = next(iter(train_loader))
        assert isinstance(train_batch, Data)

        preprocessor.close()


class TestDatasetInterface:
    """Test PyTorch Dataset protocol compliance."""

    def test_dataset_interface(self, preprocessor_with_index):
        """Verify Dataset protocol (__len__, __getitem__)."""
        # Test __len__
        assert len(preprocessor_with_index) == 10

        # Test __getitem__ single node
        subgraph = preprocessor_with_index[0]
        assert isinstance(subgraph, Data)

        # Test __getitem__ multiple nodes
        subgraph = preprocessor_with_index[[0, 1, 2]]
        assert isinstance(subgraph, Data)
        assert subgraph.num_nodes >= 3

        # Test out of bounds
        with pytest.raises((IndexError, ValueError)):
            _ = preprocessor_with_index[999]


class TestEdgeCases:
    """Test corner cases and robustness."""

    def test_graph_topology_variations(self, temp_index_dir):
        """Verify handling of various graph topologies."""
        # Test 1: Disconnected components
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
                [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5],
            ],
            dtype=torch.long,
        )
        graph_data = Data(
            x=torch.randn(6, 8), edge_index=edge_index, num_nodes=6
        )

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=temp_index_dir / "disconnected",
            max_structure_size=3,
        )
        preprocessor.build_index()
        assert preprocessor.num_structures > 0

        # Query should not cross components
        structures = preprocessor.query_batch([0, 1, 2], fully_contained=False)
        for _, nodes in structures:
            assert all(n < 3 for n in nodes)
        preprocessor.close()

        # Test 2: Isolated nodes
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        graph_data = Data(
            x=torch.randn(4, 8), edge_index=edge_index, num_nodes=4
        )

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=temp_index_dir / "isolated",
            max_structure_size=3,
        )
        preprocessor.build_index()
        assert preprocessor.query_batch([3], fully_contained=True) == []
        preprocessor.close()

        # Test 3: Complete graph (dense)
        nodes = list(range(5))
        edges = [(i, j) for i in nodes for j in nodes if i < j]
        edge_index = torch.tensor(
            [
                [e[0] for e in edges] + [e[1] for e in edges],
                [e[1] for e in edges] + [e[0] for e in edges],
            ],
            dtype=torch.long,
        )
        graph_data = Data(
            x=torch.randn(5, 8), edge_index=edge_index, num_nodes=5
        )

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=graph_data,
            data_dir=temp_index_dir / "complete",
            max_structure_size=3,
        )
        preprocessor.build_index()
        assert preprocessor.num_structures == 10  # K5 has 10 triangles
        preprocessor.close()


class TestTrainingUsage:
    """Test how preprocessor is actually used in training."""

    def test_complete_training_workflow(
        self, karate_graph_data, temp_index_dir
    ):
        """Verify complete training workflow: preprocessor → splits → mini-batches."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=karate_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )
        preprocessor.build_index()

        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 10,
                "node_budget": 50,
            }
        )
        train, val, test = preprocessor.load_dataset_splits(split_config)

        # Verify splits created correctly
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # Verify iteration through dataset works (mini-batching)
        for batch_idx in range(min(3, len(train))):
            batch = train[batch_idx]
            assert isinstance(batch, Data)
            assert hasattr(batch, "x")
            assert hasattr(batch, "edge_index")
            assert batch.num_nodes > 0

        preprocessor.close()

    def test_automatic_batching_with_datamodule(
        self, karate_graph_data, temp_index_dir
    ):
        """Verify automatic mini-batching through TBDataloader.

        Tests the complete flow: preprocessor → splits → TBDataloader → batches.
        This is how training with trainer.fit(model, datamodule) works.
        """
        from topobench.dataloader import TBDataloader

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=karate_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 10,
                "node_budget": 50,
            }
        )
        train, val, test = preprocessor.load_dataset_splits(split_config)

        datamodule = TBDataloader(
            dataset_train=train,
            dataset_val=val,
            dataset_test=test,
            batch_size=1,
            num_workers=0,
        )

        train_loader = datamodule.train_dataloader()
        assert len(train_loader) > 0

        # Verify iteration through dataloader works
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:
                break
            assert isinstance(batch, Data)
            assert batch.num_nodes > 0

        preprocessor.close()


class TestLiftings:
    """Test lifting compatibility with on-disk preprocessor."""

    def test_manual_lifting_on_subgraphs(self, preprocessor_with_index):
        """Verify manual lifting works on subgraphs."""
        from topobench.transforms.liftings.graph2simplicial.clique_lifting import (
            SimplicialCliqueLifting,
        )

        batch = preprocessor_with_index.get_subgraph([0, 1, 2, 3, 4, 5])

        lifting = SimplicialCliqueLifting(complex_dim=2)
        lifted = lifting.forward(batch)

        assert hasattr(lifted, "incidence_1")
        assert hasattr(lifted, "x_1")
        assert lifted.num_nodes == 6

    def test_preprocessor_with_transforms_config(
        self, small_graph_data, temp_index_dir
    ):
        """Verify preprocessor accepts transforms_config."""
        transforms_config = OmegaConf.create(
            {
                "clique_lifting": {
                    "transform_type": "lifting",
                    "transform_name": "SimplicialCliqueLifting",
                    "complex_dim": 2,
                }
            }
        )

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=small_graph_data,
            data_dir=temp_index_dir,
            transforms_config=transforms_config,
            max_structure_size=3,
        )
        preprocessor.build_index()

        batch = preprocessor.get_subgraph([0, 1, 2])
        assert isinstance(batch, Data)
        assert batch.num_nodes == 3

        preprocessor.close()


class TestMemoryEfficiency:
    """Test memory efficiency of on-disk approach.

    To experiment with different graph sizes, modify NUM_NODES below.
    Larger graphs will stress-test memory efficiency more thoroughly.
    """

    @pytest.fixture
    def memory_test_graph(self):
        """Configurable graph for memory testing.

        Modify NUM_NODES to test different graph sizes (34, 100, 150, 200, 500+).
        Generates graphs with community structure for realistic triangle density.
        """
        NUM_NODES = 50

        if NUM_NODES == 34:
            G = nx.karate_club_graph()
        else:
            num_communities = max(3, NUM_NODES // 20)
            G = nx.gaussian_random_partition_graph(
                n=NUM_NODES,
                s=NUM_NODES // num_communities,
                v=2,
                p_in=0.7,
                p_out=0.1,
            )

        # Convert to PyG
        edges = list(G.edges())
        edge_index = torch.tensor(
            [
                [e[0] for e in edges] + [e[1] for e in edges],
                [e[1] for e in edges] + [e[0] for e in edges],
            ],
            dtype=torch.long,
        )

        num_nodes = G.number_of_nodes()
        x = torch.randn(num_nodes, 16)
        y = torch.randint(0, 3, (num_nodes,))

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)

        train_mask[:train_size] = True
        val_mask[train_size : train_size + val_size] = True
        test_mask[train_size + val_size :] = True

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    def test_memory_efficiency_during_workflow(
        self, memory_test_graph, temp_index_dir
    ):
        """Verify on-disk preprocessing keeps memory bounded.

        Tests complete workflow: index build, queries, splits, and iteration.
        Key validation: memory should NOT grow linearly with graph size.
        """
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        gc.collect()

        baseline_memory = process.memory_info().rss / 1024 / 1024

        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=memory_test_graph,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )
        preprocessor.build_index()

        after_index_memory = process.memory_info().rss / 1024 / 1024

        # Test query operations
        num_queries = min(10, memory_test_graph.num_nodes // 5)
        for i in range(num_queries):
            query_nodes = list(
                range(i * 5, min((i + 1) * 10, memory_test_graph.num_nodes))
            )
            _ = preprocessor.query_batch(query_nodes, fully_contained=True)

        after_queries_memory = process.memory_info().rss / 1024 / 1024

        # Test split creation and batch iteration
        split_config = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 20,
                "node_budget": 100,
            }
        )
        train, _val, _test = preprocessor.load_dataset_splits(split_config)

        num_batches = min(5, len(train))
        for i in range(num_batches):
            batch = train[i]
            assert isinstance(batch, Data)

        final_memory = process.memory_info().rss / 1024 / 1024

        preprocessor.close()

        # Calculate memory metrics
        memory_increase = final_memory - baseline_memory
        peak_increase = max(
            after_index_memory - baseline_memory,
            after_queries_memory - baseline_memory,
            final_memory - baseline_memory,
        )

        # Assertions: Memory should stay bounded (scale thresholds with graph size)
        num_nodes = memory_test_graph.num_nodes
        size_factor = num_nodes / 34  # Karate (34 nodes) is baseline
        max_total_increase = 100 * size_factor
        max_peak_increase = 150 * size_factor

        assert memory_increase < max_total_increase, (
            f"Total memory increase too large: {memory_increase:.2f}MB (max: {max_total_increase:.2f}MB)"
        )
        assert peak_increase < max_peak_increase, (
            f"Peak memory increase too high: {peak_increase:.2f}MB (max: {max_peak_increase:.2f}MB)"
        )

        # KEY validation: Memory per node should be minimal (proves sub-linear scaling)
        memory_per_node = memory_increase / num_nodes
        assert memory_per_node < 5.0, (
            f"Memory per node too high: {memory_per_node:.2f}MB/node (max: 5MB/node)"
        )


class TestIntegrationWithLoaders:
    """Test integration with transductive loaders."""

    def test_loader_integration_both_strategies(
        self, karate_graph_data, temp_index_dir
    ):
        """Verify both loader strategies work end-to-end."""
        preprocessor = OnDiskTransductivePreprocessor(
            graph_data=karate_graph_data,
            data_dir=temp_index_dir,
            max_structure_size=3,
        )

        # Test structure-centric
        config_sc = OmegaConf.create(
            {
                "strategy": "structure_centric",
                "structures_per_batch": 20,
                "node_budget": 100,
            }
        )
        train_sc, _, _ = preprocessor.load_dataset_splits(config_sc)
        assert len(train_sc) > 0
        assert isinstance(train_sc[0], Data)

        # Test extended context
        config_ec = OmegaConf.create(
            {
                "strategy": "extended_context",
                "nodes_per_batch": 10,
                "max_expansion_ratio": 1.5,
                "sampler_method": "louvain",
            }
        )
        train_ec, _, _ = preprocessor.load_dataset_splits(config_ec)
        assert len(train_ec) > 0
        assert isinstance(train_ec[0], Data)

        preprocessor.close()
