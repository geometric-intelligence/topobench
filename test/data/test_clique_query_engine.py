"""Streamlined tests for CliqueQueryEngine and SQLite backend.

This test suite validates core functionality with minimal redundancy,
testing multiple aspects in combined scenarios to reduce test count
while maintaining >90% coverage.
"""

import pytest
import networkx as nx

from topobench.data.index import SQLiteIndexBackend
from topobench.data.clique_query import CliqueQueryEngine


def verify_query_correctness(
    query_engine: CliqueQueryEngine, batch_nodes: list[int]
) -> dict:
    """Verify query correctness against in-memory baseline.
    
    Helper function for testing - compares query engine results
    against naive enumeration.
    """
    from topobench.data.clique_detection import enumerate_k_cliques_streaming
    
    # Get results from query engine
    engine_results = query_engine.query_batch(batch_nodes, fully_contained=True)
    engine_cliques = {tuple(sorted(nodes)) for _, nodes in engine_results}
    
    # Baseline: enumerate and filter
    batch_set = set(batch_nodes)
    baseline_cliques = set()
    
    clique_iter = enumerate_k_cliques_streaming(
        query_engine.graph, k=query_engine.max_clique_size
    )
    
    for _, nodes in clique_iter:
        if set(nodes).issubset(batch_set):
            baseline_cliques.add(tuple(sorted(nodes)))
    
    missing = baseline_cliques - engine_cliques
    extra = engine_cliques - baseline_cliques
    
    return {
        "correct": len(missing) == 0 and len(extra) == 0,
        "engine_count": len(engine_cliques),
        "baseline_count": len(baseline_cliques),
        "missing": list(missing),
        "extra": list(extra),
    }


@pytest.fixture
def karate_graph():
    """Karate Club graph for testing."""
    return nx.karate_club_graph()


@pytest.fixture
def triangle_graph():
    """Simple graph with 3 connected triangles."""
    G = nx.Graph()
    edges = [
        (0, 1), (1, 2), (2, 0),  # Triangle 1
        (2, 3), (3, 4), (4, 2),  # Triangle 2
        (4, 5), (5, 6), (6, 4),  # Triangle 3
    ]
    G.add_edges_from(edges)
    return G


class TestSQLiteBackendIntegrated:
    """Test SQLite backend through query engine integration."""
    
    def test_backend_full_workflow(self, triangle_graph, tmp_path):
        """Test complete backend workflow: init, insert, query, retrieve."""
        backend = SQLiteIndexBackend(data_dir=str(tmp_path / "test_db"))
        backend.open()
        
        # Test insertion (single and batch)
        backend.insert(clique_id=0, nodes=[1, 2, 3])
        assert backend.count_cliques() == 1
        
        backend.insert_batch(iter([(1, [2, 3, 4]), (2, [5, 6, 7])]))
        assert backend.count_cliques() == 3
        
        # Test queries (fully contained and partial)
        results_full = backend.query_by_nodes([1, 2, 3, 4], fully_contained=True)
        assert len(results_full) == 2  # Structures 0 and 1
        
        results_partial = backend.query_by_nodes([1, 2], fully_contained=False)
        assert len(results_partial) == 2  # Structures with 1 or 2
        
        # Test get_clique and exists
        assert backend.get_clique(0) == [1, 2, 3]
        assert backend.exists()
        
        # Test clear
        backend.clear()
        assert backend.count_cliques() == 0
        
        backend.close()


class TestCliqueQueryEngine:
    """Test CliqueQueryEngine core functionality."""
    
    def test_complete_workflow_with_persistence(self, karate_graph, tmp_path):
        """Test full workflow: build, query, persist, reload."""
        index_dir = tmp_path / "karate_index"
        
        # Initial build
        engine1 = CliqueQueryEngine(karate_graph, index_dir=index_dir, max_clique_size=3)
        engine1.open()
        engine1.build_index()
        
        num_cliques = engine1.num_cliques
        assert num_cliques > 0
        
        # Test queries
        batch_nodes = [0, 1, 2, 3, 4]
        cliques = engine1.query_batch(batch_nodes, fully_contained=True)
        assert len(cliques) > 0
        
        # Test query_by_id
        if len(cliques) > 0:
            clique_ids = [sid for sid, _ in cliques[:2]]
            retrieved = engine1.query_cliques_by_id(clique_ids)
            assert len(retrieved) == len(clique_ids)
        
        # Test stats
        stats = engine1.get_stats()
        assert stats["num_cliques"] == num_cliques
        assert stats["num_nodes"] == 34
        
        engine1.close()
        
        # Test persistence - reload
        engine2 = CliqueQueryEngine(karate_graph, index_dir=index_dir, max_clique_size=3)
        engine2.open()
        engine2.build_index()  # Should load existing
        assert engine2.num_cliques == num_cliques
        engine2.close()
    
    def test_query_correctness_validation(self, triangle_graph, tmp_path):
        """Test query correctness against baseline enumeration."""
        engine = CliqueQueryEngine(triangle_graph, index_dir=tmp_path / "tri", max_clique_size=3)
        engine.open()
        engine.build_index()
        
        # Verify correctness for various batch sizes
        test_cases = [
            [0, 1, 2],  # One triangle
            [0, 1, 2, 3, 4],  # Two triangles
            list(range(7)),  # All triangles
        ]
        
        for batch_nodes in test_cases:
            result = verify_query_correctness(engine, batch_nodes)
            assert result["correct"], f"Verification failed for batch {batch_nodes}: {result}"
        
        engine.close()
    
    def test_edge_cases(self, tmp_path):
        """Test edge cases: empty graph, no triangles, complete graph."""
        # Empty graph
        G_empty = nx.Graph()
        engine_empty = CliqueQueryEngine(G_empty, index_dir=tmp_path / "empty", max_clique_size=3)
        engine_empty.open()
        engine_empty.build_index()
        assert engine_empty.num_cliques == 0
        engine_empty.close()
        
        # Tree (no triangles)
        G_tree = nx.path_graph(10)
        engine_tree = CliqueQueryEngine(G_tree, index_dir=tmp_path / "tree", max_clique_size=3)
        engine_tree.open()
        engine_tree.build_index()
        assert engine_tree.num_cliques == 0
        engine_tree.close()
        
        # Complete graph K5 has C(5,3) = 10 triangles
        G_complete = nx.complete_graph(5)
        engine_complete = CliqueQueryEngine(G_complete, index_dir=tmp_path / "complete", max_clique_size=3)
        engine_complete.open()
        engine_complete.build_index()
        assert engine_complete.num_cliques == 10
        
        # Test querying all nodes returns all triangles
        all_cliques = engine_complete.query_batch(list(range(5)), fully_contained=True)
        assert len(all_cliques) == 10
        engine_complete.close()
    
    def test_different_clique_sizes(self, karate_graph, tmp_path):
        """Test with different max_clique_size values."""
        # 3-cliques
        engine3 = CliqueQueryEngine(karate_graph, index_dir=tmp_path / "tri", max_clique_size=3)
        engine3.open()
        engine3.build_index()
        num_3cliques = engine3.num_cliques
        engine3.close()
        
        # 4-cliques
        engine4 = CliqueQueryEngine(karate_graph, index_dir=tmp_path / "quad", max_clique_size=4)
        engine4.open()
        engine4.build_index()
        num_4cliques = engine4.num_cliques
        engine4.close()
        
        # Should have more triangles than 4-cliques
        assert num_3cliques > num_4cliques
    
    def test_performance_and_context_manager(self, karate_graph, tmp_path):
        """Test performance and context manager usage."""
        import time
        
        # Test context manager
        with CliqueQueryEngine(karate_graph, index_dir=tmp_path / "ctx", max_clique_size=3) as engine:
            start = time.time()
            engine.build_index()
            build_time = time.time() - start
            
            assert build_time < 2.0  # Should be fast for small graph
            
            # Test query performance
            start = time.time()
            for _ in range(100):
                engine.query_batch(list(range(10)), fully_contained=True)
            query_time = time.time() - start
            
            assert query_time < 0.5  # 100 queries should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
