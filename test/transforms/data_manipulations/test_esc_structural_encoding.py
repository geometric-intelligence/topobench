from collections import Counter
from itertools import combinations, combinations_with_replacement, product
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset

import topobench.transforms.data_manipulations.esc_structural_encoding as esc
from topobench.data.preprocessor import PreProcessor
from topobench.data.utils.esc import (
    ESC_DISTANCE_OFFSET,
    ESC_EDGE_OFFSET,
    ESC_NUM_STRUCTURAL_CODES,
)
from topobench.dataloader import DataloadDataset, TBDataloader
from topobench.dataloader.utils import DomainData

EDGE_HISTOGRAM = (
    (1, 2),
    (300, 1),
    (301, 1),
    (303, 1),
    (304, 1),
    (318, 1),
)
PATH_OUTER_TO_CENTER = (
    (1, 2),
    (2, 1),
    (300, 1),
    (301, 1),
    (302, 1),
    (303, 1),
    (304, 2),
    (318, 1),
    (340, 1),
)
PATH_CENTER_TO_OUTER = (
    (1, 2),
    (2, 1),
    (300, 1),
    (301, 2),
    (303, 1),
    (304, 1),
    (305, 1),
    (318, 1),
    (320, 1),
)
TRIANGLE_HISTOGRAM = (
    (2, 3),
    (300, 1),
    (301, 2),
    (303, 1),
    (304, 2),
    (318, 1),
    (319, 1),
    (337, 1),
)
C4_HISTOGRAM = (
    (2, 4),
    (300, 1),
    (301, 2),
    (302, 1),
    (303, 1),
    (304, 2),
    (305, 1),
    (318, 1),
    (320, 1),
    (340, 1),
    (358, 1),
)
STAR_CENTER_TO_LEAF = (
    (1, 3),
    (3, 1),
    (300, 1),
    (301, 3),
    (303, 1),
    (304, 1),
    (305, 2),
    (318, 1),
    (320, 2),
)
STAR_LEAF_TO_CENTER = (
    (1, 3),
    (3, 1),
    (300, 1),
    (301, 1),
    (302, 2),
    (303, 1),
    (304, 3),
    (318, 1),
    (340, 2),
)

ROOTED_MOTIF_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 4),
    (1, 5),
    (4, 5),
    (0, 6),
    (0, 7),
    (1, 7),
    (6, 7),
    (0, 8),
    (1, 8),
    (0, 9),
    (1, 9),
    (8, 9),
)

# pin codes to paper. packing changes cannot rewrite oracle
ROOT_TO_COMMON_NEIGHBOR = 319  # (0, 1, 1, 1)
COMMON_NEIGHBOR_EDGE = 346  # (1, 1, 1, 1)
C4_EDGE_WEIGHTS = {
    347: 1,  # (1, 1, 1, 2)
    349: 1,  # (1, 1, 2, 1)
    358: 1,  # (1, 2, 2, 1)
    346: 2,  # label closes both rooted cycles
}


class _TinyDataset(InMemoryDataset):
    def __init__(self, data_list: list[Data]) -> None:
        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)


def _directed_data(
    num_nodes: int,
    directed_edges: list[tuple[int, int]],
    **attributes,
) -> Data:
    edge_index = (
        torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        if directed_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    attributes.setdefault(
        "x", torch.arange(num_nodes, dtype=torch.float32).view(-1, 1)
    )
    return Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        **attributes,
    )


def _undirected_data(
    num_nodes: int,
    edges: list[tuple[int, int]],
    **attributes,
) -> Data:
    directed_edges = [
        directed
        for source, target in edges
        for directed in ((source, target), (target, source))
    ]
    return _directed_data(num_nodes, directed_edges, **attributes)


def _rooted_motif_counts(
    edges: tuple[tuple[int, int], ...], root: tuple[int, int]
) -> tuple[int, int, int]:
    nodes = {node for edge in edges for node in edge}
    neighbors = {node: set() for node in nodes}
    for source, target in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)

    first, second = root
    common = neighbors[first] & neighbors[second]
    triangles = len(common)
    four_cliques = sum(
        target in neighbors[source]
        for source, target in combinations(common, 2)
    )
    four_cycles = sum(
        target in neighbors[source]
        for source in neighbors[first] - {second}
        for target in neighbors[second] - {first, source}
    )
    return triangles, four_cliques, four_cycles


def _edge_histograms(data: Data) -> list[tuple[tuple[int, int], ...]]:
    histograms = []
    offset = 0
    for nnz in data.esc_nnz_per_edge.tolist():
        end = offset + nnz
        histograms.append(
            tuple(
                zip(
                    data.esc_code_id[offset:end].tolist(),
                    [
                        int(count)
                        for count in data.esc_code_count[offset:end].tolist()
                    ],
                    strict=True,
                )
            )
        )
        offset = end
    assert offset == data.esc_code_id.numel()
    return histograms


def _assert_exact_histograms(
    data: Data,
    expected: list[tuple[tuple[int, int], ...]],
) -> None:
    assert data.esc_nnz_per_edge.tolist() == [
        len(histogram) for histogram in expected
    ]
    assert data.esc_code_id.tolist() == [
        code_id for histogram in expected for code_id, _ in histogram
    ]
    assert data.esc_code_count.tolist() == [
        float(count) for histogram in expected for _, count in histogram
    ]


def _histograms_by_edge(
    data: Data,
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    return {
        tuple(data.edge_index[:, column].tolist()): histogram
        for column, histogram in enumerate(_edge_histograms(data))
    }


def _transform_config():
    return OmegaConf.create(
        {
            "ESCStructuralEncoding": {
                "transform_name": "ESCStructuralEncoding",
                "transform_type": "data manipulation",
                "hop_radius": 1,
                "degree_bins": 300,
                "rooted_subgraph": "induced_union",
                "include_resistance_distance": False,
                "add_self_loops": False,
                "internal_edge_orientation": "lexicographic",
                "encoder_version": "esc-paper-induced-v1",
            }
        }
    )


@pytest.mark.parametrize(
    ("num_nodes", "edges", "expected"),
    [
        (2, [(0, 1)], [EDGE_HISTOGRAM, EDGE_HISTOGRAM]),
        (
            3,
            [(0, 1), (1, 2)],
            [
                PATH_OUTER_TO_CENTER,
                PATH_CENTER_TO_OUTER,
                PATH_CENTER_TO_OUTER,
                PATH_OUTER_TO_CENTER,
            ],
        ),
        (3, [(0, 1), (1, 2), (2, 0)], [TRIANGLE_HISTOGRAM] * 6),
        (
            4,
            [(0, 1), (0, 2), (0, 3)],
            [STAR_CENTER_TO_LEAF, STAR_LEAF_TO_CENTER] * 3,
        ),
    ],
    ids=["edge", "path", "triangle", "star"],
)
def test_exact_dense_and_sparse_histograms(num_nodes, edges, expected):
    output = esc.ESCStructuralEncoding()(_undirected_data(num_nodes, edges))

    _assert_exact_histograms(output, expected)


def test_degree_histogram_uses_the_rooted_subgraph():
    output = esc.ESCStructuralEncoding()(
        _undirected_data(4, [(0, 1), (1, 2), (2, 3)])
    )

    histogram = dict(_histograms_by_edge(output)[(0, 1)])
    assert histogram[1] == 2
    assert histogram[2] == 1


def test_induced_c4_retains_the_opposite_internal_edge():
    output = esc.ESCStructuralEncoding()(
        _undirected_data(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    )

    _assert_exact_histograms(output, [C4_HISTOGRAM] * 8)
    rooted_histogram = dict(_edge_histograms(output)[0])
    assert (
        sum(
            count
            for code_id, count in rooted_histogram.items()
            if code_id >= ESC_EDGE_OFFSET
        )
        == 4
    )
    assert rooted_histogram[358] == 1


def test_appendix_e_rooted_motif_identities():
    output = esc.ESCStructuralEncoding()(
        _undirected_data(10, list(ROOTED_MOTIF_EDGES))
    )
    histogram = dict(_histograms_by_edge(output)[(0, 1)])
    triangles, four_cliques, four_cycles = _rooted_motif_counts(
        ROOTED_MOTIF_EDGES, (0, 1)
    )

    assert histogram[ROOT_TO_COMMON_NEIGHBOR] == triangles
    assert histogram[COMMON_NEIGHBOR_EDGE] == four_cliques
    assert (
        sum(
            histogram.get(code_id, 0) * weight
            for code_id, weight in C4_EDGE_WEIGHTS.items()
        )
        == four_cycles
    )


def test_isolated_node_has_typed_empty_outputs():
    output = esc.ESCStructuralEncoding()(_undirected_data(1, []))

    assert output.esc_code_id.shape == (0,)
    assert output.esc_code_id.dtype == torch.long
    assert output.esc_code_count.shape == (0,)
    assert output.esc_code_count.dtype == torch.float32
    assert output.esc_nnz_per_edge.shape == (0,)
    assert output.esc_nnz_per_edge.dtype == torch.long


def test_maximum_legal_local_degree_is_encoded():
    output = esc.ESCStructuralEncoding()(
        _undirected_data(300, [(0, leaf) for leaf in range(1, 300)])
    )
    center_to_leaf = (
        (1, 299),
        (299, 1),
        (300, 1),
        (301, 299),
        (303, 1),
        (304, 1),
        (305, 298),
        (318, 1),
        (320, 298),
    )
    leaf_to_center = (
        (1, 299),
        (299, 1),
        (300, 1),
        (301, 1),
        (302, 298),
        (303, 1),
        (304, 299),
        (318, 1),
        (340, 298),
    )
    histograms = _edge_histograms(output)
    assert len(histograms) == 598
    assert Counter(histograms) == Counter(
        {center_to_leaf: 299, leaf_to_center: 299}
    )


def test_parameters_expose_cache_identity():
    transform = esc.ESCStructuralEncoding(
        transform_name="ESCStructuralEncoding",
        transform_type="data manipulation",
    )
    assert transform.parameters == {
        "hop_radius": 1,
        "degree_bins": 300,
        "rooted_subgraph": "induced_union",
        "include_resistance_distance": False,
        "add_self_loops": False,
        "internal_edge_orientation": "lexicographic",
        "encoder_version": "esc-paper-induced-v1",
        "transform_name": "ESCStructuralEncoding",
        "transform_type": "data manipulation",
    }


@pytest.mark.parametrize(
    ("num_nodes", "edge_index", "message"),
    [
        (2, torch.tensor([[0], [0]]), "self-loop"),
        (
            2,
            torch.tensor([[0, 0, 1], [1, 1, 0]]),
            "duplicate directed edge",
        ),
        (2, torch.tensor([[0], [1]]), "reciprocal"),
        (2, torch.tensor([[0, -1], [-1, 0]]), "outside"),
        (2, torch.empty((3, 0), dtype=torch.long), "shape"),
        (
            2,
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "torch.long",
        ),
    ],
    ids=[
        "loop",
        "duplicate",
        "missing-reciprocal",
        "negative-id",
        "shape",
        "dtype",
    ],
)
def test_invalid_graphs_are_rejected(num_nodes, edge_index, message):
    data = Data(
        x=torch.ones(num_nodes, 1),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    with pytest.raises(ValueError, match=message):
        esc.ESCStructuralEncoding()(data)


def test_degree_at_or_above_codebook_limit_is_rejected():
    data = _undirected_data(301, [(0, leaf) for leaf in range(1, 301)])
    with pytest.raises(ValueError, match="degree"):
        esc.ESCStructuralEncoding()(data)


class _TargetGuardData(Data):
    def __getattr__(self, key):
        if key == "y":
            raise AssertionError("ESC preprocessing must not read the target")
        return super().__getattr__(key)


def test_existing_fields_are_unchanged_and_target_is_not_read():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_attr = torch.tensor([[5.0], [6.0]])
    train_mask = torch.tensor([True, False])
    target = object()
    data = _TargetGuardData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_mask=train_mask,
        y=target,
        num_nodes=2,
    )
    original_keys = set(data.keys())
    original_tensors = {
        key: data._store[key].clone()
        for key in ("x", "edge_index", "edge_attr", "train_mask")
    }
    original_target = data._store["y"]

    output = esc.ESCStructuralEncoding()(data)

    for key, expected in original_tensors.items():
        assert torch.equal(output._store[key], expected)
    assert output._store["y"] is original_target
    assert set(output.keys()) == original_keys | {
        "esc_code_id",
        "esc_code_count",
        "esc_nnz_per_edge",
    }


def test_codebook_segments_and_edge_packing_are_collision_free():
    assert ESC_DISTANCE_OFFSET == 300
    assert ESC_EDGE_OFFSET == 306
    assert ESC_NUM_STRUCTURAL_CODES == 387

    labels = list(product(range(3), repeat=2))
    unordered_codes = [
        esc._edge_code(first, second)
        for first, second in combinations_with_replacement(labels, 2)
    ]
    assert len(unordered_codes) == len(set(unordered_codes))
    examples = {
        ((0, 0), (0, 0)): 306,
        ((0, 1), (1, 1)): 319,
        ((1, 1), (1, 1)): 346,
        ((2, 2), (2, 2)): 386,
    }
    assert {pair: esc._edge_code(*pair) for pair in examples} == examples
    assert all(
        esc._edge_code(first, second) == esc._edge_code(second, first)
        for first in labels
        for second in labels
    )


def test_node_relabeling_is_equivariant():
    directed_edges = [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ]
    original = esc.ESCStructuralEncoding()(_directed_data(5, directed_edges))
    permutation = [3, 0, 4, 1, 2]
    relabeled_edges = [
        (permutation[source], permutation[target])
        for source, target in reversed(directed_edges)
    ]
    relabeled_input = _directed_data(5, relabeled_edges)
    relabeled_edge_index = relabeled_input.edge_index.clone()
    relabeled = esc.ESCStructuralEncoding()(relabeled_input)
    expected = {
        (permutation[source], permutation[target]): histogram
        for (source, target), histogram in _histograms_by_edge(
            original
        ).items()
    }

    assert torch.equal(relabeled.edge_index, relabeled_edge_index)
    assert _histograms_by_edge(relabeled) == expected


def test_c6_and_two_triangles_have_different_raw_encodings():
    cycle = esc.ESCStructuralEncoding()(
        _undirected_data(6, [(node, (node + 1) % 6) for node in range(6)])
    )
    triangles = esc.ESCStructuralEncoding()(
        _undirected_data(
            6,
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3),
            ],
        )
    )

    assert cycle.edge_index.size(1) == triangles.edge_index.size(1) == 12
    assert Counter(_edge_histograms(cycle)) != Counter(
        _edge_histograms(triangles)
    )


def test_preprocessor_reload_and_tbdataloader_preserve_sparse_routing(
    tmp_path: Path,
):
    raw_graphs = [
        _undirected_data(3, [(0, 1), (1, 2)]),
        _undirected_data(3, [(0, 1), (1, 2), (2, 0)]),
        _undirected_data(2, []),
    ]
    expected_graphs = [
        esc.ESCStructuralEncoding()(graph.clone()) for graph in raw_graphs
    ]
    data_dir = tmp_path / "esc-cache"
    processed = PreProcessor(
        _TinyDataset(raw_graphs), str(data_dir), _transform_config()
    )
    reloaded = PreProcessor(
        _TinyDataset(raw_graphs), str(data_dir), _transform_config()
    )
    graphs = reloaded.data_list

    assert Path(processed.processed_paths[0]).is_file()
    assert processed.processed_paths == reloaded.processed_paths
    for graph, expected in zip(graphs, expected_graphs, strict=True):
        assert torch.equal(graph.edge_index, expected.edge_index)
        assert torch.equal(graph.esc_code_id, expected.esc_code_id)
        assert torch.equal(graph.esc_code_count, expected.esc_code_count)
        assert torch.equal(graph.esc_nnz_per_edge, expected.esc_nnz_per_edge)

    dataset = DataloadDataset(graphs)
    dataloader = TBDataloader(
        dataset_train=dataset,
        dataset_val=dataset,
        dataset_test=dataset,
        batch_size=len(graphs),
        num_workers=0,
    )
    batch = next(iter(dataloader.val_dataloader()))

    assert isinstance(batch, DomainData)
    assert torch.equal(
        batch.esc_code_id,
        torch.cat([graph.esc_code_id for graph in expected_graphs]),
    )
    assert torch.equal(
        batch.esc_code_count,
        torch.cat([graph.esc_code_count for graph in expected_graphs]),
    )
    assert torch.equal(
        batch.esc_nnz_per_edge,
        torch.cat([graph.esc_nnz_per_edge for graph in expected_graphs]),
    )
    assert batch.esc_nnz_per_edge.numel() == batch.edge_index.size(1)
    assert int(batch.esc_nnz_per_edge.sum()) == batch.esc_code_id.numel()
