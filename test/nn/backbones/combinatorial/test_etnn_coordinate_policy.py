"""Tests for the consolidated coordinate-policy ETNN backbone.

The coordinate-policy ETNN exposes one TopoBench backbone that runs without
coordinates, with LapPE structural pseudo-coordinates, or with physical
Euclidean coordinates.  These tests protect both model wiring and the geometry
contracts behind the physical policy.

The test file is organized around the coordinate-policy model's public
behavior and geometry contracts:

1. Does the unified backbone preserve the coordinate-free GraphUniverse path?
2. Does the structural LapPE policy consume graph-derived pseudo-coordinates
   without requiring physical ``pos``?
3. Does the physical policy require real rank-0 coordinates and transform them
   into E(n)-invariant relation features?
4. Are physical centroids and diameters computed from incident rank-0 vertices,
   rather than from arbitrary higher-rank feature tensors?
5. Does physical mode update rank-0 coordinates equivariantly when
   ``pos_update=True``?
6. Does the helper code respect TopoBench's sparse-aware batching contract?
"""

import math

import pytest
import torch
from torch_geometric.data import Data

import topobench.nn.backbones.combinatorial.etnn_coordinate_policy as etnn_policy_module
from topobench.data.utils.utils import load_manual_graph_second_structure
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.combinatorial.etnn_coordinate_policy import (
    ETNNCoordinatePolicy,
    _batch_norm_physical_invariants,
    _build_lappe_cell_coordinates,
    _build_physical_cell_geometry,
    _build_vertex_memberships,
    _edge_channels_for_coordinate_policy,
    _ETNNCoordinatePolicyLayer,
    _neighborhood_to_edge_index,
    _normalize_physical_invariants,
    _physical_relation_invariants,
    _squared_coordinate_distances,
    _validate_physical_coordinates,
)
from topobench.nn.wrappers.combinatorial import TuneWrapper
from topobench.transforms.liftings.graph2combinatorial.graph_induced_cc import (
    GraphTriangleInducedCC,
)


def create_mock_complex_batch():
    """Create a standalone lifted combinatorial-complex test batch.

    This fixture directly describes the rank-wise feature and
    sparse-neighborhood contract produced by TopoBench's
    graph-to-combinatorial lifting.

    Returns
    -------
    torch_geometric.data.Data
        Three-rank complex with node, edge, and face features plus every
        relation consumed by the public coordinate-policy configs.
    """
    x_0 = torch.randn(4, 16)
    x_1 = torch.randn(4, 16)
    x_2 = torch.randn(2, 16)

    # Canonical node-edge and edge-face incidence matrices.  Their transposes
    # supply the reverse message directions used by the ETNN neighborhood set.
    incidence_1 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 0, 3],
                [0, 0, 1, 1, 2, 2, 3, 3],
            ]
        ),
        values=torch.ones(8),
        size=(4, 4),
    ).coalesce()
    incidence_2 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 2, 1, 2, 3],
                [0, 0, 0, 1, 1, 1],
            ]
        ),
        values=torch.ones(6),
        size=(4, 2),
    ).coalesce()

    # Same-rank adjacencies exercise feature updates at all visible ranks.
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        values=torch.ones(6),
        size=(4, 4),
    ).coalesce()
    adjacency_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        values=torch.ones(6),
        size=(4, 4),
    ).coalesce()
    adjacency_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.ones(2),
        size=(2, 2),
    ).coalesce()

    return Data(
        x_0=x_0,
        x_1=x_1,
        x_2=x_2,
        y=torch.tensor([1]),
        batch_0=torch.zeros(x_0.shape[0], dtype=torch.long),
        batch_1=torch.zeros(x_1.shape[0], dtype=torch.long),
        batch_2=torch.zeros(x_2.shape[0], dtype=torch.long),
        **{
            "up_adjacency-0": adjacency_0,
            "up_adjacency-1": adjacency_1,
            "up_adjacency-2": adjacency_2,
            "up_incidence-0": incidence_1.T.coalesce(),
            "down_incidence-1": incidence_1,
            "up_incidence-1": incidence_2.T.coalesce(),
            "down_incidence-2": incidence_2,
        },
    )


def create_lappe_complex_batch():
    """Add coordinate-lifting incidences to the standalone mock complex.

    Returns
    -------
    torch_geometric.data.Data
        Mock complex whose canonical ``incidence_r`` tensors support recursive
        LapPE coordinate averaging from rank 0 through rank 2.
    """
    batch = create_mock_complex_batch()
    batch.incidence_1 = batch["down_incidence-1"]
    batch.incidence_2 = batch["down_incidence-2"]
    return batch


def _collate_topobench_data_list(data_list):
    """Collate lifted data with TopoBench's sparse-aware collate contract.

    Raw PyG batching is not sufficient for combinatorial sparse incidence
    tensors because incidence matrices need block-diagonal concatenation.  The
    production TopoBench dataloader uses ``collate_fn`` from
    ``topobench.dataloader.utils``; this helper mirrors that path for tests.
    """
    collate_items = []
    for data in data_list:
        keys = list(data.keys())
        values = [data[key] for key in keys]
        collate_items.append((values, keys))
    return collate_fn(collate_items)


def create_coordinate_policy_etnn(coordinate_policy, **kwargs):
    """Instantiate the consolidated ETNN with the baseline relation set.

    The neighborhood list mirrors the public ETNN and ETNN-LapPE configs.  That
    keeps these unit tests aligned with the graph-to-combinatorial lifting
    defaults used by the challenge pipeline.
    """
    return ETNNCoordinatePolicy(
        in_channels=16,
        hidden_channels=8,
        out_channels=16,
        neighborhoods=[
            "up_adjacency-0",
            "up_adjacency-1",
            "up_adjacency-2",
            "up_incidence-0",
            "down_incidence-1",
            "up_incidence-1",
            "down_incidence-2",
        ],
        num_layers=2,
        coordinate_policy=coordinate_policy,
        **kwargs,
    )


def _rank_projected_features(model, batch):
    """Project a test batch into the hidden state expected by one ETNN layer."""
    return {
        rank: model.input_projection[str(rank)](getattr(batch, f"x_{rank}"))
        for rank in range(model.max_rank + 1)
    }


def _force_deterministic_coordinate_update_weight(layer, value=1.0):
    """Make the bias-free NSAPH coordinate-update map deterministic."""
    assert isinstance(layer.coordinate_update_mlp, torch.nn.Linear)
    assert layer.coordinate_update_mlp.bias is None
    layer.coordinate_update_mlp.weight.data.fill_(value)


def _first_layer_updated_coordinates(model, batch):
    """Run the first layer and return its internal updated rank-0 coordinates."""
    x = _rank_projected_features(model, batch)
    geometry = _build_physical_cell_geometry(
        batch=batch,
        coordinate_attr="pos",
        max_rank=model.max_rank,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    _, updated_coordinates = model.layers[0](
        x=x,
        batch=batch,
        structural_coordinates=None,
        physical_geometry=geometry,
        physical_coordinates=batch.pos.clone(),
    )
    return updated_coordinates


def test_coordinate_policy_none_runs_without_coordinates():
    """The unified backbone should preserve coordinate-free ETNN behavior.

    This is the GraphUniverse-safe policy: no ``LapPE`` and no physical
    ``pos`` are required.  The relation message sees only the scalar sparse
    TopoBench relation value, so the expected edge-channel width is one.
    """
    batch = create_lappe_complex_batch()

    out = create_coordinate_policy_etnn("none")(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
    assert _edge_channels_for_coordinate_policy("none") == 1


def test_coordinate_policy_structural_lappe_runs_from_lappe():
    """Structural mode should consume LapPE without requiring physical pos.

    LapPE mode is deliberately structural rather than physical.  The model
    receives graph-derived pseudo-coordinates, lifts them through incidence
    averaging, and appends one squared-distance channel to each relation edge.
    """
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    assert "pos" not in batch

    model = create_coordinate_policy_etnn("structural_lappe")
    out = model(batch)

    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
    assert model.layers[0].edge_channels == 2


def test_coordinate_policy_none_outputs_fit_tune_wrapper_contract():
    """Coordinate-free outputs should satisfy ``TuneWrapper``."""
    batch = create_mock_complex_batch()
    wrapper = TuneWrapper(
        backbone=create_coordinate_policy_etnn("none"),
        out_channels=16,
        num_cell_dimensions=3,
        residual_connections=False,
    )

    out = wrapper(batch)

    assert torch.equal(out["labels"], batch.y)
    assert torch.equal(out["batch_0"], batch.batch_0)
    assert out["x_0"].shape == batch.x_0.shape
    assert out["x_1"].shape == batch.x_1.shape
    assert out["x_2"].shape == batch.x_2.shape


def test_coordinate_policy_none_handles_empty_rank_with_stored_zero_edges():
    """Stored zero placeholders should not create messages for empty ranks."""
    batch = create_mock_complex_batch()
    batch.x_2 = torch.empty(0, 16)
    batch.batch_2 = torch.empty(0, dtype=torch.long)
    batch["up_adjacency-2"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.zeros(2),
        size=(2, 2),
    ).coalesce()
    batch["up_incidence-1"] = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(0, 4),
    ).coalesce()
    batch["down_incidence-2"] = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(4, 0),
    ).coalesce()

    out = create_coordinate_policy_etnn("none")(batch)

    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_neighborhood_to_edge_index_uses_columns_as_senders():
    """Sparse relation rows are receivers and columns are senders."""
    up_incidence_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.ones(2),
        size=(1, 2),
    ).coalesce()
    batch = Data(**{"up_incidence-0": up_incidence_0})

    edge_index, edge_attr = _neighborhood_to_edge_index(
        batch=batch,
        neighborhood="up_incidence-0",
        src_rank=0,
        dst_rank=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_src_cells=2,
        num_dst_cells=1,
    )

    expected_edge_index = torch.tensor([[0, 1], [0, 0]])
    assert torch.equal(edge_index, expected_edge_index)
    assert edge_attr.shape == (2, 1)


def test_neighborhood_to_edge_index_compacts_empty_rank_placeholders():
    """Sparse axes should compact placeholders for graphs with empty ranks."""
    rank_2_adjacency = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2], [0, 2, 1]]),
        values=torch.tensor([0.0, 1.0, 1.0]),
        size=(3, 3),
    ).coalesce()
    batch = Data(
        x_2=torch.randn(2, 16),
        batch_2=torch.tensor([1, 1]),
        **{"up_adjacency-2": rank_2_adjacency},
    )

    edge_index, edge_attr = _neighborhood_to_edge_index(
        batch=batch,
        neighborhood="up_adjacency-2",
        src_rank=2,
        dst_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_src_cells=2,
        num_dst_cells=2,
    )

    expected_edge_index = torch.tensor([[1, 0], [0, 1]])
    assert torch.equal(edge_index, expected_edge_index)
    assert edge_attr.shape == (2, 1)


def test_neighborhood_to_edge_index_rejects_ambiguous_sparse_axis():
    """Sparse-axis compaction should require rank-wise batch metadata."""
    rank_2_adjacency = torch.sparse_coo_tensor(
        indices=torch.tensor([[1, 2], [2, 1]]),
        values=torch.ones(2),
        size=(3, 3),
    ).coalesce()
    batch = Data(
        x_2=torch.randn(2, 16),
        **{"up_adjacency-2": rank_2_adjacency},
    )

    with pytest.raises(ValueError, match="batch_2"):
        _neighborhood_to_edge_index(
            batch=batch,
            neighborhood="up_adjacency-2",
            src_rank=2,
            dst_rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_src_cells=2,
            num_dst_cells=2,
        )


def test_coordinate_policy_requires_neighborhoods():
    """A typed ETNN backbone should require at least one relation."""
    with pytest.raises(ValueError, match="at least one neighborhood"):
        ETNNCoordinatePolicy(
            in_channels=16,
            hidden_channels=8,
            out_channels=16,
            neighborhoods=[],
        )


def test_coordinate_policy_structural_lappe_requires_lappe_attribute():
    """Structural mode should not silently synthesize missing coordinates."""
    batch = create_lappe_complex_batch()

    with pytest.raises(AttributeError, match="LapPE"):
        create_coordinate_policy_etnn("structural_lappe")(batch)


def test_coordinate_policy_structural_lappe_validates_rank_0_rows():
    """LapPE rows should align one-to-one with rank-0 cells."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0] + 1, 3)

    with pytest.raises(ValueError, match="one coordinate row per rank-0"):
        create_coordinate_policy_etnn("structural_lappe")(batch)


def test_structural_lappe_coordinates_are_barycentric_by_rank():
    """Higher-rank structural coordinates should use incidence averages."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_rank_1 = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    expected_rank_2 = torch.tensor(
        [
            [4.0 / 3.0, 1.0],
            [1.0, 4.0 / 3.0],
        ]
    )
    assert torch.allclose(coordinates[0], batch.LapPE)
    assert torch.allclose(coordinates[1], expected_rank_1)
    assert torch.allclose(coordinates[2], expected_rank_2)


def test_structural_lappe_coordinates_handle_empty_rank():
    """Empty ranks should receive empty structural-coordinate tensors."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.x_2 = torch.empty(0, 16)
    batch.batch_2 = torch.empty(0, dtype=torch.long)
    batch.incidence_2 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_1.shape[0], 0),
    ).coalesce()

    coordinates = _build_lappe_cell_coordinates(
        batch=batch,
        coordinate_attr="LapPE",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coordinates[2].shape == (0, 3)


def test_structural_lappe_coordinates_validate_incidence_source_axis():
    """Incidence source rows should align with lower-rank coordinates."""
    batch = create_lappe_complex_batch()
    batch.LapPE = torch.randn(batch.x_0.shape[0], 3)
    batch.incidence_1 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_0.shape[0] + 1, batch.x_1.shape[0]),
    ).coalesce()

    with pytest.raises(ValueError, match="source rows"):
        _build_lappe_cell_coordinates(
            batch=batch,
            coordinate_attr="LapPE",
            max_rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_structural_lappe_squared_distance_is_rigid_motion_invariant():
    """Structural distances should ignore rigid coordinate-frame choices."""
    src = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    dst = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [0, 1]])
    base = _squared_coordinate_distances(
        src,
        dst,
        edge_index,
        torch.float32,
    )

    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([4.0, -2.0])
    transformed = _squared_coordinate_distances(
        src @ rotation_reflection.T + translation,
        dst @ rotation_reflection.T + translation,
        edge_index,
        torch.float32,
    )

    assert torch.allclose(base, transformed)


@pytest.mark.parametrize("coordinate_policy", ["none", "structural_lappe"])
def test_invariant_normalization_requires_physical_policy(coordinate_policy):
    """Invariant normalization should not be silently ignored.

    Physical invariant normalization acts on centroid, diameter, and Hausdorff
    channels. Coordinate-free and structural-LapPE policies do not build those
    channels, so enabling normalization there is a malformed config rather than
    a meaningful no-op.
    """
    with pytest.raises(ValueError, match="coordinate_policy='physical'"):
        create_coordinate_policy_etnn(
            coordinate_policy,
            invariant_normalization="batch_norm",
        )


def test_coordinate_policy_physical_runs_from_pos():
    """Physical mode should append NSAPH-style invariants from pos.

    Physical mode is the path closest to the original ETNN/NSAPH setting.  It
    requires rank-0 Euclidean coordinates and, by default, uses five invariant
    edge channels: centroid distance, sender diameter, receiver diameter, and
    two directed Hausdorff-style distances. The sparse TopoBench relation
    supplies edge indices/relation type, not an additional scalar physical edge
    attribute.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert "LapPE" not in batch

    model = create_coordinate_policy_etnn("physical")
    out = model(batch)

    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
    assert model.layers[0].edge_channels == 5
    assert (
        _edge_channels_for_coordinate_policy(
            "physical",
            hausdorff_dists=True,
        )
        == 5
    )
    assert (
        _edge_channels_for_coordinate_policy(
            "physical",
            hausdorff_dists=False,
        )
        == 3
    )


def test_coordinate_policy_physical_pos_update_changes_internal_coordinates():
    """Physical mode should support NSAPH-style rank-0 coordinate updates.

    Updated coordinates stay internal to the backbone/layer so the TopoBench
    wrapper/readout contract remains feature-only.  The original ``batch.pos``
    must not be mutated in place.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    original_pos = batch.pos.clone()
    model = create_coordinate_policy_etnn("physical", pos_update=True)

    def deterministic_update(*, coordinates, edge_index, aggregated_message):
        return coordinates + coordinates.new_tensor([0.25, -0.125])

    model.layers[0]._update_rank_0_coordinates = deterministic_update

    updated_coordinates = _first_layer_updated_coordinates(model, batch)

    assert updated_coordinates.shape == batch.pos.shape
    assert not torch.allclose(updated_coordinates, original_pos)
    assert torch.allclose(batch.pos, original_pos)


@pytest.mark.parametrize("pos_update", [False, True])
def test_physical_policy_forward_backward_smoke(pos_update):
    """Physical mode should keep gradients through invariant geometry.

    This small synthetic smoke test catches broken autograd paths through
    physical coordinates, Hausdorff/diameter invariants, message passing, and
    the optional coordinate-update loop before an end-to-end QM9 evaluation.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        requires_grad=True,
    )
    model = create_coordinate_policy_etnn(
        "physical",
        pos_update=pos_update,
        hausdorff_dists=True,
    )

    out = model(batch)
    loss = sum(features.pow(2).mean() for features in out.values())
    loss.backward()

    assert batch.pos.grad is not None
    assert torch.isfinite(batch.pos.grad).all()


def test_physical_coordinate_update_uses_bias_free_linear_map():
    """Physical coordinate updates should match NSAPH's simple radial map."""
    model = create_coordinate_policy_etnn("physical", pos_update=True)

    update_map = model.layers[0].coordinate_update_mlp

    assert isinstance(update_map, torch.nn.Linear)
    assert update_map.in_features == model.hidden_channels
    assert update_map.out_features == 1
    assert update_map.bias is None


def test_physical_forward_recomputes_geometry_after_coordinate_update(
    monkeypatch,
):
    """Later layers should receive geometry from updated physical coordinates.

    NSAPH recomputes geometric invariants after coordinate updates.  This test
    spies on the top-level geometry summarizer and verifies that the second layer
    sees coordinates changed by the first layer, while the input ``batch.pos``
    remains untouched.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    original_pos = batch.pos.clone()
    model = create_coordinate_policy_etnn("physical", pos_update=True)

    def deterministic_update(*, coordinates, edge_index, aggregated_message):
        return coordinates + coordinates.new_tensor([0.25, -0.125])

    model.layers[0]._update_rank_0_coordinates = deterministic_update

    seen_coordinates = []
    original_summarizer = etnn_policy_module._summarize_physical_cell_geometry

    def spy_summarizer(vertex_coordinates, vertex_memberships):
        seen_coordinates.append(vertex_coordinates.detach().clone())
        return original_summarizer(vertex_coordinates, vertex_memberships)

    monkeypatch.setattr(
        etnn_policy_module,
        "_summarize_physical_cell_geometry",
        spy_summarizer,
    )

    model(batch)

    assert len(seen_coordinates) == model.num_layers
    assert torch.allclose(seen_coordinates[0], original_pos)
    assert not torch.allclose(seen_coordinates[1], original_pos)
    assert torch.allclose(batch.pos, original_pos)


def test_physical_coordinate_update_is_rigid_motion_equivariant():
    """Rank-0 coordinate updates should transform equivariantly under E(n).

    The learned scalar is computed from invariant inputs, while the displacement
    uses relative coordinate vectors.  Therefore applying a rigid transformation
    before the layer should transform the updated coordinates in the same way.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    model = create_coordinate_policy_etnn("physical", pos_update=True)
    _force_deterministic_coordinate_update_weight(model.layers[0], value=1.0)

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    aggregated_message = torch.ones(batch.pos.shape[0], model.hidden_channels)

    updated = model.layers[0]._update_rank_0_coordinates(
        coordinates=batch.pos,
        edge_index=edge_index,
        aggregated_message=aggregated_message,
    )

    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([5.0, -3.0])
    transformed_coordinates = batch.pos @ rotation_reflection.T + translation

    transformed_updated = model.layers[0]._update_rank_0_coordinates(
        coordinates=transformed_coordinates,
        edge_index=edge_index,
        aggregated_message=aggregated_message,
    )
    expected = updated @ rotation_reflection.T + translation

    assert not torch.allclose(updated, batch.pos)
    assert torch.allclose(transformed_updated, expected, atol=1e-5)


def test_coordinate_policy_physical_requires_pos():
    """Physical mode should fail loudly when Euclidean coordinates are absent.

    The model should not silently fall back to coordinate-free behavior when a
    config explicitly requests physical ETNN invariants.  A missing ``pos``
    tensor is therefore a malformed physical-policy input.
    """
    batch = create_lappe_complex_batch()

    with pytest.raises(AttributeError, match="pos"):
        create_coordinate_policy_etnn("physical")(batch)


def test_coordinate_policy_rejects_unknown_policy():
    """Policy names should be explicit to avoid ambiguous model behavior.

    The public config should be reproducible. Unsupported modes such as
    ``auto`` are rejected here so coordinate use is a visible model choice
    instead of hidden dataset-dependent behavior.
    """
    with pytest.raises(ValueError, match="Unsupported"):
        create_coordinate_policy_etnn("auto")


def test_pos_update_requires_physical_policy():
    """Coordinate updates are meaningful only for real physical coordinates."""
    with pytest.raises(ValueError, match="pos_update"):
        create_coordinate_policy_etnn("structural_lappe", pos_update=True)


def test_pos_update_requires_rank_0_to_rank_0_neighborhood():
    """Coordinate updates need a rank-0 physical adjacency relation."""
    with pytest.raises(ValueError, match="rank-0 to rank-0"):
        ETNNCoordinatePolicy(
            in_channels=16,
            hidden_channels=8,
            out_channels=16,
            neighborhoods=["up_incidence-0"],
            num_layers=1,
            coordinate_policy="physical",
            pos_update=True,
            coordinate_update_neighborhood="up_incidence-0",
        )


def test_physical_coordinates_require_explicit_pos_attribute():
    """Physical policy should not silently invent missing coordinates.

    Coordinate-free and structural modes are valid without ``pos``.  Physical
    mode is different: the invariant channels are meaningful only when the
    batch carries an explicit rank-0 coordinate tensor.
    """
    batch = create_lappe_complex_batch()

    with pytest.raises(AttributeError, match="pos"):
        _validate_physical_coordinates(
            batch=batch,
            coordinate_attr="pos",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_physical_coordinates_validate_rank_0_row_alignment():
    """Every rank-0 cell needs exactly one physical coordinate row.

    If the coordinate tensor has a different number of rows from ``x_0``, a
    later message edge could attach geometry to the wrong vertex.  The physical
    policy should catch that before building centroids or diameters.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.randn(batch.x_0.shape[0] + 1, 3)

    with pytest.raises(ValueError, match="one coordinate row"):
        _validate_physical_coordinates(
            batch=batch,
            coordinate_attr="pos",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_physical_cell_geometry_uses_incident_rank_0_vertices():
    """Centroids and diameters should be based on vertex membership.

    The helper composes incidence matrices to recover which rank-0 vertices
    belong to each higher-order cell.  This differs from LapPE's recursive
    coordinate averaging: physical ETNN invariants are defined directly from
    the Euclidean coordinates of incident vertices.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )

    geometry = _build_physical_cell_geometry(
        batch=batch,
        coordinate_attr="pos",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected_rank_1_centroids = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    expected_rank_1_diameters = torch.full((4, 1), 2.0)

    # In this mock complex, each rank-2 cell reaches all four vertices through
    # its incident edges, so both rank-2 cells have the square centroid and the
    # square diagonal as diameter.
    expected_rank_2_centroids = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    expected_rank_2_diameters = torch.full((2, 1), math.sqrt(8.0))

    assert torch.allclose(geometry.centroids[0], batch.pos)
    assert torch.allclose(geometry.diameters[0], torch.zeros(4, 1))
    assert torch.allclose(geometry.centroids[1], expected_rank_1_centroids)
    assert torch.allclose(geometry.diameters[1], expected_rank_1_diameters)
    assert torch.allclose(geometry.centroids[2], expected_rank_2_centroids)
    assert torch.allclose(geometry.diameters[2], expected_rank_2_diameters)


def test_physical_memberships_validate_incidence_axes():
    """Malformed incidence axes should fail before geometry is attached.

    TopoBench incidence matrices follow the convention ``incidence_r`` =
    rank-(r-1) rows by rank-r columns.  Physical cell geometry depends on
    composing those matrices to recover vertex membership, so axis mismatches
    must raise a useful error instead of producing incorrect invariants.
    """
    batch = create_lappe_complex_batch()
    batch.incidence_1 = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(batch.x_0.shape[0] + 1, batch.x_1.shape[0]),
    ).coalesce()

    with pytest.raises(ValueError, match="source rows"):
        _build_vertex_memberships(
            batch=batch,
            max_rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_physical_relation_invariants_are_rigid_motion_invariant():
    """Physical invariant channels should ignore E(n) frame choices.

    The physical policy uses distances and diameters, not raw coordinate
    directions.  Translating, rotating, or reflecting all coordinates in the
    same graph should therefore leave the relation invariant channels
    unchanged.
    """
    batch = create_lappe_complex_batch()
    batch.pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])

    geometry = _build_physical_cell_geometry(
        batch=batch,
        coordinate_attr="pos",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    base = _physical_relation_invariants(
        src_centroids=geometry.centroids[1],
        dst_centroids=geometry.centroids[2],
        src_diameters=geometry.diameters[1],
        dst_diameters=geometry.diameters[2],
        src_membership=geometry.vertex_memberships[1],
        dst_membership=geometry.vertex_memberships[2],
        vertex_coordinates=geometry.vertex_coordinates,
        edge_index=edge_index,
        dtype=torch.float32,
        hausdorff_dists=True,
    )

    # Rotate, reflect, and translate the same physical coordinates.  Distances
    # and diameters should be unchanged because they are E(n)-invariant.
    rotation_reflection = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    translation = torch.tensor([5.0, -3.0])
    batch.pos = batch.pos @ rotation_reflection.T + translation
    transformed_geometry = _build_physical_cell_geometry(
        batch=batch,
        coordinate_attr="pos",
        max_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    transformed = _physical_relation_invariants(
        src_centroids=transformed_geometry.centroids[1],
        dst_centroids=transformed_geometry.centroids[2],
        src_diameters=transformed_geometry.diameters[1],
        dst_diameters=transformed_geometry.diameters[2],
        src_membership=transformed_geometry.vertex_memberships[1],
        dst_membership=transformed_geometry.vertex_memberships[2],
        vertex_coordinates=transformed_geometry.vertex_coordinates,
        edge_index=edge_index,
        dtype=torch.float32,
        hausdorff_dists=True,
    )

    assert torch.allclose(base, transformed)


def test_physical_relation_invariants_include_directed_hausdorff_distances():
    """Physical invariants should include both directed Hausdorff channels.

    Directed Hausdorff distances are asymmetric when one cell has several
    vertices and the other has one far-away vertex.  That makes this tiny
    example a useful guard for the NSAPH five-channel invariant order.
    """
    vertex_coordinates = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    src_membership = torch.tensor([[1.0], [1.0], [0.0]])
    dst_membership = torch.tensor([[0.0], [0.0], [1.0]])
    src_centroids = torch.tensor([[1.0, 0.0]])
    dst_centroids = torch.tensor([[5.0, 0.0]])
    src_diameters = torch.tensor([[2.0]])
    dst_diameters = torch.tensor([[0.0]])

    invariants = _physical_relation_invariants(
        src_centroids=src_centroids,
        dst_centroids=dst_centroids,
        src_diameters=src_diameters,
        dst_diameters=dst_diameters,
        src_membership=src_membership,
        dst_membership=dst_membership,
        vertex_coordinates=vertex_coordinates,
        edge_index=torch.tensor([[0], [0]]),
        dtype=torch.float32,
        hausdorff_dists=True,
    )

    expected = torch.tensor([[4.0, 2.0, 0.0, 5.0, 3.0]])
    assert torch.allclose(invariants, expected)


def test_physical_invariant_normalization_uses_channel_scales():
    """Invariant normalization should rescale each physical channel separately."""
    invariants = torch.tensor(
        [
            [2.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
        ]
    )

    normalized = _normalize_physical_invariants(invariants, eps=1e-8)

    expected = torch.tensor(
        [
            [0.5, 2.0 / 3.0, 0.0],
            [1.5, 4.0 / 3.0, 0.0],
        ]
    )
    assert torch.allclose(normalized, expected)


def test_physical_relation_invariants_handle_empty_relations():
    """Empty sparse relations should produce a zero-row invariant matrix.

    Some lifted mini-batches can contain empty higher-rank relations.  Empty
    physical invariants should keep the same edge-channel width without
    indexing into empty centroid or diameter tensors.
    """
    invariants = _physical_relation_invariants(
        src_centroids=torch.empty(0, 3),
        dst_centroids=torch.empty(2, 3),
        src_diameters=torch.empty(0, 1),
        dst_diameters=torch.zeros(2, 1),
        src_membership=torch.empty(0, 0),
        dst_membership=torch.empty(0, 2),
        vertex_coordinates=torch.empty(0, 3),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        dtype=torch.float32,
        hausdorff_dists=True,
    )

    assert invariants.shape == (0, 5)


def test_physical_policy_edge_attributes_are_invariants_only():
    """Physical edge attributes should not include sparse relation values.

    NSAPH uses adjacency/incidence to define relation edges and passes geometric
    invariants as edge attributes.  The TopoBench sparse value should therefore
    not be concatenated in physical mode.
    """
    layer = _ETNNCoordinatePolicyLayer(
        neighborhoods=["up_adjacency-0"],
        routes=[(0, 0)],
        hidden_channels=8,
        dropout=0.0,
        activation="silu",
        use_batch_norm=False,
        coordinate_policy="physical",
        max_rank=0,
        pos_update=False,
        coordinate_update_scale=0.1,
        coordinate_update_neighborhood="up_adjacency-0",
        hausdorff_dists=False,
        invariant_normalization="none",
        invariant_normalization_eps=1e-8,
    )
    physical_geometry = type(
        "PhysicalGeometry",
        (),
        {
            "vertex_coordinates": torch.zeros(2, 2),
            "vertex_memberships": {0: torch.eye(2)},
            "centroids": {0: torch.zeros(2, 2)},
            "diameters": {0: torch.zeros(2, 1)},
        },
    )()

    edge_attr = layer._build_policy_edge_attributes(
        route_idx=0,
        edge_attr=torch.full((1, 1), 42.0),
        edge_index=torch.tensor([[0], [1]]),
        src_rank=0,
        dst_rank=0,
        structural_coordinates=None,
        physical_geometry=physical_geometry,
    )

    assert edge_attr.shape == (1, 3)
    assert torch.allclose(edge_attr, torch.tensor([[0.0, 0.0, 0.0]]))
    assert not torch.any(edge_attr == 42.0)


def test_physical_policy_can_mean_abs_normalize_invariant_edge_attributes():
    """Physical policy should support mean-absolute invariant normalization."""
    layer = _ETNNCoordinatePolicyLayer(
        neighborhoods=["up_adjacency-0"],
        routes=[(0, 0)],
        hidden_channels=8,
        dropout=0.0,
        activation="silu",
        use_batch_norm=False,
        coordinate_policy="physical",
        max_rank=0,
        pos_update=False,
        coordinate_update_scale=0.1,
        coordinate_update_neighborhood="up_adjacency-0",
        hausdorff_dists=False,
        invariant_normalization="mean_abs",
        invariant_normalization_eps=1e-8,
    )
    physical_geometry = type(
        "PhysicalGeometry",
        (),
        {
            "vertex_coordinates": torch.tensor([[0.0], [2.0], [6.0]]),
            "vertex_memberships": {0: torch.eye(3)},
            "centroids": {0: torch.tensor([[0.0], [2.0], [6.0]])},
            "diameters": {0: torch.zeros(3, 1)},
        },
    )()

    edge_attr = layer._build_policy_edge_attributes(
        route_idx=0,
        edge_attr=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        src_rank=0,
        dst_rank=0,
        structural_coordinates=None,
        physical_geometry=physical_geometry,
    )

    assert torch.allclose(
        edge_attr,
        torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]
        ),
    )


def test_physical_policy_can_batch_norm_invariant_edge_attributes():
    """Physical policy should support NSAPH-style invariant BatchNorm."""
    layer = _ETNNCoordinatePolicyLayer(
        neighborhoods=["up_adjacency-0"],
        routes=[(0, 0)],
        hidden_channels=8,
        dropout=0.0,
        activation="silu",
        use_batch_norm=False,
        coordinate_policy="physical",
        max_rank=0,
        pos_update=False,
        coordinate_update_scale=0.1,
        coordinate_update_neighborhood="up_adjacency-0",
        hausdorff_dists=False,
        invariant_normalization="batch_norm",
        invariant_normalization_eps=1e-8,
    )
    physical_geometry = type(
        "PhysicalGeometry",
        (),
        {
            "vertex_coordinates": torch.tensor([[0.0], [2.0], [6.0]]),
            "vertex_memberships": {0: torch.eye(3)},
            "centroids": {0: torch.tensor([[0.0], [2.0], [6.0]])},
            "diameters": {0: torch.zeros(3, 1)},
        },
    )()

    edge_attr = layer._build_policy_edge_attributes(
        route_idx=0,
        edge_attr=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        src_rank=0,
        dst_rank=0,
        structural_coordinates=None,
        physical_geometry=physical_geometry,
    )

    expected = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(edge_attr, expected, atol=1e-5)


def test_batch_norm_physical_invariants_uses_running_stats_in_eval_mode():
    """Singleton relations should only bypass BatchNorm during training.

    This keeps TopoBench batches robust when a relation has one edge, while
    still allowing eval-mode physical ETNN layers to use learned running
    statistics just like ordinary ``BatchNorm1d`` modules.
    """
    empty_attr = torch.empty(0, 3)
    invariant_attr = torch.tensor([[3.0, 5.0, 7.0]])
    normalizer = torch.nn.BatchNorm1d(3, affine=False)
    normalizer.running_mean = torch.tensor([1.0, 1.0, 1.0])
    normalizer.running_var = torch.tensor([4.0, 16.0, 36.0])

    assert torch.equal(
        _batch_norm_physical_invariants(empty_attr, normalizer),
        empty_attr,
    )

    # Training-mode BatchNorm cannot estimate statistics from a singleton
    # relation, so the compatibility guard leaves the edge attributes unchanged.
    assert torch.allclose(
        _batch_norm_physical_invariants(invariant_attr, normalizer),
        invariant_attr,
    )

    normalizer.eval()
    assert torch.equal(
        _batch_norm_physical_invariants(empty_attr, normalizer),
        empty_attr,
    )
    expected = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.allclose(
        _batch_norm_physical_invariants(invariant_attr, normalizer),
        expected,
        atol=1e-5,
    )


def test_physical_helpers_accept_topobench_collated_lifted_batches():
    """TopoBench collate should preserve ``pos`` and block-batched incidences.

    Raw PyG batching can misalign sparse incidence axes for combinatorial
    complexes.  This test uses TopoBench's ``collate_fn`` so the helper is
    validated against the batching contract used during real training.
    """
    base_graph = load_manual_graph_second_structure()
    base_graph.pos = torch.arange(
        base_graph.num_nodes * 3,
        dtype=torch.float32,
    ).view(base_graph.num_nodes, 3)

    lifting = GraphTriangleInducedCC(complex_dim=3)
    lifted_1 = lifting.forward(base_graph.clone())
    lifted_2 = lifting.forward(base_graph.clone())
    batch = _collate_topobench_data_list([lifted_1, lifted_2])
    visible_ranks = sorted(
        int(key.split("_")[1])
        for key in list(batch.keys())
        if key.startswith("x_")
    )

    geometry = _build_physical_cell_geometry(
        batch=batch,
        coordinate_attr="pos",
        max_rank=max(visible_ranks),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert batch.pos.shape[0] == batch.x_0.shape[0]
    for rank in visible_ranks:
        rank_features = getattr(batch, f"x_{rank}")
        assert geometry.centroids[rank].shape[0] == rank_features.shape[0]
        assert geometry.diameters[rank].shape == (
            rank_features.shape[0],
            1,
        )
    assert torch.all(geometry.diameters[1] > 0)
