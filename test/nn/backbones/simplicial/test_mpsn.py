"""Unit tests for the MPSN (Message Passing Simplicial Network) backbone.

MPSN: Bodnar et al., "Weisfeiler and Lehman Go Topological: Message Passing
Simplicial Networks", ICML 2021, arXiv:2103.03212.

These tests assert behaviour on tiny, hand-built simplicial complexes (NO
GraphUniverse loading), including an explicit numerical oracle for Supplement
Equation 35 and a valid ``C6`` versus ``2K3`` higher-order example.
"""

import pytest
import torch

from topobench.nn.backbones.simplicial.mpsn import MPSN


# --------------------------------------------------------------------------- #
# Helpers: tiny hand-built complexes (unsigned 0/1 incidences, as the clique  #
# lifting produces with signed=False).                                         #
# --------------------------------------------------------------------------- #
def _triangle_complex():
    """A single filled triangle on nodes {0,1,2}.

    Nodes: 0,1,2 ; Edges: (0,1),(0,2),(1,2) ; Triangles: (0,1,2).

    Returns
    -------
    tuple
        ((x_0, x_1, x_2), (incidence_1, incidence_2)).
    """
    # incidence_1: nodes x edges  (node belongs to edge -> 1)
    #   edge0=(0,1), edge1=(0,2), edge2=(1,2)
    incidence_1 = torch.tensor(
        [
            [1.0, 1.0, 0.0],  # node 0 in e0, e1
            [1.0, 0.0, 1.0],  # node 1 in e0, e2
            [0.0, 1.0, 1.0],  # node 2 in e1, e2
        ]
    )
    # incidence_2: edges x triangles  (edge belongs to triangle -> 1)
    #   tri0 = (0,1,2) -> all three edges
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0]])

    x_0 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)  # (3, 1)
    x_1 = torch.arange(1, 4, dtype=torch.float).unsqueeze(1)  # (3, 1)
    x_2 = torch.ones(1, 1)  # (1, 1)
    return (x_0, x_1, x_2), (incidence_1, incidence_2)


def _c6_and_two_triangles():
    """Return clique complexes for the cycle C6 and disjoint union 2K3."""
    incidence_1_c6 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    incidence_1_2k3 = torch.block_diag(
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    incidence_2_2k3 = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    c6 = (
        (torch.ones(6, 1), torch.ones(6, 1), torch.empty(0, 1)),
        (incidence_1_c6, torch.empty(6, 0)),
    )
    two_triangles = (
        (torch.ones(6, 1), torch.ones(6, 1), torch.ones(2, 1)),
        (incidence_1_2k3, incidence_2_2k3),
    )
    return c6, two_triangles


def _make_model(in_channels_all=(1, 1, 1), hidden_dim=8, n_layers=3):
    """Construct an MPSN backbone with fixed seed for reproducibility.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input channels on (nodes, edges, triangles).
    hidden_dim : int
        Hidden width (shared across ranks).
    n_layers : int
        Number of MPSN layers.

    Returns
    -------
    MPSN
        The constructed backbone.
    """
    torch.manual_seed(0)
    return MPSN(
        in_channels_all=in_channels_all,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_per_rank_output_shapes():
    """Forward returns one tensor per rank with hidden_dim width."""
    hidden_dim = 8
    model = _make_model(hidden_dim=hidden_dim)
    x_all, inc_all = _triangle_complex()

    x_0, x_1, x_2 = model(x_all, inc_all)

    assert x_0.shape == (3, hidden_dim)
    assert x_1.shape == (3, hidden_dim)
    assert x_2.shape == (1, hidden_dim)


def test_forward_runs_end_to_end_and_is_finite():
    """The forward pass runs on a toy complex and produces finite outputs."""
    model = _make_model()
    x_all, inc_all = _triangle_complex()

    outs = model(x_all, inc_all)

    assert len(outs) == 3
    assert all(torch.isfinite(o).all() for o in outs)


def test_mpsn_is_exported_without_private_layer_helpers():
    """Auto-discovery exports the public backbone and no private layer type."""
    import topobench.nn.backbones as backbones

    assert backbones.MODEL_CLASSES["MPSN"] is backbones.MPSN
    assert backbones.MPSN.__name__ == MPSN.__name__
    assert "_MPSNLayer" not in backbones.MODEL_CLASSES
    assert not hasattr(backbones, "_MPSNLayer")


def test_one_layer_matches_supplement_equation_35_hand_calculation():
    """A deterministic layer matches every term of Supplement Eq. 35.

    Deliberately asymmetric maps make branch collapse, reversed pair fields,
    lost neighbour/coface pairings, or reordered combine inputs observable.
    Negative upper-message, upper-branch, and combine preactivations exercise
    ELU's negative branch rather than reducing the oracle to linear arithmetic.
    """
    model = MPSN(in_channels_all=(1, 1, 1), hidden_dim=1, n_layers=1)
    with torch.no_grad():
        for projection in (
            model.in_linear_0,
            model.in_linear_1,
            model.in_linear_2,
        ):
            projection.weight.fill_(1.0)
            projection.bias.zero_()

        for update in model.layers[0].rank_updates:
            # M_B(z) = ELU(0.5 * ELU(2z - 1) + 1)
            update.boundary_mlp[0].weight.fill_(2.0)
            update.boundary_mlp[0].bias.fill_(-1.0)
            update.boundary_mlp[2].weight.fill_(0.5)
            update.boundary_mlp[2].bias.fill_(1.0)

            # M_up(z) = ELU(0.5 * ELU(0.5z - 2) + 1)
            update.upper_mlp[0].weight.fill_(0.5)
            update.upper_mlp[0].bias.fill_(-2.0)
            update.upper_mlp[2].weight.fill_(0.5)
            update.upper_mlp[2].bias.fill_(1.0)

            # M_U(b || u) = ELU(0.5b - 2u + 0.25)
            update.combine_mlp[0].weight.copy_(torch.tensor([[0.5, -2.0]]))
            update.combine_mlp[0].bias.fill_(0.25)

            if update.upper_message_mlp is not None:
                # M_M(h_neighbour || h_coface) = ELU(2h_neighbour - h_coface)
                update.upper_message_mlp[0].weight.copy_(
                    torch.tensor([[2.0, -1.0]])
                )
                update.upper_message_mlp[0].bias.zero_()

    incidence_1 = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    )
    incidence_2 = torch.ones(3, 1)
    features = (
        torch.tensor([[1.0], [2.0], [4.0]]),
        torch.tensor([[3.0], [5.0], [7.0]]),
        torch.tensor([[11.0]]),
    )

    actual = model(features, (incidence_1, incidence_2))

    # Write E(z) = z for z >= 0 and exp(z)-1 otherwise.  The four maps above
    # are therefore:
    #   M_M(n,c)=E(2n-c), B(z)=E(0.5E(2z-1)+1),
    #   U(z)=E(0.5E(0.5z-2)+1), C(b,u)=E(0.5b-2u+0.25).
    #
    # Rank 0 has no boundary.  Its ordered (neighbour, shared-edge) messages
    # give upper inputs
    #   [1 + 3, E(-1) + 1, E(-3) + E(-3)] + self
    # = [5, 2.3678794412, 2.0995741367].
    # Thus B=[1.5,2.5,4.5], U=[1.25,0.7210851274,0.6933293414],
    # and C=[E(-1.5),0.0578297453,1.1133413172].
    #
    # Rank 1 boundary sums are [1+2,1+4,2+4]=[3,5,6], so its boundary inputs
    # [6,10,13] yield B=[6.5,10.5,13.5].  Ordered messages through triangle 11
    # give upper inputs [5.3678794412,7.0067379470,5.3746173882], hence
    # U=[1.3419698603,1.7516844867,1.3436543470] and the listed final outputs.
    # Rank 2 has boundary input 11+(3+5+7)=26 and empty-upper input 11:
    # B=26.5, U=2.75, C=8.0.
    expected = (
        torch.tensor([[-0.77686984], [0.05782975], [1.11334133]]),
        torch.tensor([[0.81606030], [1.99663103], [4.31269121]]),
        torch.tensor([[8.0]]),
    )

    for actual_rank, expected_rank in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_rank, expected_rank, atol=1e-6, rtol=0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for accelerator routing",
)
def test_dense_cuda_incidence_requires_sparse_input():
    """CUDA topology requires and correctly routes canonical sparse support."""
    model = _make_model(hidden_dim=4, n_layers=1).cuda()
    features, incidence = _triangle_complex()
    cuda_features = tuple(feature.cuda() for feature in features)
    dense_cuda_incidence = tuple(item.cuda() for item in incidence)

    with pytest.raises(
        ValueError, match="Dense accelerator incidence.*sparse"
    ):
        model(cuda_features, dense_cuda_incidence)

    sparse_cuda_incidence = tuple(
        item.to_sparse_coo().cuda() for item in incidence
    )
    outputs = model(cuda_features, sparse_cuda_incidence)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_upper_messages_preserve_neighbor_shared_coface_pairing():
    """Equal marginals with different upper pairings remain distinguishable.

    The target edge belongs to both triangles.  Swapping the two coface feature
    assignments preserves its four neighbours and both marginal feature sums,
    but changes which coface feature is paired with each neighbour.
    """
    model = _make_model(hidden_dim=4, n_layers=1)
    incidence_1 = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    pairing_a = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    features_a = (
        torch.zeros(4, 1),
        torch.tensor([[0.0], [1.0], [2.0], [4.0], [8.0]]),
        torch.tensor([[2.0], [5.0]]),
    )
    features_b = (features_a[0], features_a[1], features_a[2].flip(0))

    target_a = model(features_a, (incidence_1, pairing_a))[1][0]
    target_b = model(features_b, (incidence_1, pairing_a))[1][0]

    assert not torch.allclose(target_a, target_b, atol=1e-6, rtol=0)


def test_dense_and_sparse_incidence_supports_are_equivalent():
    """Dense and sparse COO representations define identical routing."""
    model = _make_model(n_layers=2).eval()
    features, dense_incidence = _triangle_complex()
    sparse_incidence = tuple(
        incidence.to_sparse_coo() for incidence in dense_incidence
    )

    dense_out = model(features, dense_incidence)
    sparse_out = model(features, sparse_incidence)

    for dense_rank, sparse_rank in zip(dense_out, sparse_out, strict=True):
        torch.testing.assert_close(
            dense_rank, sparse_rank, atol=1e-6, rtol=1e-6
        )


def test_incidence_signs_do_not_change_support_routing():
    """Orientation signs do not affect the unsigned topological routes."""
    model = _make_model(n_layers=2).eval()
    features, unsigned = _triangle_complex()
    signed = (
        torch.tensor([[-1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]),
        torch.tensor([[1.0], [-1.0], [1.0]]),
    )
    expected = model(features, unsigned)

    for incidence in (signed, tuple(item.to_sparse() for item in signed)):
        actual = model(features, incidence)
        for expected_rank, actual_rank in zip(expected, actual, strict=True):
            torch.testing.assert_close(
                expected_rank, actual_rank, atol=1e-6, rtol=1e-6
            )


def test_explicit_sparse_zeros_do_not_create_topological_routes():
    """Explicit sparse zeros are excluded from incidence support."""
    model = _make_model(n_layers=2).eval()
    features, dense_incidence = _triangle_complex()
    expected = model(features, dense_incidence)

    sparse_with_zero = []
    for incidence in dense_incidence:
        canonical = incidence.to_sparse_coo().coalesce()
        indices = torch.cat(
            (canonical.indices(), torch.tensor([[0], [0]])), dim=1
        )
        values = torch.cat((canonical.values(), torch.zeros(1)))
        sparse_with_zero.append(
            torch.sparse_coo_tensor(indices, values, incidence.shape)
        )

    actual = model(features, tuple(sparse_with_zero))
    for expected_rank, actual_rank in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            expected_rank, actual_rank, atol=1e-6, rtol=1e-6
        )


def test_simplex_permutation_equivariance():
    """Consistent independent rank permutations only permute the outputs."""
    model = _make_model(hidden_dim=4, n_layers=2).eval()
    features = (
        torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        torch.tensor([[5.0], [6.0], [7.0], [8.0], [9.0]]),
        torch.tensor([[10.0], [11.0]]),
    )
    incidence_1 = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )
    incidence_2 = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    permutations = (
        torch.tensor([3, 1, 0, 2]),
        torch.tensor([4, 2, 0, 3, 1]),
        torch.tensor([1, 0]),
    )

    expected = model(features, (incidence_1, incidence_2))
    permuted_features = tuple(
        x[permutation]
        for x, permutation in zip(features, permutations, strict=True)
    )
    permuted_incidence = (
        incidence_1[permutations[0]][:, permutations[1]],
        incidence_2[permutations[1]][:, permutations[2]],
    )
    actual = model(permuted_features, permuted_incidence)

    for rank in range(3):
        torch.testing.assert_close(
            actual[rank],
            expected[rank][permutations[rank]],
            atol=1e-6,
            rtol=1e-6,
        )


def test_disjoint_union_matches_concatenated_individual_outputs():
    """Block-diagonal batching cannot pass messages between complexes."""
    model = _make_model(hidden_dim=4, n_layers=2).eval()
    features_a, incidence_a = _triangle_complex()
    features_b = tuple(
        feature + offset
        for feature, offset in zip(features_a, (3.0, 5.0, 7.0), strict=True)
    )
    incidence_b = tuple(item.clone() for item in incidence_a)

    outputs_a = model(features_a, incidence_a)
    outputs_b = model(features_b, incidence_b)
    union_features = tuple(
        torch.cat((first, second), dim=0)
        for first, second in zip(features_a, features_b, strict=True)
    )
    union_incidence = tuple(
        torch.block_diag(first, second)
        for first, second in zip(incidence_a, incidence_b, strict=True)
    )
    union_outputs = model(union_features, union_incidence)

    for union_rank, first, second in zip(
        union_outputs, outputs_a, outputs_b, strict=True
    ):
        torch.testing.assert_close(
            union_rank,
            torch.cat((first, second), dim=0),
            atol=1e-6,
            rtol=1e-6,
        )


def test_empty_and_minimal_higher_ranks_work_in_train_and_eval():
    """A lone vertex and a single edge need no artificial higher cells."""
    cases = (
        (
            (torch.tensor([[1.0]]), torch.empty(0, 1), torch.empty(0, 1)),
            (torch.empty(1, 0), torch.empty(0, 0)),
        ),
        (
            (
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[3.0]]),
                torch.empty(0, 1),
            ),
            (torch.ones(2, 1), torch.empty(1, 0)),
        ),
    )
    model = _make_model(hidden_dim=4, n_layers=2)

    for training in (True, False):
        model.train(training)
        for features, incidence in cases:
            outputs = model(features, incidence)
            for output, inputs in zip(outputs, features, strict=True):
                assert output.shape == (inputs.shape[0], 4)
                assert torch.isfinite(output).all()


def test_every_trainable_branch_has_finite_gradients():
    """A filled triangle exercises every registered trainable map."""
    model = _make_model(hidden_dim=4, n_layers=1).train()
    base_features, incidence = _triangle_complex()
    features = tuple(x.clone().requires_grad_() for x in base_features)

    outputs = model(features, incidence)
    loss = sum(output.square().sum() for output in outputs)
    loss.backward()

    assert all(torch.isfinite(output).all() for output in outputs)
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), name


def test_c6_and_two_disjoint_triangles_have_different_higher_order_readouts():
    """The valid 1-WL-indistinguishable pair C6 and 2K3 separates after lift.

    Both graphs have six vertices, six edges, degree two, and constant features.
    Clique lifting creates no 2-simplices for C6 and two for 2K3.  This example
    verifies that the backbone consumes that higher-order structure; it is not
    a claim that this finite learned model realizes the full SWL test.
    """
    model = MPSN(in_channels_all=(1, 1, 1), hidden_dim=1, n_layers=1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(1.0)
                module.bias.zero_()
    c6, two_triangles = _c6_and_two_triangles()

    out_c6 = model(*c6)
    out_2k3 = model(*two_triangles)
    readout_c6 = sum(rank.sum() for rank in out_c6)
    readout_2k3 = sum(rank.sum() for rank in out_2k3)

    assert out_c6[2].shape[0] == 0
    assert out_2k3[2].shape[0] == 2
    assert not torch.allclose(readout_c6, readout_2k3)


def test_upper_adjacency_uses_shared_coface_feature():
    """The rank-1 upper message reads the shared triangle's feature.

    Two complexes identical except for the *feature value* on the filled
    triangle must produce different edge embeddings, proving the upper message
    incorporates h_{shared coface} (paper's M_up term), not just the
    edge-edge adjacency pattern.
    """
    model = _make_model()

    x_all, inc_all = _triangle_complex()
    (x_0, x_1, x_2), (inc_1, inc_2) = x_all, inc_all

    out_a = model((x_0, x_1, x_2), (inc_1, inc_2))
    # Same topology, different triangle feature.
    x_2_b = x_2 + 5.0
    out_b = model((x_0, x_1, x_2_b), (inc_1, inc_2))

    # Edge embeddings depend on the triangle feature via the upper message.
    assert not torch.allclose(out_a[1], out_b[1])


def test_node_upper_adjacency_through_shared_edge():
    """Node (rank-0) embeddings depend on edge features via upper-adjacency.

    Nodes have no boundary; their only neighbourhood is upper-adjacency through
    a shared edge whose feature the message reads. Changing an edge feature must
    change node embeddings.
    """
    model = _make_model()
    (x_0, x_1, x_2), (inc_1, inc_2) = _triangle_complex()

    out_a = model((x_0, x_1, x_2), (inc_1, inc_2))
    x_1_b = x_1 + 3.0
    out_b = model((x_0, x_1_b, x_2), (inc_1, inc_2))

    assert not torch.allclose(out_a[0], out_b[0])


def test_n_layers_configurable():
    """n_layers controls the number of stacked MPSN layers."""
    assert len(_make_model(n_layers=1).layers) == 1
    assert len(_make_model(n_layers=4).layers) == 4
