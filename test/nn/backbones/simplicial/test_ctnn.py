r"""Unit tests for the Copresheaf Topological Neural Network (CTNN).

The tests are organised around the claims of Hajij et al., "Copresheaf
Topological Neural Networks: A Generalized Deep Learning Framework", NeurIPS
2025 (https://arxiv.org/abs/2505.21251), referred to as [1]:

* the attention coefficients of [1], Definition 11, Eq. (7) and Definition 16,
  Eq. (12) normalise over ``N_k(x)``, so they are row-stochastic per receiver
  and per head;
* the layer of [1], Definition 10 with those attention messages, checked
  against a dense Python replay of the same algebra, including the transport
  step ``rho_{y->x}(v_y)`` and the summation that instantiates ``otimes``;
* the structural guarantees of the transport maps of [1], Table 18: SheafFC
  starts at the identity, SheafSPD is symmetric positive definite with
  eigenvalues bounded below by one, and the diagonal map is a gate;
* directionality, ``rho_{y->x} != rho_{x->y}^T``, which is what separates a
  copresheaf from a cellular sheaf ([1], Table 7) and what makes the
  containment of [1], Theorem 4 strict. A cellular-sheaf transport is used as
  the negative control;
* ``rho = Id`` recovering the Cellular Transformer, as observed in [1],
  Appendix H.5.
"""

import math

import pytest
import torch
import torch_geometric
from torch_geometric.utils import to_undirected

from topobench.nn.backbones.simplicial.ctnn import (
    COPRESHEAF_MAPS,
    CopresheafAttention,
    CopresheafTNN,
    CopresheafTNNLayer,
    DiagonalMLPMap,
    HeadwiseLinear,
    SheafFCMap,
    SheafSPDMap,
)
from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)

MAP_NAMES = sorted(COPRESHEAF_MAPS)

#: Deliberately wider than the shipped `configs/model/simplicial/ctnn.yaml`,
#: which carries only the three downward paths of [1], Appendix H.5. The upward
#: routes are here so the tests also cover a rank that both sends and receives,
#: and a receiver rank fed by two neighborhoods at once.
NEIGHBORHOODS = [
    "up_adjacency-0",
    "down_incidence-1",
    "up_incidence-0",
    "down_incidence-2",
    "up_incidence-1",
]

CHANNELS = 12
HEADS = 3


def lifted_complex(channels=CHANNELS, seed=0, neighborhoods=None):
    """Build a small 2-dimensional simplicial complex with random features.

    Two overlapping cliques, so every rank holds several cells and every
    neighborhood gives most receivers more than one sender. A receiver with a
    single sender has a degenerate softmax, which hides errors in the attention
    weights.

    Parameters
    ----------
    channels : int, optional
        Feature width assigned to every rank. Default is ``CHANNELS``.
    seed : int, optional
        Seed for the feature draw. Default is 0.
    neighborhoods : list of str, optional
        Neighborhood names to materialise. Default is ``NEIGHBORHOODS``.

    Returns
    -------
    torch_geometric.data.Data
        Lifted complex carrying ``x_0``, ``x_1``, ``x_2`` and one sparse matrix
        per neighborhood.
    """
    torch.manual_seed(seed)
    edges = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
            [1, 2, 3, 2, 3, 3, 4, 5, 4, 5, 6, 5, 6, 6],
        ]
    )
    graph = torch_geometric.data.Data(
        x=torch.zeros(7, 1), edge_index=to_undirected(edges), num_nodes=7
    )
    lifting = SimplicialCliqueLifting(
        complex_dim=2,
        neighborhoods=NEIGHBORHOODS
        if neighborhoods is None
        else neighborhoods,
    )
    data = lifting(graph)
    for rank in range(3):
        data[f"x_{rank}"] = torch.randn(data[f"x_{rank}"].shape[0], channels)
    return data


def nonzero_pairs(neighborhood):
    r"""Directed pairs ``y -> x`` carrying a map, per [1], Definition 7.

    The support of a neighborhood matrix is its set of *nonzeros*, which is
    narrower than its sparsity pattern: toponetx stores ``adjacency_matrix``
    with an explicit zero on the diagonal.

    Parameters
    ----------
    neighborhood : torch.Tensor
        Sparse neighborhood matrix of shape ``[num_receivers, num_senders]``.

    Returns
    -------
    torch.Tensor
        Receiver and sender indices, of shape ``[2, num_pairs]``.
    """
    neighborhood = neighborhood.coalesce()
    return neighborhood.indices()[:, neighborhood.values().flatten() != 0]


def dense_attention_message(module, x_receiver, x_sender, neighborhood):
    r"""Recompute ``m_x`` of [1], Eq. (7) and (12) with an explicit loop.

    Reuses the module's own projections and transport parameters, but derives
    the softmax scoping, the transport application and the aggregation
    independently, one directed pair at a time.

    Parameters
    ----------
    module : CopresheafAttention
        Attention module whose parameters to reuse.
    x_receiver : torch.Tensor
        Receiver features of shape ``[num_receivers, channels]``.
    x_sender : torch.Tensor
        Sender features of shape ``[num_senders, channels]``.
    neighborhood : torch.Tensor
        Sparse neighborhood matrix of shape ``[num_receivers, num_senders]``.

    Returns
    -------
    torch.Tensor
        Messages of shape ``[num_receivers, channels]``.
    """
    heads, stalk_dim = module.heads, module.stalk_dim
    # `dense[receiver].nonzero()` below already scopes the softmax to the true
    # support of Definition 3, so this reference is unaffected by the explicit
    # zeros that TopoBench stores in `up_adjacency-r`.
    query = module.lin_query(x_receiver).view(-1, heads, stalk_dim)
    key = module.lin_key(x_sender).view(-1, heads, stalk_dim)
    value = module.lin_value(x_sender).view(-1, heads, stalk_dim)

    dense = neighborhood.coalesce().to_dense()
    message = torch.zeros_like(query)
    for receiver in range(x_receiver.shape[0]):
        senders = dense[receiver].nonzero().flatten().tolist()
        if not senders:
            continue
        for head in range(heads):
            scores = torch.stack(
                [
                    query[receiver, head] @ key[sender, head]
                    for sender in senders
                ]
            ) / math.sqrt(stalk_dim)
            weights = torch.softmax(scores, dim=0)
            for weight, sender in zip(weights, senders, strict=True):
                pair_query = query[receiver, head].view(1, 1, -1)
                pair_key = key[sender, head].view(1, 1, -1)
                rho = module.transport(
                    pair_query.expand(1, heads, -1),
                    pair_key.expand(1, heads, -1),
                )[0, head]
                transported = (
                    rho * value[sender, head]
                    if module.transport.is_diagonal
                    else rho @ value[sender, head]
                )
                message[receiver, head] = (
                    message[receiver, head] + weight * transported
                )
    return message.flatten(start_dim=1)


class TestPaperEquations:
    """Check the implementation against the equations of [1]."""

    @pytest.mark.parametrize("copresheaf_map", MAP_NAMES)
    def test_attention_is_row_stochastic(self, copresheaf_map):
        """Eq. (7) and (12) normalise over ``N_k(x)``, per receiver and head.

        Recovers the coefficients by transporting one-hot value vectors, which
        turns the message into the attention weights themselves.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        """
        data = lifted_complex()
        module = CopresheafAttention(CHANNELS, HEADS, copresheaf_map).eval()
        neighborhood = data["up_adjacency-0"].coalesce()
        receiver, sender = nonzero_pairs(neighborhood)

        query = module.lin_query(data["x_0"]).view(-1, HEADS, module.stalk_dim)
        key = module.lin_key(data["x_0"]).view(-1, HEADS, module.stalk_dim)
        score = (query[receiver] * key[sender]).sum(-1) / math.sqrt(
            module.stalk_dim
        )
        weights = torch_geometric.utils.softmax(
            score, receiver, num_nodes=data["x_0"].shape[0]
        )

        totals = torch.zeros(data["x_0"].shape[0], HEADS)
        totals.index_add_(0, receiver, weights)
        # Every 0-cell of two overlapping cliques has a neighbour.
        torch.testing.assert_close(
            totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("copresheaf_map", MAP_NAMES)
    def test_message_matches_dense_reference(self, copresheaf_map):
        """The aggregated message reproduces Eq. (7) and (12) pair by pair.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        """
        data = lifted_complex()
        module = CopresheafAttention(CHANNELS, HEADS, copresheaf_map).eval()
        # A cross-rank route, so this exercises Definition 16 as well.
        neighborhood = data["down_incidence-1"]

        with torch.no_grad():
            actual = module(data["x_0"], data["x_1"], neighborhood)
            expected = dense_attention_message(
                module, data["x_0"], data["x_1"], neighborhood
            )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_neighborhoods_are_combined_by_summation(self):
        """``otimes`` of Definition 10 is the sum used in Appendix H.5.

        Builds a layer over two neighborhoods that both target rank 0 and
        checks its update against the same update driven by the sum of the two
        single-neighborhood messages.
        """
        names = ["up_adjacency-0", "down_incidence-1"]
        data = lifted_complex(neighborhoods=names)
        routes = [[0, 0], [1, 0]]
        layer = CopresheafTNNLayer(
            CHANNELS, routes, HEADS, "sheaf_fc", dropout=0.0
        ).eval()
        features = {rank: data[f"x_{rank}"] for rank in range(2)}

        with torch.no_grad():
            actual = layer(features, [data[name] for name in names])[0]

            total = layer.attentions[0](
                features[0], features[0], data[names[0]]
            ) + layer.attentions[1](features[0], features[1], data[names[1]])
            hidden = layer.norm_message["0"](features[0] + total)
            expected = layer.norm_update["0"](
                hidden + layer.update["0"](hidden)
            )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_stored_zeros_are_not_neighbours(self):
        r"""A zero entry carries no map, so it sends no message.

        [1], Definition 7 puts a map at each nonzero of the binary matrix of
        Definition 3. TopoBench materialises ``up_adjacency-0`` from toponetx,
        which keeps an explicit zero on the diagonal, so reading the sparsity
        pattern instead of the support would give every 0-cell a self-loop
        ``y = x`` that no neighborhood function asked for.
        """
        data = lifted_complex()
        adjacency = data["up_adjacency-0"].coalesce()
        # The matrix under test must actually exercise the distinction.
        assert (adjacency.values() == 0).any()

        module = CopresheafAttention(CHANNELS, HEADS, "sheaf_fc").eval()
        pairs = nonzero_pairs(adjacency)
        pruned = torch.sparse_coo_tensor(
            pairs, torch.ones(pairs.shape[1]), adjacency.shape
        )

        with torch.no_grad():
            torch.testing.assert_close(
                module(data["x_0"], data["x_0"], adjacency),
                module(data["x_0"], data["x_0"], pruned),
                atol=1e-6,
                rtol=1e-6,
            )

    def test_sheaf_fc_starts_at_the_identity(self):
        """Table 18 zero-initialises SheafFC, so ``rho = Id`` before training."""
        stalk_dim = 4
        transport = SheafFCMap(HEADS, stalk_dim)
        rho = transport(
            torch.randn(5, HEADS, stalk_dim), torch.randn(5, HEADS, stalk_dim)
        )

        torch.testing.assert_close(
            rho, torch.eye(stalk_dim).expand_as(rho), atol=1e-7, rtol=0.0
        )

    def test_identity_transport_recovers_plain_attention(self):
        """Appendix H.5: ``rho = Id`` leaves the Cellular Transformer.

        At initialisation SheafFC is the identity, so the message must equal
        plain attention-weighted value aggregation with no transport.
        """
        data = lifted_complex()
        module = CopresheafAttention(CHANNELS, HEADS, "sheaf_fc").eval()
        neighborhood = data["up_adjacency-0"].coalesce()
        receiver, sender = nonzero_pairs(neighborhood)
        num_cells = data["x_0"].shape[0]

        with torch.no_grad():
            query = module.lin_query(data["x_0"]).view(
                num_cells, HEADS, module.stalk_dim
            )
            key = module.lin_key(data["x_0"]).view(
                num_cells, HEADS, module.stalk_dim
            )
            value = module.lin_value(data["x_0"]).view(
                num_cells, HEADS, module.stalk_dim
            )
            score = (query[receiver] * key[sender]).sum(-1) / math.sqrt(
                module.stalk_dim
            )
            weights = torch_geometric.utils.softmax(
                score, receiver, num_nodes=num_cells
            )
            expected = torch.zeros_like(value)
            expected.index_add_(
                0, receiver, weights.unsqueeze(-1) * value[sender]
            )

            actual = module(data["x_0"], data["x_0"], neighborhood)

        torch.testing.assert_close(
            actual, expected.flatten(start_dim=1), atol=1e-5, rtol=1e-5
        )

    def test_spd_transport_is_positive_definite(self):
        """SheafSPD of Table 18 gives ``Id + Q Q^T``, eigenvalues at least one."""
        stalk_dim = 5
        torch.manual_seed(0)
        transport = SheafSPDMap(HEADS, stalk_dim)
        rho = transport(
            torch.randn(6, HEADS, stalk_dim), torch.randn(6, HEADS, stalk_dim)
        )

        torch.testing.assert_close(
            rho, rho.transpose(-1, -2), atol=1e-6, rtol=1e-6
        )
        assert torch.linalg.eigvalsh(rho).min() >= 1.0 - 1e-5

    def test_diagonal_transport_is_a_gate(self):
        """The Diagonal MLP Map of Table 18 is ``O(d)`` and lands in ``(0, 1)``."""
        stalk_dim = 5
        torch.manual_seed(0)
        transport = DiagonalMLPMap(HEADS, stalk_dim)
        rho = transport(
            torch.randn(6, HEADS, stalk_dim), torch.randn(6, HEADS, stalk_dim)
        )

        assert transport.is_diagonal
        assert rho.shape == (6, HEADS, stalk_dim)
        assert torch.all((rho > 0.0) & (rho < 1.0))

    @pytest.mark.parametrize("copresheaf_map", ["sheaf_fc", "sheaf_spd"])
    def test_transport_is_directional(self, copresheaf_map):
        """``rho_{y->x} != rho_{x->y}^T``, unlike a cellular sheaf.

        Table 7 of [1] contrasts a copresheaf, whose maps are attached to
        directed edges, with a cellular sheaf, whose transport across an
        undirected edge is ``F_{x<|e}^T F_{y<|e}`` and therefore satisfies
        ``rho_{x->y} = rho_{y->x}^T``. The proof of [1], Theorem 4 turns exactly
        that reciprocity into the strictness of ``F_SNN`` inside ``F_CTNN``. A
        sheaf-style transport built from two restriction maps is the control: it
        must satisfy the identity that the copresheaf map violates.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        """
        stalk_dim = 4
        torch.manual_seed(0)
        transport = COPRESHEAF_MAPS[copresheaf_map](HEADS, stalk_dim)
        # SheafFC is the identity at initialisation, which is reciprocal, so
        # the weights are perturbed to reach a generic point of the family.
        for parameter in transport.parameters():
            parameter.data.normal_(std=0.5)

        first = torch.randn(1, HEADS, stalk_dim)
        second = torch.randn(1, HEADS, stalk_dim)
        forward = transport(first, second)
        backward = transport(second, first)

        assert not torch.allclose(
            forward, backward.transpose(-1, -2), atol=1e-3
        )

        # Control: a cellular sheaf transports through the shared edge stalk.
        restriction_x = torch.randn(stalk_dim, stalk_dim)
        restriction_y = torch.randn(stalk_dim, stalk_dim)
        torch.testing.assert_close(
            restriction_x.T @ restriction_y,
            (restriction_y.T @ restriction_x).T,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_routes_follow_the_neighborhood_names(self):
        """Each neighborhood name fixes its sender and receiver rank."""
        model = CopresheafTNN(CHANNELS, NEIGHBORHOODS, heads=HEADS)

        assert model.routes == [[0, 0], [1, 0], [0, 1], [2, 1], [1, 2]]
        assert model.max_rank == 2

    def test_untargeted_ranks_pass_through(self):
        """Definition 10 leaves ``h_x`` alone when every ``N_k(x)`` is empty.

        With only ``down_incidence-1`` no 1-cell or 2-cell receives a message,
        so those stalks must come out of the layer untouched.
        """
        names = ["down_incidence-1"]
        data = lifted_complex(neighborhoods=names)
        model = CopresheafTNN(CHANNELS, names, layers=2, heads=HEADS).eval()

        with torch.no_grad():
            out = model(data)

        assert not torch.allclose(out[0], data["x_0"])
        torch.testing.assert_close(out[1], data["x_1"])


class TestCopresheafTNN:
    """Check construction, propagation and gradients of the backbone."""

    def test_default_construction(self):
        """Defaults follow the Appendix H.5 instantiation of [1]."""
        model = CopresheafTNN(64, NEIGHBORHOODS)

        assert model.copresheaf_map == "sheaf_fc"
        assert len(model.layers) == 2
        assert model.layers[0].attentions[0].heads == 4
        assert model.layers[0].attentions[0].stalk_dim == 16

    def test_empty_neighborhoods_rejected(self):
        """The collection ``N`` of Definition 10 cannot be empty."""
        with pytest.raises(ValueError, match="at least one neighborhood"):
            CopresheafTNN(CHANNELS, [])

    def test_unknown_copresheaf_map_rejected(self):
        """Only the transport maps of Table 18 that are wired up are accepted."""
        with pytest.raises(ValueError, match="Unknown copresheaf_map"):
            CopresheafTNN(CHANNELS, NEIGHBORHOODS, copresheaf_map="outer")

    @pytest.mark.parametrize("heads", [0, 5])
    def test_indivisible_head_count_rejected(self, heads):
        """A head count that does not divide the width would truncate it.

        Parameters
        ----------
        heads : int
            Invalid head count under test.
        """
        with pytest.raises(ValueError, match="must be positive and divide"):
            CopresheafTNN(CHANNELS, NEIGHBORHOODS, heads=heads)

    @pytest.mark.parametrize("copresheaf_map", MAP_NAMES)
    @pytest.mark.parametrize("layers", [1, 3])
    def test_forward_preserves_cell_counts(self, copresheaf_map, layers):
        """Every rank keeps its cell count and its width.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        layers : int
            Number of message-passing layers.
        """
        data = lifted_complex()
        model = CopresheafTNN(
            CHANNELS,
            NEIGHBORHOODS,
            layers=layers,
            heads=HEADS,
            copresheaf_map=copresheaf_map,
        ).eval()

        with torch.no_grad():
            out = model(data)

        assert sorted(out) == [0, 1, 2]
        for rank in out:
            assert out[rank].shape == data[f"x_{rank}"].shape
            assert torch.all(torch.isfinite(out[rank]))

    @pytest.mark.parametrize("copresheaf_map", MAP_NAMES)
    def test_every_parameter_receives_gradient(self, copresheaf_map):
        """No dead parameters: the layer uses every weight it registers.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        """
        data = lifted_complex()
        model = CopresheafTNN(
            CHANNELS, NEIGHBORHOODS, heads=HEADS, copresheaf_map=copresheaf_map
        )
        out = model(data)
        sum((value**2).sum() for value in out.values()).backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} got no gradient"
            assert parameter.grad.abs().sum() > 0, f"{name} has zero gradient"

    def test_messages_stay_inside_the_neighborhood(self):
        """One layer moves information only along ``N_k``.

        Perturbing a single 0-cell must leave every 0-cell that is not adjacent
        to it unchanged. Batched complexes are a disjoint union, so this is also
        what keeps attention from crossing complex boundaries.
        """
        data = lifted_complex()
        model = CopresheafTNN(
            CHANNELS, NEIGHBORHOODS, layers=1, heads=HEADS
        ).eval()
        adjacency = data["up_adjacency-0"].coalesce().to_dense()
        perturbed_cell = 0

        with torch.no_grad():
            before = model(data)[0]
            data["x_0"] = data["x_0"].clone()
            data["x_0"][perturbed_cell] += 1.0
            after = model(data)[0]

        changed = (~torch.isclose(before, after, atol=1e-6)).any(dim=1)
        expected = adjacency[:, perturbed_cell] != 0
        expected[perturbed_cell] = True
        torch.testing.assert_close(changed, expected)

    @pytest.mark.parametrize(
        "neighborhoods",
        [
            NEIGHBORHOODS,
            # The shipped config: the 2 -> 0 route is the composite
            # `incidence_1 . incidence_2`, and TopoBench builds it by dividing
            # the product's values by themselves. A triangle-free complex makes
            # `incidence_2` a zero matrix of width 0, so this is the case where
            # that division could produce NaN or fail outright.
            ["up_adjacency-0", "down_incidence-1", "2-down_incidence-2"],
        ],
        ids=["test_neighborhoods", "shipped_config"],
    )
    @pytest.mark.parametrize(
        "edges",
        [[[0, 1, 2], [1, 2, 3]], [[0], [1]], [[], []]],
        ids=["path", "single_edge", "isolated_nodes"],
    )
    def test_degenerate_complex_still_runs(self, neighborhoods, edges):
        """Empty 2-cells, and even empty 1-cells, must propagate finitely.

        GraphUniverse draws average degrees as low as 1, so triangle-free and
        edge-free complexes reach the model in the challenge grid.

        Parameters
        ----------
        neighborhoods : list of str
            Neighborhood collection under test.
        edges : list of list of int
            Edge index of the graph to lift, possibly empty.
        """
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
        graph = torch_geometric.data.Data(
            x=torch.zeros(4, 1),
            edge_index=to_undirected(edge_index),
            num_nodes=4,
        )
        data = SimplicialCliqueLifting(
            complex_dim=2, neighborhoods=neighborhoods
        )(graph)
        for rank in range(3):
            data[f"x_{rank}"] = torch.randn(
                data[f"x_{rank}"].shape[0], CHANNELS
            )
        model = CopresheafTNN(CHANNELS, neighborhoods, heads=HEADS).eval()

        with torch.no_grad():
            out = model(data)

        assert out[2].shape == (0, CHANNELS)
        assert torch.all(torch.isfinite(out[0]))

    def test_eval_is_deterministic_and_train_is_not(self):
        """Dropout is active in training mode only."""
        data = lifted_complex()
        model = CopresheafTNN(
            CHANNELS, NEIGHBORHOODS, heads=HEADS, dropout=0.5
        )

        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(data)[0], model(data)[0])

        model.train()
        torch.manual_seed(0)
        first = model(data)[0]
        torch.manual_seed(1)
        assert not torch.allclose(first, model(data)[0])

    @pytest.mark.parametrize("copresheaf_map", MAP_NAMES)
    def test_overfits_a_single_complex(self, copresheaf_map):
        """The backbone has enough capacity to drive a toy loss to zero.

        Separates a wrong hyperparameter from a broken model: if transport or
        aggregation were disconnected from the parameters, this would plateau.
        The update of Appendix H.5 ends in a normalisation, so the target is
        normalised too; an arbitrary Gaussian target is outside the layer's
        image and would cap the achievable loss whatever the model does.

        Parameters
        ----------
        copresheaf_map : str
            Transport parameterisation under test.
        """
        data = lifted_complex()
        torch.manual_seed(0)
        target = torch.nn.functional.layer_norm(
            torch.randn_like(data["x_0"]), (CHANNELS,)
        )
        model = CopresheafTNN(
            CHANNELS, NEIGHBORHOODS, heads=HEADS, copresheaf_map=copresheaf_map
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(300):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(data)[0], target)
            loss.backward()
            optimizer.step()

        assert loss.item() < 1e-2


class TestHeadwiseLinear:
    """Check the per-head linear map backing the transport parameterisations."""

    def test_heads_are_independent(self):
        """Each head owns its weights, so zeroing one leaves the others alone."""
        module = HeadwiseLinear(HEADS, 4, 5, bias=False)
        x = torch.randn(2, HEADS, 4)

        before = module(x)
        module.weight.data[1].zero_()
        after = module(x)

        torch.testing.assert_close(after[:, 1], torch.zeros(2, 5))
        torch.testing.assert_close(after[:, 0], before[:, 0])

    def test_matches_per_head_matmul(self):
        """The einsum agrees with an explicit per-head matrix product."""
        module = HeadwiseLinear(HEADS, 4, 5)
        x = torch.randn(6, HEADS, 4)

        expected = torch.stack(
            [
                x[:, head] @ module.weight[head] + module.bias[head]
                for head in range(HEADS)
            ],
            dim=1,
        )

        torch.testing.assert_close(module(x), expected)

    def test_zero_initialisation(self):
        """``zero_init`` makes the map output zero, as Table 18 requires."""
        module = HeadwiseLinear(HEADS, 4, 5, bias=False, zero_init=True)

        torch.testing.assert_close(
            module(torch.randn(3, HEADS, 4)), torch.zeros(3, HEADS, 5)
        )
