"""Walk prediction must not consume dropout draws for unused graph logits."""

import torch

from topobench.nn.readouts.trawl import TRAWLReadout


def test_walk_readout_matches_single_head_call_rng():
    readout = TRAWLReadout(
        4, 1, head_hidden=8, dropout=0.5, aggregation="walk_logits"
    )
    walks = torch.randn(6, 4)
    graph = torch.randn(2, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    torch.manual_seed(81)
    expected = readout.head(walks).reshape(2, 3, 1).mean(1)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(81)
    actual = readout(
        {
            "graph_embedding": graph,
            "walk_embedding": walks,
            "walk_batch": batch,
        },
        None,
    )["logits"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
