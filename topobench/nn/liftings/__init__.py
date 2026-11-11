"""Learnable lifting modules applied inside models.

Unlike :mod:`topobench.transforms.liftings`, which compute a fixed
complex during preprocessing, the liftings in this package are
trainable components: the topology they produce is differentiable and
is learned end to end with the downstream network.

A backbone plugs a learnable lifting in by instantiating it (or its
pieces) in its constructor and multiplying the gates it produces into
the affected features and messages. The components work for cell
complexes and hypergraphs alike: gates can select 2-cells, learned
edges, or candidate hyperedges. ``model=cell/smcn_difflift`` is a
working example of the plug-in pattern.
"""

from topobench.nn.liftings.difflift import (
    CellScorer,
    DiffLift,
    DiffLiftEncoder,
    EdgeSampler,
)

__all__ = [
    "CellScorer",
    "DiffLift",
    "DiffLiftEncoder",
    "EdgeSampler",
]
