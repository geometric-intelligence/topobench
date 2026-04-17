"""General class for lifting simplicial complexes to combinatorial complexes."""

from topobench.transforms.liftings.liftings import SimplicialLifting


class Simplicial2CombinatorialLifting(SimplicialLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "simplicial2combinatorial"
