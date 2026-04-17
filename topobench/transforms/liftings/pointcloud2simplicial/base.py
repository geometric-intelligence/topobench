"""Abstract class for lifting pointclouds to simplicial complexes."""

from toponetx.classes import SimplicialComplex

from topobench.data.utils.utils import get_complex_connectivity
from topobench.transforms.liftings.liftings import PointCloudLifting


class PointCloud2SimplicialLifting(PointCloudLifting):
    r"""Abstract class for lifting pointclouds to simplicial complexes.

    Parameters
    ----------
    complex_dim : int, optional
        The dimension of the simplicial complex to be generated. Default is 2.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, complex_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = complex_dim
        self.type = "pointcloud2simplicial"

    def _get_lifted_topology(
        self, simplicial_complex: SimplicialComplex
    ) -> dict:
        r"""Return the lifted topology.

        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.

        Returns
        -------
        dict
            The lifted topology.
        """
        return get_complex_connectivity(simplicial_complex, self.complex_dim)
