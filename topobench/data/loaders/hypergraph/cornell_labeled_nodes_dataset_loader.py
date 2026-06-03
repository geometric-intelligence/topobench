"""Loader for Cornell labeled nodes hypergraph datasets.

This loader handles hypergraph datasets with labeled nodes from Cornell's
collection at https://www.cs.cornell.edu/~arb/data/. These datasets are designed
for community detection and node prediction experiments where nodes are labeled
into discrete classes.

Available datasets span multiple domains:
    - E-commerce: walmart-trips, amazon-reviews, trivago-clicks
    - Face-to-face contacts: contact-primary-school, contact-high-school
    - US Congress: senate-bills, house-bills, senate-committees, house-committees

See individual config files in configs/dataset/hypergraph/ for dataset details.

References
----------
Generative hypergraph clustering: from blockmodels to modularity.
Philip S. Chodrow, Nate Veldt, and Austin R. Benson.
Science Advances, 2021.

Minimizing Localized Ratio Cut Objectives in Hypergraphs.
Nate Veldt, Austin R. Benson, and Jon Kleinberg.
Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining (KDD), 2020.

Clustering in graphs and hypergraphs with categorical edge labels.
Ilya Amburg, Nate Veldt, and Austin R. Benson.
Proceedings of the Web Conference (WWW), 2020.
"""

from omegaconf import DictConfig

from topobench.data.datasets.cornell_labeled_nodes_dataset import (
    CornellLabeledNodesDataset,
)
from topobench.data.loaders.base import AbstractLoader


class CornellLabeledNodesDatasetLoader(AbstractLoader):
    """Load Cornell labeled nodes hypergraph datasets.

    This loader provides a unified interface for all Cornell hypergraph datasets
    that have labeled nodes for node classification tasks. The specific dataset
    is determined by the `data_name` parameter in the configuration.

    All Cornell labeled node datasets share the same format:
        - hyperedges-{name}.txt: Comma-separated node IDs per line
        - node-labels-{name}.txt: One label per line (line i = label for node i)
        - label-names-{name}.txt: Class names (one per line)

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data storage
            - data_name: Name of the specific dataset (e.g., 'walmart-trips')

    Examples
    --------
    Load walmart-trips dataset:
        >>> params = DictConfig({'data_dir': './data', 'data_name': 'walmart-trips'})
        >>> loader = CornellLabeledNodesDatasetLoader(params)
        >>> dataset = loader.load_dataset()
        >>> data = dataset[0]
        >>> print(f"Nodes: {data.num_nodes}, Classes: {data.num_class}")
        Nodes: 88860, Classes: 11

    Notes
    -----
    Not yet implemented:
        - trivago-clicks: Statistics from downloaded data do not match Cornell
          website, requires verification before implementation.
        - stackoverflow-answers, mathoverflow-answers: These are multi-label
          datasets (nodes can have multiple labels/tags) and require a different
          implementation.
    """

    def __init__(self, parameters: DictConfig) -> None:
        """Initialize the Cornell dataset loader.

        Parameters
        ----------
        parameters : DictConfig
            Configuration containing data_dir and data_name.
        """
        super().__init__(parameters)

    def load_dataset(self) -> CornellLabeledNodesDataset:
        """Load the specified Cornell labeled nodes dataset.

        Returns
        -------
        CornellLabeledNodesDataset
            The loaded dataset containing hypergraph structure and node labels.

        Raises
        ------
        ValueError
            If the specified dataset name is not available in
            CornellLabeledNodesDataset.DATASETS.
        RuntimeError
            If dataset loading or processing fails.
        """
        dataset = self._initialize_dataset()
        self.data_dir = self.get_data_dir()
        return dataset

    def _initialize_dataset(self) -> CornellLabeledNodesDataset:
        """Initialize the Cornell labeled nodes dataset.

        Returns
        -------
        CornellLabeledNodesDataset
            The initialized dataset instance ready for use.
        """
        return CornellLabeledNodesDataset(
            data_dir=self.parameters.data_dir,
            data_name=self.parameters.data_name,
        )
