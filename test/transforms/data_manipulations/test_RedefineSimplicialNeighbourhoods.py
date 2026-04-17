"""Test RedefineSimplicialNeighbourhoods class."""

import pytest
import hydra
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import RedefineSimplicialNeighbourhoods
from topobench.data.preprocessor.preprocessor import PreProcessor


class TestRedefineSimplicialNeighbourhoods:
    """Test RedefineSimplicialNeighbourhoods class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = RedefineSimplicialNeighbourhoods()
        self.relative_config_dir = "../../../configs"


    def test_forward_simple_graph(self):
        """Test the transformation on the ManTra dataset.

        This test verifies that the dataset transformation preserves:
        - The number of keys.
        - The exact set of keys.
        - The tensor values for all relevant attributes.
        """

        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run"
        ):
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=["dataset=simplicial/mantra_orientation", "model=graph/gat"],
                return_hydra_config=True,
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            dataset, data_dir = dataset_loader.load(slice=100)

        # Define transformation configuration
        transforms_config = {
            "RedefineSimplicialNeighbourhoods": {
                "_target_": "topobench.transforms.data_transform.DataTransform",
                "transform_name": "RedefineSimplicialNeighbourhoods",
                "transform_type": None,
                "complex_dim": 3,
                "neighborhoods": None,
                "signed": False,
            }
        }

        transformer_dataset = PreProcessor(dataset, data_dir, transforms_config)

        for transformed, initial in zip(transformer_dataset, dataset):
            # Ensure key integrity
            assert len(initial.keys()) == len(transformed.keys()), "Mismatch in the number of keys"
            assert set(initial.keys()) == set(transformed.keys()), "Keys do not match between datasets"

            # Check tensor equality for all relevant keys
            for key in initial.keys():
                if key not in {"x", "x_0", "x_1", "x_2", "y", "shape"}:
                    try:
                        assert torch.equal(initial[key].to_dense(), transformed[key].to_dense()), f"Mismatch in tensor values for key: {key}"
                    except AttributeError as e:
                        pytest.fail(f"Tensor conversion to dense failed for key: {key}. Error: {e}")

    def test_repr(self):
        """Test the string representation of the transformation class.

        Ensures that the `__repr__` method correctly reflects the class name
        and transformation parameters.
        """

        # Define transformation configuration
        transforms_config = {
            "RedefineSimplicialNeighbourhoods": {
                "_target_": "topobench.transforms.data_transform.DataTransform",
                "transform_name": "RedefineSimplicialNeighbourhoods",
                "transform_type": None,
                "complex_dim": 3,
                "neighborhoods": None,
                "signed": False,
            }
        }

        # Instantiate the transformation
        transform = RedefineSimplicialNeighbourhoods(
            **transforms_config["RedefineSimplicialNeighbourhoods"]
        )

            # Get the string representation
        repr_str = repr(transform)

        # Ensure all keys appear in the representation
        for key in transforms_config["RedefineSimplicialNeighbourhoods"]:
            assert key in repr_str, f"Missing key '{key}' in __repr__ output."
