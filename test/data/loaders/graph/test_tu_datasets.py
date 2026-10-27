"""Tests for the TU dataset loader."""

from omegaconf import OmegaConf

from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader


def test_tu_loader_caches_attribute_variants_separately():
    base = {"data_dir": "root", "data_name": "PROTEINS"}
    plain = TUDatasetLoader(OmegaConf.create(base)).get_data_dir()
    attributes = TUDatasetLoader(
        OmegaConf.create({**base, "use_node_attr": True})
    ).get_data_dir()
    assert plain.endswith("PROTEINS") and attributes != plain
    assert attributes.endswith("_node_attr")
