"""Unit tests for :class:`DirSNNWrapper` and its ``adj_subset`` knob."""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from topobench.nn.backbones.simplicial.dirsnn import DirSNN
from topobench.nn.wrappers.simplicial.dirsnn_wrapper import (
    DIRSNN_ADJ_KEYS,
    DirSNNWrapper,
)
from topobench.transforms.liftings.graph2simplicial import (
    DirectedSimplicialLifting,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_lifted(simple_graph_1):
    """Run :class:`DirectedSimplicialLifting` once and reuse the output.

    Parameters
    ----------
    simple_graph_1 : torch_geometric.data.Data
        Sample graph fixture from ``test/conftest.py``.

    Returns
    -------
    torch_geometric.data.Data
        Lifted batch with the ten ``dir_*_adj_*`` attributes.
    """
    lifting = DirectedSimplicialLifting(signed=True)
    data = lifting(simple_graph_1)
    # ``DirSNNWrapper.forward`` reads ``batch.batch_0``; the test
    # fixture is not run through a DataLoader, so we set it manually
    # (mirroring ``sg1_clique_lifted`` in ``test/conftest.py``).
    data.batch_0 = "null"
    return data


# ---------------------------------------------------------------------------
# adj_subset wiring
# ---------------------------------------------------------------------------


def _make_wrapper(data, adj_subset, n_adjs):
    """Build a tiny :class:`DirSNNWrapper` around a fresh :class:`DirSNN`.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Lifted batch carrying edge features and adjacencies.
    adj_subset : str or None
        Adjacency-subset knob to test.
    n_adjs : int
        Number of adjacencies the backbone expects (must match the
        subset implied by ``adj_subset``).

    Returns
    -------
    DirSNNWrapper
        Wrapper instance ready to call ``forward``.
    """
    edge_channels = data.x_1.shape[1]
    backbone = DirSNN(
        edge_channels=edge_channels,
        n_layers=1,
        n_hid=edge_channels,
        conv_order=1,
        n_adjs=n_adjs,
        update_func="relu",
    )
    return DirSNNWrapper(
        backbone,
        adj_subset=adj_subset,
        out_channels=edge_channels,
        num_cell_dimensions=2,
    )


def test_wrapper_adj_subset_default_uses_all_ten(directed_lifted):
    """``adj_subset=None`` selects the full 10-adjacency tuple.

    Parameters
    ----------
    directed_lifted : torch_geometric.data.Data
        Lifted-graph fixture.
    """
    wrapper = _make_wrapper(directed_lifted, adj_subset=None, n_adjs=10)
    assert wrapper._adj_keys == DIRSNN_ADJ_KEYS
    assert len(wrapper._adj_keys) == 10
    out = wrapper(directed_lifted)
    for key in ("labels", "batch_0", "x_0", "x_1"):
        assert key in out


def test_wrapper_adj_subset_lower(directed_lifted):
    """``adj_subset="lower"`` selects exactly the 4 lower adjacencies.

    Parameters
    ----------
    directed_lifted : torch_geometric.data.Data
        Lifted-graph fixture.
    """
    wrapper = _make_wrapper(directed_lifted, adj_subset="lower", n_adjs=4)
    assert wrapper._adj_keys == DIRSNN_ADJ_KEYS[:4]
    assert all(k.startswith("dir_lower_adj_") for k in wrapper._adj_keys)
    out = wrapper(directed_lifted)
    for key in ("labels", "batch_0", "x_0", "x_1"):
        assert key in out


def test_wrapper_adj_subset_upper(directed_lifted):
    """``adj_subset="upper"`` selects exactly the 6 upper adjacencies.

    Parameters
    ----------
    directed_lifted : torch_geometric.data.Data
        Lifted-graph fixture.
    """
    wrapper = _make_wrapper(directed_lifted, adj_subset="upper", n_adjs=6)
    assert wrapper._adj_keys == DIRSNN_ADJ_KEYS[4:]
    assert all(k.startswith("dir_upper_adj_") for k in wrapper._adj_keys)
    out = wrapper(directed_lifted)
    for key in ("labels", "batch_0", "x_0", "x_1"):
        assert key in out


def test_wrapper_adj_subset_invalid_raises(directed_lifted):
    """Construction rejects ``adj_subset`` values outside the allow-list.

    Parameters
    ----------
    directed_lifted : torch_geometric.data.Data
        Lifted-graph fixture (only used to size the backbone).
    """
    edge_channels = directed_lifted.x_1.shape[1]
    backbone = DirSNN(
        edge_channels=edge_channels,
        n_layers=1,
        n_hid=edge_channels,
        conv_order=1,
        n_adjs=10,
        update_func="relu",
    )
    with pytest.raises(ValueError, match="Unknown adj_subset"):
        DirSNNWrapper(
            backbone,
            adj_subset="middle",  # not in (None, "lower", "upper")
            out_channels=edge_channels,
            num_cell_dimensions=2,
        )


# ---------------------------------------------------------------------------
# Hydra config wiring for the new official-lower variant
# ---------------------------------------------------------------------------


def test_official_lower_config_instantiates_and_runs(directed_lifted):
    """``configs/model/simplicial/dirsnn_official_lower.yaml`` works end-to-end.

    Loads the config via Hydra, instantiates backbone + wrapper, and
    runs a forward pass on the lifted fixture. This guards the yaml
    against accidental drift from the wrapper API (e.g. an ``n_adjs``
    that no longer matches the subset).

    Parameters
    ----------
    directed_lifted : torch_geometric.data.Data
        Lifted-graph fixture.
    """
    configs_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "configs",
            "model",
            "simplicial",
        )
    )

    # Standalone load of the model config (we don't pull in the whole
    # TBModel stack here -- just backbone + backbone_wrapper).
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        cfg = compose(
            config_name="dirsnn_official_lower",
            return_hydra_config=False,
        )
    OmegaConf.set_struct(cfg, False)
    model_cfg = cfg

    # The Hydra-side numerical values are fed through interpolations
    # (e.g. ``${model.feature_encoder.out_channels}``) that only
    # resolve at full-app launch time. For this standalone test we
    # only validate (a) the static fields the new variant cares about
    # and (b) that the backbone/wrapper sub-tree instantiates with
    # concrete values matching those fields.
    edge_channels = directed_lifted.x_1.shape[1]
    # Patch the interpolated sentinel fields so ``instantiate`` doesn't
    # try to resolve them.
    model_cfg.backbone.edge_channels = edge_channels
    model_cfg.backbone.n_hid = edge_channels
    model_cfg.backbone_wrapper.out_channels = edge_channels
    model_cfg.backbone_wrapper.num_cell_dimensions = 2

    assert model_cfg.backbone.n_adjs == 4
    assert model_cfg.backbone_wrapper.adj_subset == "lower"
    assert model_cfg.model_name == "dirsnn_official_lower"

    backbone = instantiate(model_cfg.backbone)
    # ``backbone_wrapper`` is declared as ``_partial_: true``.
    wrapper_partial = instantiate(model_cfg.backbone_wrapper)
    wrapper = wrapper_partial(backbone)
    # NB: Hydra ``_target_`` goes through the auto-discovery loader in
    # ``topobench.nn.wrappers.__init__.WrapperExportsManager`` which
    # re-imports the source file under a *different* module name, so
    # the resulting class object is a distinct instance from the
    # canonical ``simplicial.dirsnn_wrapper.DirSNNWrapper``. We
    # therefore compare by class name rather than by ``isinstance``.
    assert type(wrapper).__name__ == "DirSNNWrapper"
    assert wrapper.adj_subset == "lower"
    assert wrapper._adj_keys == DIRSNN_ADJ_KEYS[:4]

    out = wrapper(directed_lifted)
    for key in ("labels", "batch_0", "x_0", "x_1"):
        assert key in out
