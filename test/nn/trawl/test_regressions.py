"""Regressions for evaluation, pretraining, sampling and preprocessing fixes."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn

from topobench.callbacks.model_checkpoint import RankedModelCheckpoint
from topobench.data.utils.trawl.sampling import walk_seed
from topobench.dataloader.samplers import (
    EpochRandomSampler,
    UnpaddedDistributedSampler,
)
from topobench.evaluator import TBEvaluator
from topobench.model.trawl_pretraining import (
    PretrainingCheckpoint,
    validation_frequency,
)
from topobench.nn.backbones.combinatorial.trawl import SISALayer
from topobench.nn.readouts.trawl import TRAWLReadout
from topobench.optimizer import TBOptimizer
from topobench.run import enable_unused_parameter_detection

from .test_trawl import collate


def test_one_logit_auroc_is_batch_invariant():
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 2, (278,), generator=generator)
    logits = (labels - 0.5) * 0.8 + torch.randn(278, generator=generator)
    results = []
    for size in (2, 278):
        evaluator = TBEvaluator(
            "classification", num_classes=2, metrics=["accuracy", "auroc"]
        )
        for start in range(0, len(labels), size):
            stop = start + size
            evaluator.update(
                {
                    "logits": logits[start:stop, None],
                    "labels": labels[start:stop],
                }
            )
        results.append(evaluator.compute())
    expected = roc_auc_score(labels.numpy(), logits.numpy())
    for result in results:
        assert result["auroc"].item() == pytest.approx(expected, abs=1e-6)
        assert result["accuracy"].item() == pytest.approx(
            ((logits > 0).long() == labels).float().mean().item()
        )


class _Strategy:
    @staticmethod
    def reduce_boolean_decision(decision):
        return decision


class _Trainer:
    strategy = _Strategy()


def test_pretraining_checkpoint_min_delta(tmp_path):
    callback = PretrainingCheckpoint(
        dirpath=tmp_path, monitor="loss", mode="min", min_delta=1e-3
    )
    callback.best_k_models = {"a": torch.tensor(1.0)}
    callback.kth_best_model_path = "a"
    assert not callback.check_monitor_top_k(_Trainer(), torch.tensor(0.9995))
    assert callback.check_monitor_top_k(_Trainer(), torch.tensor(0.998))
    with pytest.raises(ValueError):
        PretrainingCheckpoint(dirpath=tmp_path, min_delta=-1.0)


def test_pretraining_resume_keeps_best_across_run_directories(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "epoch=3-step=9.ckpt").write_bytes(b"best")
    state = {
        "dirpath": str(old),
        "best_model_path": str(old / "moved" / "epoch=3-step=9.ckpt"),
        "best_model_score": torch.tensor(0.25),
        "best_k_models": {},
        "kth_best_model_path": "",
        "kth_value": torch.tensor(0.25),
        "last_model_path": "",
    }
    callback = PretrainingCheckpoint(
        dirpath=new, monitor="loss", mode="min", resume_dir=old
    )
    callback.load_state_dict(state)
    target = new / "epoch=3-step=9.ckpt"
    assert target.read_bytes() == b"best"
    assert callback.best_model_path == str(target)
    assert callback.best_k_models == {str(target): torch.tensor(0.25)}
    # A later, worse epoch must not replace the restored best.
    assert not callback.check_monitor_top_k(_Trainer(), torch.tensor(0.3))
    missing = PretrainingCheckpoint(dirpath=tmp_path / "other")
    state["best_model_path"] = str(tmp_path / "absent.ckpt")
    with pytest.raises(FileNotFoundError):
        missing.load_state_dict(state)


def test_plateau_scheduler_steps_with_validation():
    class Holder:
        check_val_every_n_epoch = 5

    assert validation_frequency(Holder()) == 5
    assert validation_frequency(None) == 1
    from topobench.model.model import TBModel

    module = TBModel.__new__(TBModel)
    nn.Module.__init__(module)
    module.backbone, module.readout = nn.Linear(1, 1), nn.Identity()
    module.feature_encoder = nn.Identity()
    module.optimizer = TBOptimizer(
        "AdamW",
        {"lr": 1e-3},
        scheduler={
            "scheduler_id": "ReduceLROnPlateau",
            "monitor": "val/accuracy",
            "scheduler_params": {"mode": "max"},
        },
    )
    module._trainer = Holder()
    config = module.configure_optimizers()
    assert config["lr_scheduler"]["frequency"] == 5
    assert "ReduceLROnPlateau" in repr(module.optimizer)


def test_ranked_checkpoint_evicts_latest_tie(tmp_path):
    callback = RankedModelCheckpoint(
        dirpath=tmp_path, monitor="val/accuracy", mode="max", save_top_k=3
    )
    callback.best_k_models = {
        "e0": torch.tensor(0.7),
        "e1": torch.tensor(0.6),
        "e2": torch.tensor(0.6),
    }
    callback._select_latest_kth()
    assert callback.kth_best_model_path == "e2"


def test_epoch_sampler_order_and_distributed_shards():
    dataset = list(range(11))
    sampler = EpochRandomSampler(dataset, seed=40)
    sampler.set_epoch(3)
    expected = torch.randperm(
        11, generator=torch.Generator().manual_seed(44)
    ).tolist()
    assert list(sampler) == expected and len(sampler) == 11
    shards = [
        EpochRandomSampler(dataset, seed=40, num_replicas=2, rank=rank)
        for rank in range(2)
    ]
    for shard in shards:
        shard.set_epoch(3)
    assert sorted(set(list(shards[0]) + list(shards[1]))) == dataset
    moved = shards[0]
    order = list(moved)
    moved.set_epoch(4)
    assert list(moved) != order
    evaluation = [
        list(UnpaddedDistributedSampler(dataset, num_replicas=2, rank=rank))
        for rank in range(2)
    ]
    assert sorted(evaluation[0] + evaluation[1]) == dataset
    assert len(UnpaddedDistributedSampler(dataset, 2, 1)) == 5


def test_walk_seed_is_collision_free():
    seeds = {
        walk_seed(40, identity, step, view)
        for identity in range(200)
        for step in range(3)
        for view in range(3)
    }
    assert len(seeds) == 200 * 3 * 3


def test_tu_loader_caches_attribute_variants_separately():
    from topobench.data.loaders.graph.tu_datasets import TUDatasetLoader

    base = {"data_dir": "root", "data_name": "PROTEINS"}
    plain = TUDatasetLoader(OmegaConf.create(base)).get_data_dir()
    attributes = TUDatasetLoader(
        OmegaConf.create({**base, "use_node_attr": True})
    ).get_data_dir()
    assert plain.endswith("PROTEINS") and attributes != plain
    assert attributes.endswith("_node_attr")


def test_sisa_wide_decay_matches_and_stays_finite():
    torch.manual_seed(0)
    layer = SISALayer(32, n_heads=4, d_ssm=4, attention_dropout=0.0).eval()
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        factored = layer(x)
        layer.MAX_FACTORED_SPAN = -1.0
        explicit = layer(x)
    torch.testing.assert_close(factored, explicit, atol=1e-5, rtol=1e-5)
    layer.MAX_FACTORED_SPAN = 60.0
    nn.init.constant_(layer.alpha_proj.bias, 20.0)
    layer.train()
    x.requires_grad_(True)
    output = layer(x)
    output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()


def test_linear_readout_uses_input_dropout():
    readout = TRAWLReadout(8, 1, dropout=0.5, input_dropout=0.0)
    assert readout.head[0].p == 0.0


def test_trawl_ddp_strategy_handles_unused_parameters():
    cfg = OmegaConf.create(
        {"model": {"model_name": "trawl"}, "trainer": {"strategy": "ddp"}}
    )
    enable_unused_parameter_detection(cfg)
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_true"
    other = OmegaConf.create(
        {"model": {"model_name": "gcn"}, "trainer": {"strategy": "ddp"}}
    )
    enable_unused_parameter_detection(other)
    assert other.trainer.strategy == "ddp"


def _cpu_autocast_enabled():
    # ``torch.is_autocast_enabled(device_type)`` is unavailable before torch 2.4.
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


@pytest.mark.parametrize("training", [False, True])
def test_evaluation_can_disable_autocast(training):
    from topobench.model.model import TBModel
    from topobench.model.trawl_pretraining import FullPrecisionValidation

    module = TBModel.__new__(TBModel)
    nn.Module.__init__(module)
    module.evaluation_autocast, module.training = False, training
    module.backbone, module.state_str = nn.Identity(), "Validation"
    module._device = torch.device("cpu")
    seen = []

    def forward(batch):
        seen.append(_cpu_autocast_enabled())
        return {}

    module.forward = forward
    module.process_outputs = lambda model_out, batch: model_out
    module.loss = lambda model_out, batch: model_out
    module.evaluator = type("E", (), {"update": lambda self, out: None})()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module.model_step({})
    assert seen == [training]

    class Probe(FullPrecisionValidation, nn.Module):
        device = torch.device("cpu")

        def _step(self, batch, validation=False, batch_idx=0):
            return _cpu_autocast_enabled()

    probe = Probe()
    probe.evaluation_autocast = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert probe.validation_step(None, 0) is False


def test_base_pretraining_masks_differ_per_microbatch():
    from topobench.model.trawl_pretraining import TRAWLPretrainer

    from .test_trawl import model, prepare

    graphs = [prepare(), prepare()]
    pretrainer = TRAWLPretrainer(model(graphs), mask_probability=0.5)
    pretrainer.log = lambda *args, **kwargs: None
    batch = collate(graphs)

    def loss(batch_idx):
        # Fix the walk step so only the mask seed can change.
        pretrainer.backbone.reset_sampling_step()
        return pretrainer._step(batch, batch_idx=batch_idx).item()

    first, again, second = loss(0), loss(0), loss(1)
    assert first == again and first != second


def test_late_metrics_keep_existing_csv_columns(tmp_path):
    from lightning.pytorch.loggers import CSVLogger

    from topobench.run import log_late_metrics

    # The parent's writer exists before a spawned fit, whose workers write
    # rows through their own writer that the parent never sees.
    parent = CSVLogger(tmp_path, name="csv", version=0)
    assert parent.experiment is not None
    worker = CSVLogger(tmp_path, name="csv", version=0)
    worker.log_metrics({"val/accuracy": 0.5, "epoch": 0}, step=0)
    worker.save()
    log_late_metrics([parent], {"test_best_rerun/accuracy": 0.75})
    header = (
        (tmp_path / "csv/version_0/metrics.csv").read_text().splitlines()[0]
    )
    for key in ("val/accuracy", "epoch", "test_best_rerun/accuracy"):
        assert key in header


def test_deterministic_flag_survives_trainer_construction():
    from lightning import Trainer

    cfg = OmegaConf.create({"deterministic": "strict", "trainer": {}})
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        from topobench.run import apply_determinism

        apply_determinism(cfg)
        Trainer(
            logger=False,
            enable_checkpointing=False,
            deterministic=cfg.trainer.deterministic,
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_sequence_cumsum_matches_cumsum():
    from topobench.nn.backbones.combinatorial.trawl import sequence_cumsum

    x = torch.randn(2, 4, 32, 3)
    torch.testing.assert_close(sequence_cumsum(x, 2), torch.cumsum(x, 2))
    if torch.cuda.is_available():
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            result = sequence_cumsum(x.cuda(), 2)
        finally:
            torch.use_deterministic_algorithms(previous)
        torch.testing.assert_close(result.cpu(), torch.cumsum(x, 2))


def test_fast_walk_sum_matches_numpy():
    from topobench.data.utils.trawl.sampling import _numpy_sum

    generator = np.random.default_rng(0)
    for _ in range(5000):
        size = int(generator.integers(0, 140))
        values = generator.random(size) * 10.0 ** generator.integers(-9, 9)
        assert _numpy_sum(values.tolist()) == float(np.sum(values))


def test_fast_walks_match_generator_choice():
    from topobench.data.utils.trawl.sampling import simulate_nbrw_sparse

    def reference(neighbors, weights, start, length, rng):
        # The original per-step implementation using Generator.choice.
        current, previous, path = start, -1, [start]
        for _ in range(length - 1):
            nbrs = neighbors[current]
            raw = np.asarray(weights[current], dtype=np.float64)
            probs = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            probs[[i for i, n in enumerate(nbrs) if n == previous]] = 0.0
            if probs.sum() <= 0.0:
                probs = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            if probs.sum() <= 0.0:
                probs = np.full(len(nbrs), 1.0 / len(nbrs))
            probs /= probs.sum()
            previous, current = (
                current,
                int(nbrs[int(rng.choice(len(nbrs), p=probs))]),
            )
            path.append(current)
        return path

    generator = np.random.default_rng(1)
    for _ in range(300):
        size = int(generator.integers(2, 30))
        neighbors = [
            [
                int(n)
                for n in generator.choice(size, int(generator.integers(1, 14)))
            ]
            for _ in range(size)
        ]
        weights = [
            (
                generator.random(len(n)) * generator.choice([1, 1, 0], len(n))
            ).tolist()
            for n in neighbors
        ]
        seed, start = (
            int(generator.integers(1e9)),
            int(generator.integers(size)),
        )
        fast_rng, slow_rng = (
            np.random.default_rng(seed),
            np.random.default_rng(seed),
        )
        assert simulate_nbrw_sparse(
            neighbors, weights, start, 32, fast_rng
        ) == reference(neighbors, weights, start, 32, slow_rng)
        assert fast_rng.bit_generator.state == slow_rng.bit_generator.state


def test_compiled_walks_match_pure_python():
    from topobench.data.utils.trawl import fast_walks
    from topobench.data.utils.trawl.sampling import (
        prepare_walk_rows,
        simulate_nbrw_sparse,
    )

    if not fast_walks.available:
        pytest.skip("numba is not installed")
    generator = np.random.default_rng(3)
    for _ in range(300):
        size = int(generator.integers(1, 30))
        neighbors = [
            [
                int(n)
                for n in generator.choice(size, int(generator.integers(0, 20)))
            ]
            for _ in range(size)
        ]
        weights = [
            (
                generator.random(len(n))
                * generator.choice([1.0, 1.0, 0.0, -1.0, np.nan], len(n))
            ).tolist()
            for n in neighbors
        ]
        rows = prepare_walk_rows(neighbors, weights)
        seed, start = (
            int(generator.integers(1e9)),
            int(generator.integers(size)),
        )
        fast, slow = np.random.default_rng(seed), np.random.default_rng(seed)
        # A plain list forces the pure-Python path.
        assert simulate_nbrw_sparse(
            neighbors, weights, start, 33, fast, rows=rows
        ) == simulate_nbrw_sparse(
            neighbors, weights, start, 33, slow, rows=list(rows)
        )
        assert fast.bit_generator.state == slow.bit_generator.state
