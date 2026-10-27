"""Tests for RankedModelCheckpoint."""

import torch

from topobench.callbacks.model_checkpoint import RankedModelCheckpoint


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


def test_ranking_index_survives_relocation(tmp_path):
    source = tmp_path / "run" / "checkpoints"
    source.mkdir(parents=True)
    paths = [source / f"epoch_{i:03d}.ckpt" for i in range(2)]
    for path in paths:
        path.write_bytes(b"checkpoint")
    callback = RankedModelCheckpoint(
        dirpath=source, monitor="val/accuracy", mode="max", save_top_k=2
    )
    callback.best_model_path = str(paths[1])
    callback.best_model_score = torch.tensor(0.8)
    callback.best_k_models = {
        str(paths[0]): torch.tensor(0.7),
        str(paths[1]): torch.tensor(0.8),
    }

    class Trainer:
        is_global_zero = True

    callback.on_fit_end(Trainer(), None)
    assert (source / "checkpoint_index.json").is_file()
    moved = tmp_path / "moved"
    source.rename(moved)
    replay = RankedModelCheckpoint(
        dirpath=moved, monitor="val/accuracy", mode="max"
    )
    replay.best_model_path = str(moved / paths[1].name)
    replay.restore_ranking(relocate=True)
    assert set(replay.best_k_models) == {
        str(moved / path.name) for path in paths
    }
    assert float(replay.best_model_score) == float(torch.tensor(0.8))
