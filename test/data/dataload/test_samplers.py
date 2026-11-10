"""Tests for the epoch-seeded and unpadded distributed samplers."""

import torch

from topobench.dataloader.samplers import (
    EpochRandomSampler,
    UnpaddedDistributedSampler,
)


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


def test_unpadded_sampler_visits_each_sample_once():
    dataset = list(range(11))
    evaluation = [
        list(UnpaddedDistributedSampler(dataset, num_replicas=2, rank=rank))
        for rank in range(2)
    ]
    assert sorted(evaluation[0] + evaluation[1]) == dataset
    assert len(UnpaddedDistributedSampler(dataset, 2, 1)) == 5


def test_dataloader_uses_order_seed():
    from topobench.dataloader import TBDataloader

    datamodule = TBDataloader(
        dataset_train=list(range(6)),
        dataset_val=list(range(2)),
        dataset_test=list(range(2)),
        batch_size=2,
        order_seed=7,
    )
    sampler = datamodule.train_dataloader().sampler
    assert isinstance(sampler, EpochRandomSampler) and sampler.seed == 7
    plain = TBDataloader(list(range(6)), list(range(2)), list(range(2)))
    assert not isinstance(plain.train_dataloader().sampler, EpochRandomSampler)
