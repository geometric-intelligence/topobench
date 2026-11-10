"""Optional pretraining procedures run before supervised fitting.

A pretraining config (``configs/pretraining/<name>.yaml``) targets a
callable with ``_partial_: true``. When the ``pretraining`` group is set, the
runner calls it as ``fn(model=..., datamodule=..., trainer_config=...,
output_dir=...)`` before fitting; it must update the model in place.
"""
