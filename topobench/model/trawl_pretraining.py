"""Optional masked reconstruction before ordinary TopoBench supervised fit."""

import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F

from topobench.model.model import HostBatchTransferMixin


class FullPrecisionValidation:
    """Optionally run validation steps with autocast disabled."""

    evaluation_autocast = True

    def validation_step(self, batch, batch_idx):
        """Compute the validation loss, disabling autocast if configured.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Validation batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        if self.evaluation_autocast:
            return self._step(batch, validation=True, batch_idx=batch_idx)
        with torch.autocast(self.device.type, enabled=False):
            return self._step(batch, validation=True, batch_idx=batch_idx)


def validation_frequency(trainer):
    """Return epochs between validation checks, for plateau schedulers.

    A plateau scheduler stepped on epochs without validation either fails
    (the monitored metric is absent) or reuses a stale value.

    Parameters
    ----------
    trainer : lightning.Trainer or None
        Attached trainer, if any.

    Returns
    -------
    int
        The trainer's ``check_val_every_n_epoch``, or 1 when unavailable.
    """
    if trainer is None:
        return 1
    return int(getattr(trainer, "check_val_every_n_epoch", None) or 1)


class PretrainingCheckpoint(ModelCheckpoint):
    """Best-validation checkpoint with a minimum improvement.

    ``min_delta`` matches ``EarlyStopping`` so the restored encoder is the
    one that last reset patience. Resuming into a different run directory
    keeps the earlier best instead of letting Lightning discard its score.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to ``ModelCheckpoint``.
    min_delta : float, optional
        Minimum improvement of the monitored metric that counts as better
        (default: 0.0).
    resume_dir : str or pathlib.Path, optional
        Directory of the resumed run, searched for the earlier best
        checkpoint (default: None).
    **kwargs : dict
        Keyword arguments forwarded to ``ModelCheckpoint``.
    """

    def __init__(self, *args, min_delta=0.0, resume_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.min_delta = float(min_delta)
        self.resume_dir = resume_dir

    def check_monitor_top_k(self, trainer, current=None):
        """Check whether ``current`` improves on the best by ``min_delta``.

        Parameters
        ----------
        trainer : lightning.Trainer
            Trainer running the fit.
        current : torch.Tensor, optional
            Current value of the monitored metric (default: None).

        Returns
        -------
        bool
            Whether a checkpoint should be saved.
        """
        if (
            current is None
            or self.min_delta == 0
            or self.save_top_k == -1
            or len(self.best_k_models) < self.save_top_k
        ):
            return super().check_monitor_top_k(trainer, current)
        best = self.best_k_models[self.kth_best_model_path]
        improved = (
            current < best - self.min_delta
            if self.mode == "min"
            else current > best + self.min_delta
        )
        return trainer.strategy.reduce_boolean_decision(bool(improved))

    def load_state_dict(self, state_dict):
        """Restore callback state and carry over the previous best checkpoint.

        When resuming into a different directory, the earlier best checkpoint
        is copied into ``dirpath`` so its score is kept.

        Parameters
        ----------
        state_dict : dict
            Callback state saved by ``ModelCheckpoint``.
        """
        super().load_state_dict(state_dict)
        previous = state_dict.get("dirpath")
        best = state_dict.get("best_model_path")
        score = state_dict.get("best_model_score")
        if previous == self.dirpath or not best or score is None:
            return
        name = Path(best).name
        candidates = [Path(best)]
        if self.resume_dir is not None:
            candidates.append(Path(self.resume_dir) / name)
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"Resumed pretraining best checkpoint {name} was not found; "
                "keep it next to the resumed last checkpoint"
            )
        target = Path(self.dirpath) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        self.best_model_path = self.kth_best_model_path = str(target)
        self.best_model_score = self.kth_value = score
        self.best_k_models = {str(target): score}


class TRAWLPretrainer(
    FullPrecisionValidation, HostBatchTransferMixin, LightningModule
):
    """Self-supervised rank features, colors and masked connectivity.

    Only training examples update weights. Validation uses a deterministic
    mask. Labels are never accessed. The supervised head is not optimized.

    Parameters
    ----------
    backbone : torch.nn.Module
        TRAWL backbone to pretrain.
    lr : float, optional
        AdamW learning rate (default: 1e-4).
    weight_decay : float, optional
        AdamW weight decay (default: 1e-3).
    mask_probability : float, optional
        Probability of masking each state and relation, in (0, 1]
        (default: 0.15).
    objectives : dict, optional
        Loss weights keyed by ``"features"``, ``"colors"`` and
        ``"topology"`` (default: None, meaning ``{"features": 1.0}``).
    feature_encoder : torch.nn.Module, optional
        Feature encoder applied before the backbone (default: None, meaning
        identity).
    target_widths : list of int, optional
        Feature reconstruction width per rank; defaults to the input width
        of each backbone feature layer (default: None).
    """

    def __init__(
        self,
        backbone,
        lr=1e-4,
        weight_decay=1e-3,
        mask_probability=0.15,
        objectives=None,
        feature_encoder=None,
        target_widths=None,
    ):
        super().__init__()
        if not 0 < mask_probability <= 1:
            raise ValueError("mask_probability must be in (0, 1]")
        self.backbone = backbone
        self.feature_encoder = (
            feature_encoder if feature_encoder is not None else nn.Identity()
        )
        self.lr, self.weight_decay = lr, weight_decay
        self.mask_probability = mask_probability
        self.objectives = dict(objectives or {"features": 1.0})
        if any(
            key not in {"features", "colors", "topology"}
            for key in self.objectives
        ):
            raise ValueError("Unknown pretraining objective")
        if not any(value > 0 for value in self.objectives.values()):
            raise ValueError(
                "At least one pretraining objective must be positive"
            )
        self.decoders = nn.ModuleList(
            [
                nn.Linear(
                    backbone.hidden_dim,
                    target_widths[rank]
                    if target_widths is not None
                    else layer.in_features,
                )
                for rank, layer in enumerate(backbone.features)
            ]
        )
        self.colors = nn.Linear(
            backbone.hidden_dim,
            backbone.color.num_embeddings
            if backbone.color is not None
            else backbone.max_rank + 1,
        )
        self.links = nn.Linear(
            backbone.hidden_dim, backbone.hidden_dim, bias=False
        )

    def _step(self, batch, validation=False, batch_idx=0):
        """Mask the batch, encode it and compute the weighted pretraining loss.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch of TRAWL-transformed graphs.
        validation : bool, optional
            If True, use the fixed validation mask and log the validation
            loss (default: False).
        batch_idx : int, optional
            Index of the batch, used to seed the mask (default: 0).

        Returns
        -------
        torch.Tensor
            Pretraining loss.
        """
        masked = batch.clone()
        # Masking edits device fields below; host copies would be stale and
        # could leak held-out connectivity into walk sampling.
        if "trawl_host" in masked:
            del masked.trawl_host
        generator = torch.Generator(device=self.device)
        # Distinct per microbatch; validation masks are fixed across epochs.
        generator.manual_seed(
            self.backbone.seed
            + batch_idx * 7919
            + (0 if validation else 1 + (self.current_epoch + 1) * 1000003)
        )
        masks = []
        for rank in range(self.backbone.max_rank + 1):
            key = f"trawl_signal_{rank}"
            mask = (
                torch.rand(
                    len(batch[key]), device=self.device, generator=generator
                )
                < self.mask_probability
            )
            if len(mask) and not bool(mask.any()):
                mask[0] = True
            masks.append(mask)
            masked[key][mask] = 0
        # State layout is graph-major, whereas each feature field is rank-major.
        offsets = [0] * len(masks)
        state_mask, targets, ranks = [], [], []
        for sizes in batch.trawl_counts.tolist():
            for rank, size in enumerate(sizes):
                start = offsets[rank]
                state_mask.append(masks[rank][start : start + size])
                targets.append(
                    (rank, batch[f"trawl_signal_{rank}"][start : start + size])
                )
                ranks.extend([rank] * size)
                offsets[rank] += size
        state_mask = torch.cat(state_mask)
        # Hide structural encodings at masked states to avoid a direct shortcut.
        masked.trawl_pe[state_mask] = 0
        if self.objectives.get("colors", 0):
            masked.trawl_colors[state_mask] = 0
        link_examples = []
        if self.objectives.get("topology", 0):
            slices = getattr(batch, "_slice_dict", {}).get(
                "trawl_edges", [0, len(batch.trawl_edges)]
            )
            offset = 0
            for graph_id, sizes in enumerate(batch.trawl_counts.tolist()):
                start, stop = int(slices[graph_id]), int(slices[graph_id + 1])
                edges = batch.trawl_edges[start:stop, :2]
                unique = {(int(a), int(b)) for a, b in edges.tolist()}
                # Mask both directions together, including parallel relations.
                hidden = {
                    pair
                    for pair in unique
                    if torch.rand((), generator=generator, device=self.device)
                    < self.mask_probability
                }
                hidden |= {(b, a) for a, b in hidden}
                for row, pair in enumerate(edges.tolist()):
                    if tuple(pair) in hidden:
                        masked.trawl_weights[start + row] = 0
                n = sum(sizes)
                positives = list(unique & hidden)
                negatives = set()
                # Bounded rejection sampling does not allocate an n x n array.
                for _ in range(10 * max(len(positives), 1)):
                    a, b = torch.randint(
                        n, (2,), generator=generator, device=self.device
                    ).tolist()
                    if a != b and (a, b) not in unique:
                        negatives.add((a, b))
                    if len(negatives) >= len(positives):
                        break
                if positives and negatives:
                    pairs = positives + list(negatives)
                    indices = torch.tensor(pairs, device=self.device) + offset
                    labels = torch.tensor(
                        [1.0] * len(positives) + [0.0] * len(negatives),
                        device=self.device,
                    )
                    link_examples.append((indices, labels))
                offset += n
            # Precomputed spectral features could reveal held-out edges.
            masked.trawl_pe.zero_()
        for rank in getattr(self.feature_encoder, "ranks", []):
            masked[f"x_{rank}"] = masked[f"trawl_signal_{rank}"]
        out = self.backbone(self.feature_encoder(masked))
        embeddings = out["cell_embeddings"]
        loss = embeddings.sum() * 0
        if self.objectives.get("features", 0):
            losses, offset = [], 0
            for rank, target in targets:
                select = state_mask[offset : offset + len(target)]
                if bool(select.any()):
                    predictions = self.decoders[rank](
                        embeddings[offset : offset + len(target)][select]
                    )
                    losses.append(F.mse_loss(predictions, target[select]))
                offset += len(target)
            loss = (
                loss + self.objectives["features"] * torch.stack(losses).mean()
            )
        if self.objectives.get("colors", 0) and bool(state_mask.any()):
            if int(batch.trawl_colors.max()) >= self.colors.out_features:
                raise ValueError(
                    "Color pretraining needs model.backbone.num_colors to cover "
                    "every transformed color ID"
                )
            loss = loss + self.objectives["colors"] * F.cross_entropy(
                self.colors(embeddings[state_mask]),
                batch.trawl_colors[state_mask],
            )
        if link_examples:
            predictions, labels = [], []
            for indices, target in link_examples:
                a, b = indices.unbind(1)
                predictions.append(
                    (self.links(embeddings[a]) * embeddings[b]).sum(-1)
                    / embeddings.shape[-1] ** 0.5
                )
                labels.append(target)
            loss = loss + self.objectives[
                "topology"
            ] * F.binary_cross_entropy_with_logits(
                torch.cat(predictions), torch.cat(labels)
            )
        self.log(
            "pretrain/val_loss" if validation else "pretrain/train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            batch_size=len(batch.trawl_counts),
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Compute the training loss.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Training batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        return self._step(batch, batch_idx=batch_idx)

    def configure_optimizers(self):
        """Configure AdamW with a validation-loss plateau scheduler.

        Returns
        -------
        dict
            Optimizer and learning-rate scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
                ),
                "monitor": "pretrain/val_loss",
                "interval": "epoch",
                "frequency": validation_frequency(self._trainer),
            },
        }


def run_pretraining(
    model, datamodule, config, trainer_config, output_dir=None
):
    """Train, restore the best encoder, then leave supervised fitting to runner.

    Parameters
    ----------
    model : TBModel
        Model whose TRAWL backbone and feature encoder are pretrained in
        place.
    datamodule : lightning.LightningDataModule
        Data module providing the training and validation loaders.
    config : omegaconf.DictConfig
        Pretraining options (``lr``, ``weight_decay``, ``mask_probability``,
        ``objectives``, ``max_epochs``, ``patience`` and optional
        ``min_delta``, ``ckpt_path`` and ``reset_head``).
    trainer_config : omegaconf.DictConfig
        Hydra config used to instantiate the pretraining trainer.
    output_dir : str or pathlib.Path, optional
        Run directory; checkpoints and metrics are written to its
        ``pretraining`` subdirectory, otherwise to a temporary directory
        (default: None).
    """
    from topobench.nn.backbones.combinatorial.trawl import TRAWL

    if not isinstance(model.backbone, TRAWL):
        raise ValueError(
            "The TRAWL pretraining config requires a TRAWL backbone"
        )
    from topobench.nn.encoders.trawl import TRAWLFeatureEncoder

    encoder = model.feature_encoder
    if not isinstance(encoder, (nn.Identity, TRAWLFeatureEncoder)):
        raise ValueError(
            "Wrap a native rank encoder in TRAWLFeatureEncoder for pretraining"
        )
    module = TRAWLPretrainer(
        model.backbone,
        lr=config.lr,
        weight_decay=config.weight_decay,
        mask_probability=config.mask_probability,
        objectives=config.objectives,
        feature_encoder=encoder,
        target_widths=[
            datamodule.dataset_train.data_lst[0][f"trawl_signal_{rank}"].shape[
                1
            ]
            for rank in range(model.backbone.max_rank + 1)
        ],
    )
    destination = (
        Path(output_dir) / "pretraining" if output_dir is not None else None
    )
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    context = (
        nullcontext(str(destination))
        if destination is not None
        else tempfile.TemporaryDirectory(prefix="topobench-trawl-ssl-")
    )
    module.evaluation_autocast = getattr(model, "evaluation_autocast", True)
    resume = config.get("ckpt_path")
    min_delta = float(config.get("min_delta", 0.0))
    with context as directory:
        checkpoint = PretrainingCheckpoint(
            dirpath=directory,
            monitor="pretrain/val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            min_delta=min_delta,
            resume_dir=Path(resume).parent if resume else None,
        )
        trainer = hydra.utils.instantiate(
            trainer_config,
            max_epochs=config.max_epochs,
            logger=(
                CSVLogger(save_dir=directory, name="metrics")
                if destination is not None
                else False
            ),
            callbacks=[
                checkpoint,
                EarlyStopping(
                    monitor="pretrain/val_loss",
                    patience=config.patience,
                    min_delta=min_delta,
                ),
            ],
            num_sanity_val_steps=0,
            enable_checkpointing=True,
        )
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume)
        if checkpoint.best_model_path:
            state = torch.load(
                checkpoint.best_model_path,
                map_location="cpu",
                weights_only=False,
            )
            module.load_state_dict(state["state_dict"])
    model.backbone.reset_sampling_step()
    if config.get("reset_head", False):
        for layer in model.readout.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
