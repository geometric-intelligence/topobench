"""Model checkpoint ranking that survives a spawned training process."""

import json
from pathlib import Path, PureWindowsPath

import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class RankedModelCheckpoint(ModelCheckpoint):
    """Persist the final top-K index for evaluation in a spawn parent.

    Lightning transfers the best path out of spawned workers, but not the full
    ranking. Keep an explicit index next to the checkpoints rather than infer
    their validation scores from filenames or unrelated earlier runs.
    """

    def _select_latest_kth(self):
        """Evict the most recent of several tied worst checkpoints.

        Lightning evicts the earliest tied entry; here ties are ranked by
        earlier epoch, so later tied checkpoints leave first.
        """
        if self.save_top_k < 1 or len(self.best_k_models) < self.save_top_k:
            return
        worst = (max if self.mode == "min" else min)(
            float(score) for score in self.best_k_models.values()
        )
        tied = [
            path
            for path, score in self.best_k_models.items()
            if float(score) == worst
        ]
        self.kth_best_model_path = tied[-1]
        self.kth_value = self.best_k_models[tied[-1]]

    def _update_best_and_save(self, current, trainer, monitor_candidates):
        """Update the top-K ranking, evicting the latest tied worst checkpoint.

        Parameters
        ----------
        current : torch.Tensor
            Current value of the monitored metric.
        trainer : lightning.pytorch.Trainer
            The running trainer.
        monitor_candidates : dict
            Metrics available for monitoring and checkpoint naming.
        """
        self._select_latest_kth()
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self._select_latest_kth()

    def on_fit_end(self, trainer, pl_module):
        """Write the final top-K index next to the checkpoints.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The running trainer.
        pl_module : lightning.pytorch.LightningModule
            The trained module.
        """
        super().on_fit_end(trainer, pl_module)
        if not trainer.is_global_zero or not self.best_model_path:
            return
        destination = Path(self.dirpath) / "checkpoint_index.json"
        payload = {
            "best_model_path": self.best_model_path,
            "best_model_score": (
                float(self.best_model_score)
                if self.best_model_score is not None
                else None
            ),
            "mode": self.mode,
            "monitor": self.monitor,
            "best_k_models": {
                str(path): float(score)
                for path, score in self.best_k_models.items()
            },
        }
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(destination)

    def restore_ranking(self, relocate=False):
        """Recover this run's ranking if a launcher returned only its best path.

        Parameters
        ----------
        relocate : bool, optional
            If True, match checkpoints by file name and resolve them relative
            to the directory of ``best_model_path`` (default: False).
        """
        if self.best_k_models or not self.best_model_path:
            return
        source = Path(self.best_model_path).parent / "checkpoint_index.json"
        if not source.is_file():
            return
        payload = json.loads(source.read_text())
        saved_best = PureWindowsPath(payload["best_model_path"])
        if (
            (
                saved_best.name != Path(self.best_model_path).name
                if relocate
                else payload["best_model_path"] != self.best_model_path
            )
            or payload["mode"] != self.mode
            or payload["monitor"] != self.monitor
        ):
            raise ValueError("Checkpoint index does not match this run")
        self.best_model_score = (
            torch.tensor(payload["best_model_score"])
            if payload["best_model_score"] is not None
            else None
        )
        self.best_k_models = {
            str(source.parent / PureWindowsPath(path).name)
            if relocate
            else path: torch.tensor(score)
            for path, score in payload["best_k_models"].items()
        }
        if relocate and not all(
            Path(path).is_file() for path in self.best_k_models
        ):
            raise FileNotFoundError(
                "The relocated top-K checkpoint set is incomplete"
            )
