"""Composable schedules for the categorical TRAWL experiments."""

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def warmup_cosine(
    optimizer, epochs, warmup_epochs=10, start_factor=0.01, eta_min=1e-6
):
    """Construct a linear-warmup then cosine epoch schedule.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule.
    epochs : int
        Total number of epochs.
    warmup_epochs : int, optional
        Number of linear warmup epochs (default: 10).
    start_factor : float, optional
        Learning-rate multiplier at the start of warmup (default: 0.01).
    eta_min : float, optional
        Minimum learning rate of the cosine phase (default: 1e-6).

    Returns
    -------
    torch.optim.lr_scheduler.SequentialLR
        Warmup followed by cosine annealing.
    """
    if epochs <= warmup_epochs or warmup_epochs < 1:
        raise ValueError("Require 0 < warmup_epochs < epochs")
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
            ),
        ],
        milestones=[warmup_epochs],
    )
