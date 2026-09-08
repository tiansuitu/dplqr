"""Validation-only retrospective replay of the main notebook's stopping rule.

The authoritative implementation is ``EarlyStopInMemory`` in
``results/2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb``.
This function deliberately accepts no parameter errors, truth, test losses,
model predictions, or oracle epochs.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence


MIN_DELTA = 0.0
RESTORE_BEST = True


def select_stopping_epoch(
    validation_losses: Sequence[float], patience: int, max_epochs: int
) -> dict:
    """Return the checkpoint selected by the repository's practical callback.

    Epochs are one-based. An improvement is strictly ``loss < best_loss``;
    a tie neither resets patience nor replaces the saved best weights. The
    patience counter is zero on an improvement and increases once for each
    following epoch without improvement. The first epoch whose counter is
    at least ``patience`` is the stop epoch. Best weights are restored both
    after a patience stop and after reaching the epoch cap.

    If patience triggers on the final allowed epoch, ``stopping_reason`` is
    ``early_stopping`` because the callback did return a stop signal then.
    A complete prefix through ``max_epochs`` is required: sparse checkpoints
    cannot reconstruct this rule. Use the original callback's float32,
    equal-batch-weighted validation loss, not a test or oracle metric.
    """
    for name, value in (("patience", patience), ("max_epochs", max_epochs)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(validation_losses) < max_epochs:
        raise ValueError("A validation loss for every epoch through max_epochs is required")

    best_loss = math.inf
    best_epoch = 0
    since_best = 0
    stopping_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        loss = float(validation_losses[epoch - 1])
        if not math.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        if loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, since_best = loss, epoch, 0
        else:
            since_best += 1
        if since_best >= patience:
            stopping_reason = "early_stopping"
            break

    return {
        "patience": int(patience),
        "max_epochs": int(max_epochs),
        "stop_epoch": epoch,
        "selected_epoch": best_epoch,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stopping_reason": stopping_reason,
        "restore_best": RESTORE_BEST,
        "min_delta": MIN_DELTA,
    }
