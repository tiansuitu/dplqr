"""Optional dense observation of the exact Experiment 1 fit, without changing its source.

The original runner has strict source fingerprints. Binding a recorder in a copy
of the fit function's globals reuses its exact code while preserving old caches
and avoiding mutable, process-wide monkeypatching. Only the observer changes.
"""
from __future__ import annotations

from types import FunctionType
import numpy as np
import pandas as pd
import torch
import run_trajectory as trajectory


class DenseRecorder(trajectory.TrajectoryRecorder):
    """Record all epoch metrics; retain weight archives only at original checkpoints."""

    def observe(self, optimizer_steps):
        directory = self.directory
        self.directory = directory if self.epoch in trajectory.CHECKPOINTS else None
        try:
            super().observe(optimizer_steps)
        finally:
            self.directory = directory

        # Match train_dplqr_once(..., val_batch_size=128) and torchtuples'
        # unweighted mean of batch means, including its float32 arithmetic.
        # Do not construct a DataLoader: doing so consumes the training RNG.
        net = self.model.net
        modes = [(module, module.training) for module in net.modules()]
        rng_before = torch.get_rng_state().clone()
        try:
            net.eval()
            with torch.inference_mode():
                pair = self.inputs['validation']
                target = torch.tensor(self.data['y_validation'][:, None], dtype=torch.float32)
                losses = [self.model.loss(net(*(x[start:start+128] for x in pair)), target[start:start+128])
                          for start in range(0, len(target), 128)]
                criterion = float(torch.stack(losses).mean().item())
        finally:
            for module, mode in modes:
                module.training = mode
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert trajectory.state_hash(net.state_dict()) == self.last_state_sha256
        assert np.isfinite(criterion)
        self.rows[-1].update(stopping_validation_check_loss=criterion,
                             state_sha256=self.last_state_sha256)
        if directory is not None and self.epoch % 100 == 0:
            trajectory.atomic_csv(pd.DataFrame(self.rows), directory / 'partial_trajectory.csv')
            print(f'Dense path case={self.case} replicate={self.replicate}: epoch {self.epoch}', flush=True)


def fit_dense(config, case, replicate, directory=None):
    """Call the original fit bytecode with a private, recording-only callback binding."""
    if tuple(config.checkpoints) != tuple(range(1, config.max_epochs + 1)):
        raise ValueError('Dense recording requires every integer epoch through the ceiling')
    original = trajectory.fit_one
    fit_globals = dict(original.__globals__, TrajectoryRecorder=DenseRecorder)
    fit = FunctionType(original.__code__, fit_globals, original.__name__, original.__defaults__, original.__closure__)
    return fit(config, case, replicate, directory)
