"""Small scientific/parity and checkpoint checks; never fits benchmarks.

Run after the four-replication smoke test. One unchanged-source DPLQR fit
is repeated intentionally to test numerical equality, not added to MC data.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

from source_bridge import HERE, ROOT, load_original
from run_normality import atomic_json, fit_replication


def verify(output):
    start = time.perf_counter()
    meta = json.loads((output / "run_config.json").read_text())
    fingerprint = meta["identity_sha256"]
    hp = meta["identity"]["hp"]
    seed = meta["identity"]["seed"]
    torch.set_num_threads(meta["identity"]["threads"])
    torch.use_deterministic_algorithms(True)
    original = load_original()
    case, n, rep = 3, 1000, 1
    directory = output / "replications" / f"case_{case}_n_{n}_rep_{rep:04d}"
    previous = json.loads((directory / "result.json").read_text())["row"]
    rng = np.random.default_rng(previous["setting_seed"])
    x, z, y, _ = original.generate_dataset(n, case, rng)
    order = rng.permutation(n)
    tr, va = order[:int(.8*n)], order[int(.8*n):]
    # No benchmark module/definition was imported or executed.
    assert "fit_lqr" not in original.namespace and "smf" not in original.namespace
    theta, se, _, _, _, epochs, density, projection_epochs = original.fit_dplqr(
        x[tr], z[tr], y[tr], x[va], z[va], y[va], x[va], z[va], .5, hp, previous["fit_seed"])
    np.testing.assert_array_equal(theta.astype(float), [previous["theta1"], previous["theta2"]])
    np.testing.assert_array_equal(se, [previous["se_theta1"], previous["se_theta2"]])
    assert density == previous["density_zero"] and epochs == previous["epochs_run"]
    assert projection_epochs == json.loads(previous["projection_epochs"])

    # Prove all three completed network fits are recovered without fitting.
    def must_not_fit(*args, **kwargs):
        raise AssertionError("Resume attempted to refit a completed network")

    original.select_dplqr = must_not_fit
    original.fit_projection = must_not_fit
    resumed = fit_replication(original, hp, case, n, rep, seed, directory, fingerprint)
    science = ["theta1", "theta2", "se_theta1", "se_theta2", "density_zero", "t_theta1",
               "covered_theta1", "root_n_error_theta1", "root_n_train_error_theta1",
               "epochs_run", "best_epoch", "projection_epochs", "projection_best_epochs", "data_sha256"]
    assert all(resumed[key] == previous[key] for key in science)

    # The R helper contains the exact original density function and nothing
    # from the original simulation/benchmark entry point is executed here.
    old_r = (ROOT / "results/2026-09-04-homoscedastic-simulation/simulate_homoscedastic.R").read_text()
    new_r = (HERE / "density_only.R").read_text()
    pattern = r"residual_density_zero <- function\(residuals\) \{.*?\n\}"
    assert re.search(pattern, old_r, re.S).group() == re.search(pattern, new_r, re.S).group()
    result = dict(status="passed", numerical_parity="bit-for-bit theta1, theta2, both SEs, density and epochs",
                  checkpoint_reuse="main and both projection fits prohibited; identical scientific outputs recovered",
                  r_density="function text exactly matches September 4; base R only",
                  benchmark_definitions_loaded=False, extra_validation_dplqr_fits=1,
                  extra_validation_auxiliary_fits=2, elapsed_seconds=time.perf_counter()-start)
    atomic_json(result, HERE / "verification.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "pilot")
