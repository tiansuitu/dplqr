"""Independently audit saved DPLQR trajectories without fitting a network.

Run: python scripts/verify_outputs.py OUTPUT_DIR
Writes OUTPUT_DIR/verification.json after checking saved artifacts and rebuilding
selected diagnostics from their checkpoint weights and reproducible datasets.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

import run_trajectory as rt


DIAGNOSTICS = (
    "theta_error_l2", "train_check_loss", "validation_check_loss", "test_check_loss",
    "m_l2_error", "q_l2_error", "linear_error_mse", "nonlinear_error_mse",
    "cross_error_moment", "cancellation_fraction", "q_mse", "decomposition_residual",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-11, rtol=1e-11, equal_nan=False):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=equal_nan, err_msg=label)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_raw(raw, config):
    keys = ["case", "replicate", "epoch"]
    expected = set(itertools.product(config.cases, range(1, config.repetitions + 1), config.checkpoints))
    actual = set(raw[keys].itertuples(index=False, name=None))
    require(len(raw) == len(expected) and actual == expected, "Incomplete or unexpected checkpoint grid")
    require(not raw.duplicated(keys).any(), "Duplicate checkpoint rows")
    numeric = raw.select_dtypes(include="number")
    require(np.isfinite(numeric.to_numpy()).all(), "Nonfinite numeric raw values")
    counts = {"n": config.n, "n_train": int(0.8 * config.n),
              "n_validation": config.n - int(0.8 * config.n),
              "n_test": config.test_size, "n_eval": config.eval_size,
              "tau": config.tau, "learning_rate": config.learning_rate,
              "model_instance": 1, "fit_calls": 1}
    for name, value in counts.items():
        require(raw[name].eq(value).all(), f"Incorrect {name} label")
    for case in config.cases:
        require(raw.loc[raw.case.eq(case), "case_name"].eq(rt.CASE_NAMES[case]).all(), "Incorrect case label")
    theta = raw[["theta_hat_1", "theta_hat_2"]].to_numpy(float)
    truth = np.array([1.0, -1.0])
    errors = theta - truth
    for j in (1, 2):
        require(raw[f"theta_true_{j}"].eq(truth[j - 1]).all(), "Incorrect true theta")
        close(raw[f"theta_error_{j}"], errors[:, j - 1], "Signed theta error")
        close(raw[f"abs_theta_error_{j}"], np.abs(errors[:, j - 1]), "Absolute theta error")
    close(raw.theta_error_l2, np.linalg.norm(errors, axis=1), "Vector theta L2")
    close(raw.m_l2_error ** 2, raw.nonlinear_error_mse, "Nuisance L2 vs MSE")
    close(raw.q_l2_error ** 2, raw.q_mse, "Quantile L2 vs MSE")
    residual = raw.q_mse - raw.linear_error_mse - raw.nonlinear_error_mse - 2 * raw.cross_error_moment
    close(raw.decomposition_residual, residual, "Recorded error-decomposition residual")
    close(residual, np.zeros(len(raw)), "Error decomposition", atol=1e-9, rtol=0)
    denominator = raw.linear_error_mse.to_numpy() + raw.nonlinear_error_mse.to_numpy()
    cancellation = np.divide(-2 * raw.cross_error_moment.to_numpy(), denominator,
                             out=np.zeros(len(raw)), where=denominator > 0)
    close(raw.cancellation_fraction, cancellation, "Cancellation fraction")
    steps_per_epoch = math.ceil(int(0.8 * config.n) / config.batch_size)
    require(raw.optimizer_steps.eq(raw.epoch * steps_per_epoch).all(), "Discontinuous optimizer steps")
    for (case, replicate), group in raw.groupby(["case", "replicate"]):
        for name, value in rt.seed_scheme(config, int(case), int(replicate)).items():
            require(group[name].eq(value).all(), f"Incorrect {name}")
    return dict(raw_rows=len(raw), complete_unique_grid=True, all_numeric_raw_values_finite=True,
                correct_labels_and_errors=True, optimizer_steps_per_epoch=steps_per_epoch,
                max_abs_decomposition_residual=float(np.abs(residual).max()))


def verify_summaries(output, raw):
    theta_summary = read_csv(output / "monte_carlo_summary.csv")
    diagnostic_summary = read_csv(output / "diagnostic_summary.csv")
    groups = list(raw.groupby(["case", "epoch"]))
    require(len(theta_summary) == len(groups) * 2, "Unexpected coefficient-summary length")
    require(len(diagnostic_summary) == len(groups) * len(DIAGNOSTICS), "Unexpected diagnostic-summary length")
    require(not theta_summary.duplicated(["case", "epoch", "theta_component"]).any(), "Duplicate coefficient summary")
    require(not diagnostic_summary.duplicated(["case", "epoch", "metric"]).any(), "Duplicate diagnostic summary")
    for (case, epoch), group in groups:
        q = len(group)
        for j in (1, 2):
            row = theta_summary.loc[theta_summary.case.eq(case) & theta_summary.epoch.eq(epoch)
                                    & theta_summary.theta_component.eq(f"theta_{j}")]
            require(len(row) == 1, "Missing coefficient summary")
            row = row.iloc[0]
            estimates = group[f"theta_hat_{j}"].to_numpy(float)
            truth = (1.0, -1.0)[j - 1]
            error = estimates - truth
            sd = float(np.std(estimates, ddof=1)) if q > 1 else np.nan
            expected = dict(theta_true=truth, mean_theta=float(np.mean(estimates)), bias=float(np.mean(error)),
                            sd=sd, rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                            bias_mcse=sd / math.sqrt(q), repetitions=q)
            for name, value in expected.items():
                close(row[name], value, f"Coefficient summary {case}/{epoch}/{j}/{name}", equal_nan=True)
            require(bool(row.mc_sd_available) == (q > 1), "Incorrect MC SD availability")
        for metric in DIAGNOSTICS:
            row = diagnostic_summary.loc[diagnostic_summary.case.eq(case) & diagnostic_summary.epoch.eq(epoch)
                                         & diagnostic_summary.metric.eq(metric)]
            require(len(row) == 1, "Missing diagnostic summary")
            row = row.iloc[0]
            values = group[metric].to_numpy(float)
            sd = float(np.std(values, ddof=1)) if q > 1 else np.nan
            expected = dict(mean=float(np.mean(values)), sd=sd, mcse=sd / math.sqrt(q), repetitions=q)
            for name, value in expected.items():
                close(row[name], value, f"Diagnostic summary {case}/{epoch}/{metric}/{name}", equal_nan=True)
            require(bool(row.mc_sd_available) == (q > 1), "Incorrect diagnostic MC SD availability")
    return dict(coefficient_summary_rows=len(theta_summary), diagnostic_summary_rows=len(diagnostic_summary),
                independently_recomputed_bias_sd_rmse_and_mcse=True)


def verify_archives(output, raw, config):
    files_checked = 0
    paths_checked = 0
    steps_per_epoch = math.ceil(int(0.8 * config.n) / config.batch_size)
    for case, replicate in itertools.product(config.cases, range(1, config.repetitions + 1)):
        folder = output / "replicates" / f"case_{case}_rep_{replicate:04d}"
        audit = json_read(folder / "completion.json")
        require(audit["status"] == "complete", "Incomplete replicate audit")
        require(audit["case"] == case and audit["replicate"] == replicate, "Mislabeled replicate audit")
        require(audit["epochs_completed"] == config.max_epochs, "Path stopped before ceiling")
        require(audit["fit_calls"] == 1 and audit["model_instances"] == 1, "Multiple fits or models in a path")
        require(audit["optimizer_steps"] == config.max_epochs * steps_per_epoch, "Incorrect final optimizer count")
        for name in ("early_stopping", "validation_selection", "clipping"):
            require(audit[name] is False, f"Unexpected {name}")
        for name in ("observation_rng_unchanged", "observation_parameters_unchanged"):
            require(audit[name] is True, f"Failed {name}")
        require(audit["checkpoint_epochs"] == list(config.checkpoints), "Incorrect completed checkpoint schedule")
        checkpoints = audit["checkpoints"]
        require([checkpoint["epoch"] for checkpoint in checkpoints] == list(config.checkpoints), "Incomplete checkpoint manifest")
        previous = audit["initial_state_sha256"]
        for checkpoint in checkpoints:
            require(checkpoint["previous_checkpoint_state_sha256"] == previous, "Broken checkpoint state hash chain")
            require(checkpoint["optimizer_steps"] == checkpoint["epoch"] * steps_per_epoch, "Incorrect checkpoint optimizer step")
            filename = Path(checkpoint["file"])
            require(filename.name == str(filename), "Checkpoint manifest path escapes replicate folder")
            require(sha256(folder / filename) == checkpoint["sha256"], "Checkpoint archive hash mismatch")
            previous = checkpoint["state_sha256"]
            files_checked += 1
        require(previous == audit["final_state_sha256"], "Final state differs from last checkpoint")
        saved = read_csv(folder / "trajectory.csv")
        aggregate = raw.loc[raw.case.eq(case) & raw.replicate.eq(replicate)]
        pd.testing.assert_frame_equal(saved.reset_index(drop=True), aggregate.reset_index(drop=True), check_exact=True)
        paths_checked += 1
    return dict(continuous_paths_checked=paths_checked, checkpoint_archive_hashes_checked=files_checked,
                checkpoint_hash_chains_valid=True, all_paths_reached_requested_ceiling=True,
                no_early_stopping_restoration_or_clipping=True)


def verify_reloaded_metrics(output, raw, config):
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    checks = []
    epochs = sorted({config.checkpoints[0], config.checkpoints[-1]} | ({100} if 100 in config.checkpoints else set()))
    for case in config.cases:
        replicate = 1
        data, seeds, _ = rt.make_data(config, case, replicate)
        scaler, inputs = rt.prepare_inputs(data)
        folder = output / "replicates" / f"case_{case}_rep_{replicate:04d}"
        manifest = json_read(folder / "data_manifest.json")
        require(manifest["seeds"] == seeds and manifest["arrays_sha256"] == rt.array_hash(data), "Reconstructed dataset hash mismatch")
        rt.seed_all(seeds["init_seed"])
        net = rt.dqNetSparse(2, 8, torch.zeros((1, 2), dtype=torch.float32),
                            [config.depth, config.width], sparseRatio=0.5)
        net.linLinear.reset_parameters()
        audit = json_read(folder / "completion.json")
        require(rt.state_hash(net.state_dict()) == audit["initial_state_sha256"], "Initialization does not reproduce")
        for epoch in epochs:
            checkpoint = next(row for row in audit["checkpoints"] if row["epoch"] == epoch)
            state = torch.load(folder / checkpoint["file"], map_location="cpu", weights_only=True)
            require((state["case"], state["replicate"], state["epoch"]) == (case, replicate, epoch), "Saved checkpoint metadata differs")
            require(state["seeds"] == seeds, "Saved checkpoint seeds differ")
            net.load_state_dict(state["model_state_dict"])
            net.eval()
            require(rt.state_hash(net.state_dict()) == checkpoint["state_sha256"], "Saved state hash differs")
            close(state["scaler_mean"].numpy(), scaler.mean_, "Saved scaler mean", atol=0, rtol=0)
            close(state["scaler_scale"].numpy(), scaler.scale_, "Saved scaler scale", atol=0, rtol=0)
            optimizer_states = state["optimizer_state_dict"]["state"].values()
            expected_steps = epoch * math.ceil(len(data["y_train"]) / config.batch_size)
            require(state["optimizer_steps"] == expected_steps, "Saved payload optimizer count differs")
            require(all(int(item["step"]) == expected_steps for item in optimizer_states), "Saved Adam state has discontinuous steps")
            row = raw.loc[raw.case.eq(case) & raw.replicate.eq(replicate) & raw.epoch.eq(epoch)].iloc[0]
            recomputed = {}
            direct_loss_gap = 0.0
            with torch.inference_mode():
                theta = net.linLinear.weight.detach().numpy().reshape(-1).astype(float)
                for split in ("train", "validation", "test", "eval"):
                    x_tensor, z_tensor = inputs[split]
                    nuisance = net(torch.zeros_like(x_tensor), z_tensor).numpy().reshape(-1).astype(float)
                    prediction = data["x_" + split] @ theta + nuisance
                    direct = net(x_tensor, z_tensor).numpy().reshape(-1).astype(float)
                    close(prediction, direct, "Float64 branch assembly agrees with fitted network", atol=1e-5, rtol=2e-6)
                    if split != "eval":
                        residual = data["y_" + split] - prediction
                        check_loss = float(np.mean(np.maximum(config.tau * residual, (config.tau - 1) * residual)))
                        direct_residual = data["y_" + split] - direct
                        direct_loss = float(np.mean(np.maximum(config.tau * direct_residual, (config.tau - 1) * direct_residual)))
                        recomputed[split + "_check_loss"] = check_loss
                        direct_loss_gap = max(direct_loss_gap, abs(direct_loss - check_loss))
                    else:
                        theta_error = theta - np.array([1.0, -1.0])
                        linear_error = data["x_eval"] @ theta_error
                        nonlinear_error = nuisance - data["m_true_eval"]
                        true_q = data["x_eval"] @ np.array([1.0, -1.0]) + data["m_true_eval"]
                        q_error = prediction - true_q
                        recomputed.update(theta_error_l2=float(np.linalg.norm(theta_error)),
                                          m_l2_error=float(np.sqrt(np.mean(nonlinear_error ** 2))),
                                          q_l2_error=float(np.sqrt(np.mean(q_error ** 2))),
                                          linear_error_mse=float(np.mean(linear_error ** 2)),
                                          nonlinear_error_mse=float(np.mean(nonlinear_error ** 2)),
                                          cross_error_moment=float(np.mean(linear_error * nonlinear_error)),
                                          q_mse=float(np.mean(q_error ** 2)))
            close(theta, row[["theta_hat_1", "theta_hat_2"]].to_numpy(float), "Reloaded coefficients", atol=0, rtol=0)
            for name, value in recomputed.items():
                close(row[name], value, f"Reloaded metric {case}/{epoch}/{name}")
            checks.append(dict(case=case, replicate=replicate, epoch=epoch, optimizer_steps=expected_steps,
                               reloaded_metrics_match=True, max_direct_float32_check_loss_gap=direct_loss_gap))
    return dict(checkpoints_reconstructed_from_weights=len(checks), fixed_replicate=1,
                fixed_checkpoint_epochs=epochs, checks=checks)


def verify(output):
    output = Path(output).resolve()
    started = time.perf_counter()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), output_dir=str(output),
                  auditor_sha256=sha256(Path(__file__)), training_performed=False)
    try:
        metadata = json_read(output / "run_config.json")
        require(metadata["status"] == "complete", "Run has not completed")
        configuration = metadata["configuration"].copy()
        for name in ("cases", "checkpoints"):
            configuration[name] = tuple(configuration[name])
        config = rt.Config(**{field.name: configuration[field.name] for field in fields(rt.Config)})
        config.validate()
        for relative, digest in metadata["identity"]["sources"].items():
            require(sha256(rt.ROOT / relative) == digest, "Live scientific/reporting source differs from saved run")
        raw = read_csv(output / "raw_epoch_trajectory.csv").sort_values(["case", "replicate", "epoch"])
        result.update(raw=verify_raw(raw, config), summaries=verify_summaries(output, raw),
                      archives=verify_archives(output, raw, config),
                      reconstructed_checkpoints=verify_reloaded_metrics(output, raw, config))
        result.update(status="passed", elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        result.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        rt.atomic_json(result, output / "verification.json")
        raise
    rt.atomic_json(result, output / "verification.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    verify(arguments.output_dir)
