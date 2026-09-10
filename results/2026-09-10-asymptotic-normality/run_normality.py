"""Resumable, DPLQR-only normality diagnostic. Run --help for commands.

Defaults reproduce the September 4 fixed first-pass profile. Each completed
main/auxiliary fit is persisted independently, before density estimation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torchtuples import Model

from source_bridge import HERE, ROOT, NOTEBOOK, covNet, dqNetSparse, checkLoss
from source_bridge import find_rscript, load_original, source_digest

SCHEMA = 1
TAU = 0.5
DEFAULT_NS = [500, 1000, 2000, 4000]
DEFAULT_CASES = [1, 2, 3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_torch(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@contextmanager
def run_lock(output):
    """Prevent concurrent writers; a crashed process's stale lock is recoverable."""
    path = output / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another run may own {path}. If its process has exited, remove only this lock.") from exc
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def identity(args):
    hp = dict(mode="fixed", depth=args.depth, width=args.width, epochs=args.epochs,
              batch_size=args.batch_size, patience=args.patience, lr=args.learning_rate)
    r_version = subprocess.run([find_rscript(), "--version"], capture_output=True, text=True,
                               check=True, timeout=30)
    return dict(schema=SCHEMA, tau=TAU, theta=[1.0, -1.0], seed=args.seed,
                train_fraction=0.8, hp=hp, threads=args.threads,
                deterministic_algorithms=True,
                selected_notebook_definitions=source_digest(),
                sources={name: sha(HERE / name) for name in
                         ["source_bridge.py", "run_normality.py", "density_only.R"]},
                dqAux_sha256=sha(ROOT / "dqAux.py"),
                versions={name: importlib.metadata.version(name) for name in
                          ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torchtuples"]},
                python=platform.python_version(), platform=platform.platform(),
                r_version=(r_version.stdout + r_version.stderr).strip())


def restore_scaler(saved):
    scaler = StandardScaler()
    for key in ("mean_", "scale_", "var_"):
        setattr(scaler, key, np.array(saved[key], dtype=float))
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = saved["n_samples_seen_"]
    return scaler


def save_scaler(scaler):
    return {**{key: getattr(scaler, key).tolist() for key in ("mean_", "scale_", "var_")},
            "n_samples_seen_": int(scaler.n_samples_seen_)}


def load_stage(path, fingerprint, setting):
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["identity_sha256"] != fingerprint or saved["setting"] != setting:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    return saved


def fit_replication(original, hp, case, n, rep, seed, directory, fingerprint):
    """Add checkpointing around unchanged source fit and covariance functions."""
    directory.mkdir(parents=True, exist_ok=True)
    setting = dict(case=case, n=n, rep=rep)
    setting_seed = seed + case * 10_000_000 + n * 1_000 + rep
    fit_seed = setting_seed + int(round(TAU * 1_000_000))
    rng = np.random.default_rng(setting_seed)
    x, z, y, _ = original.generate_dataset(n, case, rng)
    order = rng.permutation(n)
    n_train = int(0.8 * n)
    tr, va = order[:n_train], order[n_train:]
    data_sha = hashlib.sha256(b"".join(a.tobytes() for a in [x, z, y, order])).hexdigest()
    main_path = directory / "main.pt"
    saved = load_stage(main_path, fingerprint, setting)
    if saved is None:
        started = time.perf_counter()
        model, scaler, stopper, selected_hp = original.select_dplqr(
            x[tr], z[tr], y[tr], x[va], z[va], y[va], TAU, hp, fit_seed)
        # Exactly the second clipping call made by September 4 fit_dplqr.
        original.clip_neural_weights(model.net)
        saved = dict(identity_sha256=fingerprint, setting=setting, data_sha256=data_sha,
                     state=model.net.state_dict(), scaler=save_scaler(scaler), hp=selected_hp,
                     epochs_run=stopper.epochs_run, best_epoch=stopper.best_epoch,
                     best_validation_loss=stopper.best, fit_seconds=time.perf_counter() - started)
        atomic_torch(saved, main_path)
    else:
        if saved["data_sha256"] != data_sha:
            raise ValueError("Regenerated data do not match the main-fit checkpoint")
        original.seed_all(fit_seed)
        net = dqNetSparse(2, 8, torch.zeros((1, 2), dtype=torch.float32),
                          [hp["depth"], hp["width"]], sparseRatio=0.5)
        net.load_state_dict(saved["state"])
        model = Model(net, checkLoss(tau=TAU), device="cpu")
        scaler = restore_scaler(saved["scaler"])

    # Cache both projections individually. Training and prediction remain the
    # original functions; neither projection is omitted for the primary target.
    fit_projection = original.fit_projection
    projection_details = []

    def cached_projection(z_train, target_train, z_val, target_val, profile, projection_seed, binary):
        column = 0 if binary else 1
        path = directory / f"projection_{column + 1}.pt"
        checkpoint = load_stage(path, fingerprint, setting)
        if checkpoint is None:
            started = time.perf_counter()
            projection, stopper = fit_projection(z_train, target_train, z_val, target_val,
                                                  profile, projection_seed, binary)
            checkpoint = dict(identity_sha256=fingerprint, setting=setting,
                              state=projection.net.state_dict(), seed=projection_seed,
                              epochs_run=stopper.epochs_run, best_epoch=stopper.best_epoch,
                              seconds=time.perf_counter() - started)
            atomic_torch(checkpoint, path)
        else:
            if checkpoint["seed"] != projection_seed:
                raise ValueError("Projection seed mismatch")
            original.seed_all(projection_seed)
            net = covNet(8, [profile["depth"], profile["width"]], logic=binary)
            net.load_state_dict(checkpoint["state"])
            projection = Model(net, nn.MSELoss(), device="cpu")
            stopper = SimpleNamespace(epochs_run=checkpoint["epochs_run"], best_epoch=checkpoint["best_epoch"])
        projection_details.append(checkpoint)
        return projection, stopper

    original.namespace["fit_projection"] = cached_projection
    started = time.perf_counter()
    try:
        se, density, projection_epochs = original.dplqr_standard_errors(
            model, scaler, saved["hp"], x[tr], z[tr], y[tr], x[va], z[va], y[va], TAU,
            fit_seed + 500_000)
    finally:
        original.namespace["fit_projection"] = fit_projection
    # Sum persisted fit timings so interrupted/resumed jobs retain their true
    # compute cost. Exclude cached loading time from fresh-run extrapolations.
    inference_wall = time.perf_counter() - started
    projection_seconds = sum(detail["seconds"] for detail in projection_details)
    # A separate explicit density timer is recorded by the inference wrapper.
    density_seconds = original.namespace.get("last_density_seconds", 0.0)
    inference_seconds = projection_seconds + density_seconds
    theta = model.net.linLinear.weight.detach().cpu().numpy().reshape(-1).astype(float)
    if not np.isfinite(theta).all() or not np.isfinite(se).all() or np.any(se <= 0):
        raise FloatingPointError("Invalid coefficients or standard errors; no row will be counted")
    lower, upper = theta - norm.ppf(.975) * se, theta + norm.ppf(.975) * se
    row = dict(case=case, n=n, n_train=n_train, n_val=n-n_train, rep=rep, tau=TAU, method="DPLQR",
               setting_seed=setting_seed, fit_seed=fit_seed, data_sha256=data_sha,
               theta1=float(theta[0]), theta2=float(theta[1]),
               se_theta1=float(se[0]), se_theta2=float(se[1]),
               lower_theta1=float(lower[0]), upper_theta1=float(upper[0]),
               lower_theta2=float(lower[1]), upper_theta2=float(upper[1]),
               covered_theta1=bool(lower[0] <= 1 <= upper[0]),
               covered_theta2=bool(lower[1] <= -1 <= upper[1]),
               t_theta1=float((theta[0]-1)/se[0]),
               root_n_error_theta1=float(np.sqrt(n)*(theta[0]-1)),
               root_n_train_error_theta1=float(np.sqrt(n_train)*(theta[0]-1)),
               fit_seconds=float(saved["fit_seconds"]), inference_seconds=float(inference_seconds),
               density_seconds=float(density_seconds), projection_seconds=float(projection_seconds),
               total_seconds=float(saved["fit_seconds"]+inference_seconds),
               inference_wall_seconds=float(inference_wall), epochs_run=int(saved["epochs_run"]),
               best_epoch=int(saved["best_epoch"]), best_validation_loss=float(saved["best_validation_loss"]),
               density_zero=float(density), projection_epochs=json.dumps(projection_epochs),
               projection_best_epochs=json.dumps([int(d["best_epoch"]) for d in projection_details]),
               selected_hyperparameters=json.dumps(saved["hp"], sort_keys=True))
    return row


def collect_results(output, fingerprint):
    rows = []
    for path in sorted((output / "replications").glob("case_*_n_*_rep_*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["identity_sha256"] != fingerprint:
            raise ValueError(f"Mismatched replication: {path}")
        for name, digest in result["stage_sha256"].items():
            if sha(path.parent / name) != digest:
                raise ValueError(f"Corrupt fit checkpoint: {path.parent / name}")
        row = result["row"]
        numeric = [row[key] for key in ["theta1", "theta2", "se_theta1", "se_theta2", "t_theta1"]]
        if not np.isfinite(numeric).all() or min(row["se_theta1"], row["se_theta2"]) <= 0:
            raise ValueError(f"Invalid completed row: {path}")
        rows.append(row)
    keys = [(row["case"], row["n"], row["rep"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate replication keys")
    return rows


def run(args):
    output = args.output_dir.resolve()
    # New artifacts are confined to this experiment; no destructive restart mode.
    if output == HERE or HERE not in output.parents:
        raise ValueError(f"Choose an output subdirectory within {HERE}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    configuration = identity(args)
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    config_path = output / "run_config.json"
    with run_lock(output):
        if config_path.exists():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if old["identity"] != configuration:
                raise ValueError("Settings, scientific source, or environment changed. Use a fresh output subdirectory.")
        elif any((output / name).exists() for name in ["replications", "raw_replications.csv"]):
            raise ValueError("Existing results have no run identity")
        rows = collect_results(output, fingerprint)
        expected = set(itertools.product(args.cases, args.sample_sizes, range(1, args.repetitions+1)))
        completed = {(row["case"], row["n"], row["rep"]) for row in rows}
        if not completed.issubset(expected):
            raise ValueError("Requested grid/Q would exclude completed rows. Retain or extend the previous grid/Q.")
        meta = dict(identity=configuration, identity_sha256=fingerprint,
                    requested_cases=args.cases, requested_sample_sizes=args.sample_sizes,
                    requested_repetitions=args.repetitions,
                    original_notebook=str(NOTEBOOK.relative_to(ROOT)),
                    requested_result_rows=len(expected),
                    started_utc=datetime.now(timezone.utc).isoformat())
        started = time.perf_counter()
        reused = len(rows)

        def checkpoint(status, error=None):
            if rows:
                atomic_csv(pd.DataFrame(rows).sort_values(["case", "n", "rep"]), output / "raw_replications.csv")
            meta.update(status=status, completed_result_rows=len(rows), reused_result_rows=reused,
                        missing_result_rows=len(expected-completed), last_error=error,
                        invocation_seconds=time.perf_counter()-started,
                        updated_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(meta, config_path)

        checkpoint("running")
        original = load_original()
        density_fn = original.namespace["residual_density_zero"]

        def timed_density(residual):
            tick = time.perf_counter()
            answer = density_fn(residual)
            original.namespace["last_density_seconds"] = time.perf_counter()-tick
            return answer

        original.namespace["residual_density_zero"] = timed_density
        print(f"Requested {len(expected)} replications; reusing {reused}; "
              "each new replication fits one DPLQR + two auxiliary networks.", flush=True)
        try:
            # Replication-first order gives balanced partial progress over the grid.
            for rep, case, n in itertools.product(range(1, args.repetitions+1), args.cases, args.sample_sizes):
                key = case, n, rep
                if key in completed:
                    continue
                directory = output / "replications" / f"case_{case}_n_{n}_rep_{rep:04d}"
                row = fit_replication(original, configuration["hp"], case, n, rep, args.seed, directory, fingerprint)
                atomic_json(dict(identity_sha256=fingerprint, row=row,
                                 stage_sha256={name: sha(directory / name) for name in
                                               ["main.pt", "projection_1.pt", "projection_2.pt"]}),
                            directory / "result.json")
                rows.append(row)
                completed.add(key)
                checkpoint("running")
                print(f"case={case} n={n} rep={rep}/{args.repetitions} "
                      f"theta1={row['theta1']:.5f} SE={row['se_theta1']:.5f} T={row['t_theta1']:.3f}; "
                      f"fit+inference={row['total_seconds']:.2f}s; {len(rows)}/{len(expected)} complete", flush=True)
        except BaseException as exc:
            checkpoint("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", repr(exc))
            raise
        checkpoint("complete")
        frame = pd.DataFrame(rows).sort_values(["case", "n", "rep"])
        if not args.no_report:
            from reporting import generate_report
            generate_report(frame, output, args.repetitions)
        print(f"Complete: {output}; invocation elapsed {time.perf_counter()-started:.1f}s", flush=True)
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report", "estimate-runtime"])
    parser.add_argument("--output-dir", type=Path, default=HERE / "pilot")
    parser.add_argument("--repetitions", "-Q", type=int, default=20)
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--cases", nargs="+", type=int, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-report", action="store_true", help="Skip reports until a later report command")
    args = parser.parse_args()
    if (args.repetitions < 1 or min(args.sample_sizes) < 20 or not set(args.cases) <= {1, 2, 3}
            or min(args.depth, args.width, args.epochs, args.batch_size, args.patience, args.threads) < 1
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seed < 0
            or len(set(args.cases)) != len(args.cases) or len(set(args.sample_sizes)) != len(args.sample_sizes)):
        parser.error("Require positive valid settings, unique cases 1-3, and unique n >= 20")
    if args.command == "run":
        run(args)
    else:
        frame = pd.read_csv(args.output_dir / "raw_replications.csv", float_precision="round_trip")
        if args.command == "report":
            from reporting import generate_report
            generate_report(frame, args.output_dir, args.repetitions)
        else:
            from reporting import timing_projection
            timing_projection(frame, args.output_dir, args.cases, args.sample_sizes)


if __name__ == "__main__":
    main()
