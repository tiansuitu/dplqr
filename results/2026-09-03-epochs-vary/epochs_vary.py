"""Concrete DPLQR epoch-ceiling sensitivity, 03 September 2026 (03092026).

Based on demo.ipynb and dqAux.py. Run with the repository's Python environment.
Each seed/quantile uses a shared training trajectory, saving the best validation
weights available at each ceiling. This is equivalent to separate capped fits
with paired random seeds; --verify checks against ordinary EarlyStopping fits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import platform
import random
import sys
import time
import warnings
from unittest.mock import patch
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import gaussian_kde
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import statsmodels
import statsmodels.formula.api as smf
import torch
import torchtuples as tt
from torchtuples import Model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dqAux import checkErrorMean, checkLoss, covNet, dqNetSparse, getSESingle

FEATURES = ["blst", "flh", "sup", "cag", "fag", "age"]
NODES = [5, 128]
BATCH_SIZE = 128
LR = 0.001
SPARSITY = 0.5
TAUS = np.arange(1, 50) / 50
DELTA = 0.02


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(split_seed):
    df = pd.read_csv(ROOT / "data/concrete.csv")
    pool, test = train_test_split(df, test_size=0.2, random_state=split_seed)
    val = pool.sample(frac=0.1, random_state=split_seed)
    train = pool.drop(val.index)
    mapper = DataFrameMapper([([col], StandardScaler()) for col in FEATURES])
    mapper.fit(train)
    data = {"pool": pool, "train": train, "val": val, "test": test}
    for label, frame in [("train", train), ("val", val), ("test", test)]:
        data[f"x_{label}"] = (
            torch.tensor(frame[["inX"]].values, dtype=torch.float32),
            torch.tensor(mapper.transform(frame).astype("float32")),
        )
        data[f"y_{label}"] = torch.tensor(frame[["outY"]].values, dtype=torch.float32)
    data["x_merge"] = tuple(torch.cat((a, b)) for a, b in zip(data["x_train"], data["x_val"]))
    data["y_merge"] = torch.cat((data["y_train"], data["y_val"]))
    return data


class CapSnapshots(tt.callbacks.Callback):
    """Save the same weights that default EarlyStopping would restore at a cap.

    The library's own callback controls stopping (patience=10, min_delta=0).
    Memory checkpoints avoid repeated training and temporary weight files.
    Equal best scores also save weights, matching torchtuples 0.2.2 behavior.
    """

    def __init__(self, caps, stopper):
        self.caps = caps
        self.stopper = stopper
        self.snapshots = {}
        self.history = []
        self.best_state = None
        self.best_epoch = 0

    def on_fit_start(self):
        self.started = time.perf_counter()

    def on_epoch_end(self):
        score = self.model.val_metrics.scores["loss"]["score"][-1]
        train_loss = self.model.train_metrics.scores["loss"]["score"][-1]
        if not np.isfinite([score, train_loss]).all():
            raise ValueError("Non-finite training or validation loss")
        epoch = len(self.history) + 1
        if score == self.stopper.cur_best:
            self.best_state = {k: v.detach().clone() for k, v in self.model.net.state_dict().items()}
            self.best_epoch = epoch
        self.history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=score))
        if epoch in self.caps:
            self.capture(epoch)

    def capture(self, cap):
        self.snapshots[cap] = dict(
            state=self.best_state, epochs_run=len(self.history), best_epoch=self.best_epoch,
            best_val_loss=float(self.stopper.cur_best),
            early_stop_triggered=self.stopper._iter_since_best >= self.stopper.patience,
        )

    def on_fit_end(self):
        for cap in self.caps:
            if cap not in self.snapshots:
                self.capture(cap)
        self.elapsed = time.perf_counter() - self.started


def make_model(stage, seed, tau, initial_coef):
    set_seed(seed)
    if stage == "main":
        net = dqNetSparse(1, len(FEATURES), torch.tensor(initial_coef, dtype=torch.float32), NODES, SPARSITY)
        loss = checkLoss(tau=tau)
    else:
        # Same continuous-projection correction as the current demo/getSESingle.
        net = covNet(len(FEATURES), NODES, logic=False)
        loss = torch.nn.MSELoss()
    model = Model(net, loss, device="cpu")
    model.optimizer.set_lr(LR)
    return model


def fit_model(model, data, stage, cap, callbacks):
    if stage == "main":
        x, y = data["x_train"], data["y_train"]
        validation = (data["x_val"], data["y_val"])
    else:
        x, y = data["x_train"][1], data["x_train"][0]
        validation = (data["x_val"][1], data["x_val"][0])
    return model.fit(x, y, BATCH_SIZE, cap, callbacks, False,
                     val_data=validation, val_batch_size=BATCH_SIZE)


def fit_caps(data, stage, seed, tau, initial_coef, caps):
    model = make_model(stage, seed, tau, initial_coef)
    stopper = tt.callbacks.EarlyStopping(checkpoint_model=False, load_best=False)
    recorder = CapSnapshots(caps, stopper)
    fit_model(model, data, stage, max(caps), [stopper, recorder])
    return model, recorder


def fit_lqr(data, output):
    rows, warning_rows = [], []
    for tau in TAUS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.quantreg("outY~inX+blst+flh+sup+cag+fag+age", data["pool"]).fit(q=tau)
        warning_rows.extend(dict(tau=float(tau), category=w.category.__name__, message=str(w.message)) for w in caught)
        ci = model.conf_int().loc["inX"]
        prediction = model.predict(data["test"]).to_numpy().reshape(-1, 1)
        rows.append(dict(tau=tau, estCoef=model.params.loc["inX"], lowerCI=ci.iloc[0], upperCI=ci.iloc[1],
                         CL=checkErrorMean(prediction, data["y_test"].numpy(), tau=tau)))
    result = pd.DataFrame(rows)
    result.to_csv(output / "lqr_reference.csv", index=False)
    (output / "lqr_warnings.json").write_text(json.dumps(warning_rows, indent=2), encoding="utf-8")
    return result


def evaluate(data, main, projection, main_snapshot, projection_snapshot, tau):
    main.net.load_state_dict(main_snapshot["state"])
    projection.net.load_state_dict(projection_snapshot["state"])
    residuals = (data["y_merge"] - main.predict(data["x_merge"])).numpy().reshape(-1)
    z_delta = (data["x_merge"][0] - projection.predict(data["x_merge"][1])).numpy()
    z_delta -= z_delta.mean()
    cov_m = float((z_delta ** 2).mean())
    density_zero = float(gaussian_kde(residuals).evaluate(0.0)[0])
    se = float(np.sqrt(tau * (1 - tau) / (len(residuals) * cov_m * density_zero ** 2)))
    coefficient = float(main.net.linLinear.weight.detach().item())
    prediction = main.predict(data["x_test"]).numpy()
    return dict(estCoef=coefficient, se=se, lowerCI=coefficient - 1.96 * se, upperCI=coefficient + 1.96 * se,
                CL=float(checkErrorMean(prediction, data["y_test"].numpy(), tau=tau)),
                residual_density_zero=density_zero, projection_residual_variance=cov_m)


@contextmanager
def memory_checkpoints():
    """Use ordinary torch save/load in RAM to avoid Windows checkpoint locks."""
    buffers = {}

    def save(model, path, **kwargs):
        buffer = io.BytesIO()
        torch.save(model.net.state_dict(), buffer, **kwargs)
        buffers[str(path)] = buffer.getvalue()

    def load(model, path, **kwargs):
        model.net.load_state_dict(torch.load(io.BytesIO(buffers[str(path)]), **kwargs))

    with patch.object(Model, "save_model_weights", save), patch.object(Model, "load_model_weights", load):
        yield


@memory_checkpoints()
def verify(data, output, lqr, caps, seed):
    """Independent default fits validate the computational shortcut and SE formula."""
    tau = 0.5
    initial = float(lqr.loc[lqr.tau == tau, "estCoef"].iloc[0])
    main_seed, aux_seed = seed * 1000 + 50, seed * 1000 + 51
    main, rec = fit_caps(data, "main", main_seed, tau, initial, caps)
    aux, aux_rec = fit_caps(data, "projection", aux_seed, tau, initial, caps)
    checks = []
    for cap in caps:
        for stage, stage_seed, recording in [("main", main_seed, rec), ("projection", aux_seed, aux_rec)]:
            direct = make_model(stage, stage_seed, tau, initial)
            stopper = tt.callbacks.EarlyStopping(file_path=output / "verification-checkpoint.pt")
            log = fit_model(direct, data, stage, cap, [stopper])
            snapshot = recording.snapshots[cap]
            difference = max(float((v - snapshot["state"][k]).abs().max()) for k, v in direct.net.state_dict().items())
            assert difference == 0.0, (stage, cap, difference)
            assert len(log.to_pandas()) == snapshot["epochs_run"]
            checks.append(dict(stage=stage, epoch_cap=cap, max_weight_difference=difference,
                               epochs_run=snapshot["epochs_run"]))
        result = evaluate(data, main, aux, rec.snapshots[cap], aux_rec.snapshots[cap], tau)
        residuals = (data["y_merge"] - main.predict(data["x_merge"])).numpy()
        set_seed(aux_seed)
        # getSESingle creates its own callback; the decorator stores it in RAM.
        import os
        previous = Path.cwd()
        try:
            os.chdir(output)
            direct_se = float(getSESingle(data["x_train"], data["x_val"], residuals, tau, NODES,
                                         BATCH_SIZE, LR, cap, [], False, logic=False)[0])
        finally:
            os.chdir(previous)
        np.testing.assert_allclose(result["se"], direct_se, rtol=1e-7, atol=1e-9)
        checks.append(dict(stage="SE_vs_getSESingle", epoch_cap=cap, absolute_difference=abs(result["se"] - direct_se)))
    (output / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("Verified all ceilings against independent default fits and getSESingle.", flush=True)


def summarize(rows, lqr, output):
    runs = rows.groupby(["seed", "epoch_cap"]).agg(
        ACL=("CL", lambda s: DELTA * s.sum()),
        CI_area=("CI_width", lambda s: DELTA * s.sum()),
        main_epochs_mean=("main_epochs_run", "mean"), main_epochs_max=("main_epochs_run", "max"),
        projection_epochs_mean=("projection_epochs_run", "mean"),
        projection_epochs_max=("projection_epochs_run", "max"),
        main_stopped_before_cap_pct=("main_stopped_before_cap", lambda s: 100 * s.mean()),
        projection_stopped_before_cap_pct=("projection_stopped_before_cap", lambda s: 100 * s.mean()),
    ).reset_index()
    baseline = runs[runs.epoch_cap == 500].set_index("seed").ACL
    runs["ACL_change_vs_500_pct"] = [100 * (r.ACL / baseline.loc[r.seed] - 1) for r in runs.itertuples()]
    runs.to_csv(output / "metrics_by_seed.csv", index=False)
    summary = runs.groupby("epoch_cap").agg(
        ACL_mean=("ACL", "mean"), ACL_sd=("ACL", "std"),
        CI_area_mean=("CI_area", "mean"), CI_area_sd=("CI_area", "std"),
        ACL_change_vs_500_pct=("ACL_change_vs_500_pct", "mean"),
        main_epochs_mean=("main_epochs_mean", "mean"), main_epochs_max=("main_epochs_max", "max"),
        projection_epochs_mean=("projection_epochs_mean", "mean"), projection_epochs_max=("projection_epochs_max", "max"),
        main_stopped_before_cap_pct=("main_stopped_before_cap_pct", "mean"),
        projection_stopped_before_cap_pct=("projection_stopped_before_cap_pct", "mean"),
    ).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    lqr_acl = float(DELTA * lqr.CL.sum())
    lqr_area = float(DELTA * (lqr.upperCI - lqr.lowerCI).sum())
    table = ["| Epoch ceiling | ACL, mean (SD) | Change vs 500 | CI area, mean (SD) | Mean epochs: main / projection |",
             "|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        table.append(f"| {r.epoch_cap:,} | {r.ACL_mean:.6f} ({r.ACL_sd:.6f}) | {r.ACL_change_vs_500_pct:+.2f}% | "
                     f"{r.CI_area_mean:.6f} ({r.CI_area_sd:.6f}) | {r.main_epochs_mean:.1f} / {r.projection_epochs_mean:.1f} |")
    (output / "summary.md").write_text(
        "# Concrete DPLQR epoch sensitivity — 03092026\n\n" + "\n".join(table) +
        f"\n\nSame-split LQR reference: ACL = {lqr_acl:.6f}; CI area = {lqr_area:.6f}.\n\n"
        "SD is across training seeds on one fixed split, not a confidence interval. "
        "Lower ACL is better. Smaller CI area alone does not establish better inference or coverage.\n",
        encoding="utf-8")
    make_figure(rows, runs, summary, lqr_acl, output)
    return summary


def make_figure(rows, runs, summary, lqr_acl, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    x = np.arange(len(summary))
    colors = ["#b75b38", "#c79532", "#54856d", "#2563a6", "#8055a0"]
    for ax, metric, label in [(axes[0, 0], "ACL", "Test ACL (lower is better)"),
                               (axes[0, 1], "CI_area", "Integrated 95% CI width")]:
        for _, group in runs.groupby("seed"):
            ax.plot(x, group.sort_values("epoch_cap")[metric], color="#a6b5c5", alpha=0.7, lw=1)
        ax.errorbar(x, summary[f"{metric}_mean"], yerr=summary[f"{metric}_sd"].fillna(0),
                    color="#183e66", marker="o", capsize=4, lw=2, label="Mean ± 1 SD across seeds")
        if metric == "ACL":
            ax.axhline(lqr_acl, color="#9c572d", ls="--", label=f"LQR reference: {lqr_acl:.4f}")
        ax.set(ylabel=label, xlabel="Epoch ceiling", xticks=x, xticklabels=summary.epoch_cap)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for color, (cap, group) in zip(colors, rows.groupby("epoch_cap")):
        coefs = group.groupby("tau").estCoef.mean()
        ax.plot(coefs.index, coefs, color=color, lw=1.8, ls="--" if cap == 1000 else "-", label=f"{cap:,}")
    ax.set(xlabel="Quantile level", ylabel="Water-cement coefficient (seed mean)")
    ax.legend(title="Epoch ceiling", ncol=3, fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    ax.plot(x, summary.main_epochs_mean, "o-", color="#2563a6", label="Main DPLQR: mean")
    ax.plot(x, summary.projection_epochs_mean, "s-", color="#54856d", label="SE projection: mean")
    ax.plot(x, summary.main_epochs_max, "o:", color="#2563a6", alpha=0.7, label="Main DPLQR: maximum")
    ax.plot(x, summary.projection_epochs_max, "s:", color="#54856d", alpha=0.7, label="SE projection: maximum")
    ax.set(xlabel="Epoch ceiling", ylabel="Epochs actually completed", xticks=x, xticklabels=summary.epoch_cap)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle("Concrete DPLQR: sensitivity to the epoch ceiling\n"
                 f"03092026  |  {rows.seed.nunique()} training seeds, one fixed split, 49 quantiles, early stopping retained", fontsize=14)
    fig.savefig(output / "epochs_comparison.png", dpi=200)
    fig.savefig(output / "epochs_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 100, 250, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--split-seed", type=int, default=20260903)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify", action="store_true", help="Check paired caps against independent fits at tau=0.5")
    args = parser.parse_args()
    caps = sorted(set(args.epochs))
    if min(caps) < 1 or 500 not in caps:
        parser.error("Epoch ceilings must be positive and include the 500-epoch reference.")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) > 4_294_966:
        parser.error("Training seeds must be distinct integers between 0 and 4294966.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data = load_data(args.split_seed)
    for label in ["train", "val", "test"]:
        data[label].rename_axis("source_row").to_csv(output / f"df_{label}.csv")
    config = dict(date="03092026", status="running", epochs=caps, seeds=args.seeds, split_seed=args.split_seed,
                  quantiles=TAUS.tolist(), batch_size=BATCH_SIZE, nodes=NODES, learning_rate=LR,
                  sparse_ratio=SPARSITY, early_stopping_patience=10, projection_logic=False,
                  epoch_scope="Both main DPLQR and auxiliary SE projection", threads=args.threads,
                  seed_rule="main = training_seed * 1000 + 2 * quantile_index (1-based); projection = main + 1",
                  split_sizes={k: len(data[k]) for k in ["train", "val", "test"]},
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in ["demo.ipynb", "dqAux.py", "data/concrete.csv"]},
                  versions=dict(python=platform.python_version(), torch=torch.__version__, torchtuples=tt.__version__,
                                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                                sklearn=sklearn.__version__, statsmodels=statsmodels.__version__, matplotlib=matplotlib.__version__))
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Split: {config['split_sizes']}; ceilings: {caps}; seeds: {args.seeds}", flush=True)
    lqr = fit_lqr(data, output)
    if args.verify:
        verify(data, output, lqr, caps, args.seeds[0])
    rows, histories = [], []
    for seed in args.seeds:
        for q_index, tau in enumerate(TAUS, start=1):
            initial = float(lqr.iloc[q_index - 1].estCoef)
            main_seed = seed * 1000 + 2 * q_index
            model, rec = fit_caps(data, "main", main_seed, tau, initial, caps)
            aux, aux_rec = fit_caps(data, "projection", main_seed + 1, tau, initial, caps)
            for stage, recording in [("main", rec), ("projection", aux_rec)]:
                histories.extend(dict(seed=seed, tau=tau, stage=stage, **entry) for entry in recording.history)
            for cap in caps:
                row = dict(seed=seed, tau=tau, epoch_cap=cap,
                           **evaluate(data, model, aux, rec.snapshots[cap], aux_rec.snapshots[cap], tau))
                row["CI_width"] = row["upperCI"] - row["lowerCI"]
                for stage, recording in [("main", rec), ("projection", aux_rec)]:
                    snapshot = recording.snapshots[cap]
                    row.update({f"{stage}_{k}": v for k, v in snapshot.items() if k != "state"})
                    row[f"{stage}_stopped_before_cap"] = snapshot["epochs_run"] < cap
                rows.append(row)
            pd.DataFrame(rows).to_csv(output / "results_by_quantile.csv", index=False)
            if q_index % 5 == 0 or q_index == len(TAUS):
                print(f"Seed {seed}, quantile {q_index}/49; main {len(rec.history)}, projection {len(aux_rec.history)} epochs; "
                      f"elapsed {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    results = pd.DataFrame(rows)
    assert len(results) == len(args.seeds) * len(TAUS) * len(caps)
    assert not results.duplicated(["seed", "tau", "epoch_cap"]).any()
    assert np.isfinite(results.select_dtypes(include="number").to_numpy()).all()
    assert (results.lowerCI < results.upperCI).all()
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    summary = summarize(results, lqr, output)
    config.update(status="complete", elapsed_wall_seconds=time.perf_counter() - started,
                  verification_performed=args.verify, result_rows=len(results))
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
