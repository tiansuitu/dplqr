"""Simple coefficient-versus-epoch diagnostics from saved Case 3 replications.

Run from the repository root (or any working directory):
    python results/2026-09-11_asymptotic_normality_case3/consistency_plots/plot_coefficient_stability.py

No simulation or model fitting runs. Repeated invocations use a fresh rerun directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(columns, aliases, *, required=True):
    """Accept case, spaces, underscores and common naming variations."""
    matches = [column for column in columns if normalized(column) in {normalized(a) for a in aliases}]
    if len(matches) == 1:
        return matches[0]
    if not matches and not required:
        return None
    raise ValueError(f"Expected one of {aliases}; matched {matches}. Available columns: {list(columns)}")


def load_results(experiment: Path):
    config_path = experiment / "run/run_config.json"
    raw_path = experiment / "run/raw_replications.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    setup = config.get("identity", config)
    if setup.get("case") != 3:
        raise ValueError("The saved setup is not Case 3.")
    truth = setup.get("theta", setup.get("theta_0", setup.get("theta_true")))
    if truth is None:
        raise ValueError("No true coefficient vector in the saved simulation setup; inspect it before plotting.")
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Invalid true coefficient vector in the saved setup.")

    raw = pd.read_csv(raw_path, float_precision="round_trip")
    print("Available saved columns:", ", ".join(raw.columns))
    print("True coefficient vector from", config_path.relative_to(experiment), ":", truth.tolist())
    aliases = {
        "n": ["n", "sample_size", "n_total", "total_sample_size"],
        "epochs": ["epochs", "epoch", "n_epochs", "num_epochs", "epoch_count"],
        "rep": ["rep", "replication", "replication_id", "replicate", "rep_id"],
    }
    mapping = {find_column(raw.columns, names): key for key, names in aliases.items()}
    for j in range(1, len(truth) + 1):
        estimate = find_column(raw.columns, [f"theta_hat_{j}", f"theta_{j}_hat",
            f"hat_theta_{j}", f"theta_estimate_{j}", f"theta_est_{j}", f"estimated_theta_{j}"])
        mapping[estimate] = f"theta_hat_{j}"
        truth_column = find_column(raw.columns, [f"theta_0_{j}", f"theta0_{j}",
            f"theta_true_{j}", f"true_theta_{j}", f"theta_{j}_true"], required=False)
        if truth_column is not None and not raw[truth_column].eq(truth[j - 1]).all():
            raise ValueError(f"Stored truth column {truth_column} disagrees with run_config.json.")
    data = raw[list(mapping)].rename(columns=mapping).copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Replication results are empty or contain non-finite values.")
    for column in ("n", "epochs", "rep"):
        if not (data[column].gt(0) & data[column].eq(np.floor(data[column]))).all():
            raise ValueError(f"{column} must contain positive integers.")
        data[column] = data[column].astype(int)
    if data.duplicated(["n", "epochs", "rep"]).any():
        raise ValueError("Duplicate replication keys; refusing to double-count fits.")
    if "case" in raw and not raw.case.eq(3).all():
        raise ValueError("Mixed simulation cases in saved results.")
    if "tau" in raw and not raw.tau.eq(setup["tau"]).all():
        raise ValueError("Mixed quantiles in saved results.")

    expected_q = config.get("requested_Q")
    counts = data.groupby(["n", "epochs"]).size()
    expected_grid = pd.MultiIndex.from_product([
        config.get("sample_sizes", sorted(data.n.unique())),
        config.get("epoch_counts", sorted(data.epochs.unique()))], names=["n", "epochs"])
    counts = counts.reindex(expected_grid, fill_value=0)
    print("\nAvailable replication counts:\n" + counts.rename("Q").to_string())
    if expected_q is not None and not counts.eq(expected_q).all():
        warnings.warn(f"Some settings do not have Q={expected_q}. Summaries use actual counts.")
    return data, truth, config, raw_path, config_path


def summarize(data: pd.DataFrame, truth: np.ndarray) -> pd.DataFrame:
    records = []
    for j, theta0 in enumerate(truth, start=1):
        for (n, epochs), group in data.groupby(["n", "epochs"], sort=True):
            values = group[f"theta_hat_{j}"].to_numpy()
            q = len(values)
            mean = float(values.mean())
            bias = mean - theta0
            sd = float(values.std(ddof=1)) if q > 1 else np.nan
            rmse = float(np.sqrt(np.mean((values - theta0) ** 2)))
            if q > 1:
                # Distinguishes mean bias from across-replication spread; checks the RMSE divisor.
                np.testing.assert_allclose(rmse**2, bias**2 + (q - 1) / q * sd**2, rtol=1e-12, atol=1e-14)
            records.append(dict(component=f"theta_{j}", n=int(n), epochs=int(epochs), Q=q,
                true_theta=float(theta0), mean_theta_hat=mean, bias=float(bias),
                absolute_bias=float(abs(bias)), MC_SD=sd, RMSE=rmse))
    return pd.DataFrame(records)


def choose_output(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in {".png", ".csv", ".json", ".md"} for p in base.iterdir()):
        output = base / ("rerun_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        output.mkdir(exist_ok=False)
        return output
    return base


def save_figure(fig, output: Path, filename: str, created: list):
    path = output / filename
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    created.append(path)


def q_caption(frame):
    counts = sorted(frame.Q.unique())
    if len(counts) == 1:
        return f"Q={counts[0]} replications per setting"
    return f"Q={min(counts)}–{max(counts)}; actual counts in summary tables"


def plot_diagnostics(data, summary, truth, output, created):
    sample_sizes = sorted(data.n.unique())
    epoch_counts = sorted(data.epochs.unique())
    colors = plt.get_cmap("tab10").colors
    # Plain Matplotlib axes; no theme, shaded bands, or smoothing.
    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11}):
        for j, theta0 in enumerate(truth, start=1):
            component = summary.loc[summary.component.eq(f"theta_{j}")]
            symbol = rf"$\hat{{\theta}}_{j}$"
            panels = [
                ("mean_theta_hat", "mean_vs_epochs", f"Monte Carlo mean of {symbol}",
                 f"Case 3: does mean {symbol} stay near the truth\nas training epochs increase?"),
                ("absolute_bias", "absolute_bias_vs_epochs", f"Absolute bias of {symbol}",
                 f"Case 3: {symbol} absolute bias versus epochs"),
                ("RMSE", "rmse_vs_epochs", f"RMSE of {symbol}",
                 f"Case 3: {symbol} RMSE versus epochs"),
            ]
            for metric, filename, ylabel, title in panels:
                fig, ax = plt.subplots(figsize=(7.8, 5.2), layout="constrained")
                for index, n in enumerate(sample_sizes):
                    group = component.loc[component.n.eq(n)].sort_values("epochs")
                    ax.plot(group.epochs, group[metric], marker="o", linewidth=1.6,
                            color=colors[index % len(colors)], label=f"n = {n}")
                if metric == "mean_theta_hat":
                    ax.axhline(theta0, color="black", linestyle="--", linewidth=1.3,
                               label=rf"True $\theta_{{0{j}}}$ = {theta0:g}")
                else:
                    ax.set_ylim(bottom=0)
                ax.set(xlabel="Training epochs", ylabel=ylabel, xticks=epoch_counts,
                       title=title + "\n" + q_caption(component))
                ax.legend(loc="best", frameon=False)
                save_figure(fig, output, f"theta{j}_{filename}.png", created)

            for n in sample_sizes:
                frame = data.loc[data.n.eq(n)]
                epochs_here = sorted(frame.epochs.unique())
                arrays = [frame.loc[frame.epochs.eq(e), f"theta_hat_{j}"].to_numpy() for e in epochs_here]
                positions = np.arange(1, len(arrays) + 1)
                fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
                ax.boxplot(arrays, positions=positions, widths=.5, whis=1.5, showfliers=True,
                           medianprops={"color": "black", "linewidth": 1.5},
                           flierprops={"marker": ".", "markersize": 4, "markeredgecolor": "0.45"})
                ax.axhline(theta0, color="black", linestyle="--", linewidth=1.3,
                           label=rf"True $\theta_{{0{j}}}$ = {theta0:g}")
                ax.set_xticks(positions, [f"{e}\nQ={len(v)}" for e, v in zip(epochs_here, arrays)])
                ax.set(xlabel="Training epochs", ylabel=f"Estimated coefficient {symbol}",
                       title=f"Case 3: distribution of {symbol} across epochs\nn = {n}")
                ax.legend(loc="best", frameon=False)
                save_figure(fig, output, f"theta{j}_n{n}_boxplot.png", created)


def markdown_table(frame):
    # Avoid requiring the optional tabulate package.
    columns = ["n", "epochs", "Q", "mean_theta_hat", "bias", "absolute_bias", "MC_SD", "RMSE"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = [str(int(x)) if i < 3 else f"{x:.5f}" for i, x in enumerate(row)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary, truth, output, created):
    lines = ["# Case 3: coefficient stability across training epochs", "",
        "These plots use only the saved DPLQR point estimates in `run/raw_replications.csv`. "
        "True values come from `run/run_config.json` and are checked against the saved truth columns. "
        "No models were fitted. The sample size n is the total generated size; 80% trains the model.", "",
        "## How to read the plots", "",
        "- **Mean versus epochs:** distance from the dashed true-value line measures signed mean bias.",
        "- **Absolute bias:** absolute distance of the Monte Carlo mean from the truth; lower is better.",
        "- **RMSE:** square root of the average squared coefficient error across replications; includes both bias and spread.",
        "- **Boxplots:** center lines are medians, boxes span the middle 50%, whiskers extend to the most extreme observations "
        "within 1.5 interquartile ranges, and dots show more extreme observations. The dashed line is the true value.", "",
        "Monte Carlo SD uses divisor Q-1; RMSE averages squared errors with divisor Q. "
        "Plots use all saved replications, with actual Q shown. Epoch settings reuse the same seeded datasets.", "",
        "## What changes between the first and last saved epochs?", ""]
    for j, theta0 in enumerate(truth, start=1):
        component = summary.loc[summary.component.eq(f"theta_{j}")]
        lines.extend([f"### theta_{j}: true value {theta0:g}", ""])
        for n, group in component.groupby("n", sort=True):
            group = group.sort_values("epochs")
            first, last = group.iloc[0], group.iloc[-1]
            direction = "decreases" if last.absolute_bias < first.absolute_bias else "increases" if last.absolute_bias > first.absolute_bias else "is unchanged"
            lines.append(f"- n={n}, {int(first.epochs)} to {int(last.epochs)} epochs: mean "
                f"{first.mean_theta_hat:.5f} to {last.mean_theta_hat:.5f}; absolute bias {direction} "
                f"({first.absolute_bias:.5f} to {last.absolute_bias:.5f}); "
                f"SD {first.MC_SD:.5f} to {last.MC_SD:.5f}; RMSE {first.RMSE:.5f} to {last.RMSE:.5f}.")
        lines.extend(["", markdown_table(component), ""])
    lines.extend(["## Interpreting sample size and consistency", ""])
    for j in range(1, len(truth) + 1):
        component = summary.loc[summary.component.eq(f"theta_{j}")]
        for epochs, group in component.groupby("epochs", sort=True):
            group = group.sort_values("n")
            bias_monotone = bool((group.absolute_bias.diff().dropna() <= 0).all())
            rmse_monotone = bool((group.RMSE.diff().dropna() <= 0).all())
            lines.append(f"- theta_{j}, {epochs} epochs: absolute bias "
                f"{'decreases at every saved increase in n' if bias_monotone else 'does not decrease at every saved increase in n'}; "
                f"RMSE {'decreases at every saved increase in n' if rmse_monotone else 'does not decrease at every saved increase in n'}.")
    lines.extend(["", "Closeness to the true coefficient **as epochs increase at fixed n** is a training-stability diagnostic. "
        "Formal asymptotic consistency concerns convergence to the true coefficient **as n tends to infinity**, "
        "under stated assumptions and a specified estimator/training sequence. These finite-sample plots do not establish that property. "
        "Small changes in empirical bias also have Monte Carlo uncertainty; they need not reflect a systematic population drift.", "",
        "## Reproduce", "", "From the repository root, using its Python environment:", "", "```powershell",
        "& ./.venv/Scripts/python.exe results/2026-09-11_asymptotic_normality_case3/consistency_plots/plot_coefficient_stability.py",
        "```", "", "Requires NumPy, pandas and Matplotlib only. Existing outputs are preserved; repeat runs write to a fresh `rerun_*` subfolder.", ""])
    path = output / "README.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    created.append(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    data, truth, config, raw_path, config_path = load_results(experiment)
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path, config_path)}
    summary = summarize(data, truth)
    output = choose_output(experiment / "consistency_plots")
    created = []
    for j in range(1, len(truth) + 1):
        frame = summary.loc[summary.component.eq(f"theta_{j}")]
        path = output / f"theta{j}_summary.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, float_format="%.17g")
        created.append(path)
        print(f"\ntheta_{j} summary (true value = {truth[j - 1]:g}):")
        print(frame.drop(columns=["component", "true_theta"]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    plot_diagnostics(data, summary, truth, output, created)
    write_report(summary, truth, output, created)
    for path, digest in protected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Input changed: {path}"
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={p.relative_to(experiment).as_posix(): digest for p, digest in protected.items()},
        true_theta=truth.tolist(), tau=config.get("identity", config).get("tau"),
        replication_rows=len(data), summary_rows=len(summary), simulations_launched=0,
        MC_SD_ddof=1, RMSE_definition="sqrt(mean((theta_hat - theta0)**2))",
        actual_Q=sorted(int(q) for q in summary.Q.unique()),
        outputs=[p.relative_to(experiment).as_posix() for p in created])
    manifest_path = output / "plot_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    created.append(manifest_path)
    print("\nNew output files:")
    for path in created:
        print(path.relative_to(experiment).as_posix())
    print("\nRun complete; source results are unchanged.")


if __name__ == "__main__":
    main()
