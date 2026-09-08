"""Monte Carlo summaries and figures for paired, uninterrupted DPLQR paths.

This module only reads checkpoint diagnostics. It never trains, chooses a
checkpoint, restores weights, or filters trajectories by the observed effect.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


DIAGNOSTICS = (
    "theta_error_l2", "train_check_loss", "validation_check_loss",
    "test_check_loss", "m_l2_error", "q_l2_error", "linear_error_mse",
    "nonlinear_error_mse", "cross_error_moment", "cancellation_fraction",
    "q_mse", "decomposition_residual",
)
COLORS = {"theta1": "#0072B2", "theta2": "#D55E00", "total": "#009E73"}


def _stats(values):
    values = np.asarray(values, dtype=float)
    count = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if count > 1 else float("nan")
    mcse = sd / math.sqrt(count) if count > 1 else float("nan")
    return dict(repetitions=count, mean=mean, sd=sd, mcse=mcse,
                mc_sd_available=count > 1)


def _summaries(raw):
    theta_rows, diagnostic_rows = [], []
    for (case, epoch), group in raw.groupby(["case", "epoch"], sort=True):
        common = dict(case=int(case), case_name=str(group.case_name.iloc[0]),
                      epoch=int(epoch))
        for component in (1, 2):
            estimates = group[f"theta_hat_{component}"].to_numpy(float)
            truths = group[f"theta_true_{component}"].to_numpy(float)
            if not np.all(truths == truths[0]):
                raise ValueError("The true coefficient varies within a case/epoch.")
            errors = estimates - truths
            stats = _stats(estimates)
            theta_rows.append({
                **common, "theta_component": f"theta_{component}",
                "theta_true": float(truths[0]), "mean_theta": stats["mean"],
                "bias": float(errors.mean()), "sd": stats["sd"],
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mae": float(np.mean(np.abs(errors))),
                "bias_mcse": stats["mcse"],
                "repetitions": stats["repetitions"],
                "mc_sd_available": stats["mc_sd_available"],
            })
        for metric in DIAGNOSTICS:
            if metric in group:
                diagnostic_rows.append({**common, "metric": metric,
                                        **_stats(group[metric])})
    return pd.DataFrame(theta_rows), pd.DataFrame(diagnostic_rows)


def _paired_endpoints(raw):
    epochs = sorted(int(e) for e in raw.epoch.unique())
    start, end = (100, 1000) if 100 in epochs and 1000 in epochs else (epochs[0], epochs[-1])
    first = raw.loc[raw.epoch == start].set_index(["case", "replicate"])
    last = raw.loc[raw.epoch == end].set_index(["case", "replicate"])
    if not first.index.equals(last.index):
        first, last = first.sort_index(), last.sort_index()
    if not first.index.equals(last.index):
        raise ValueError("Endpoint diagnostics do not contain the same paired replicates.")
    rows, counts = [], []
    metrics = ["theta_error_l2", "abs_theta_error_1", "abs_theta_error_2",
               "train_check_loss", "test_check_loss", "m_l2_error", "q_l2_error"]
    for case in sorted(raw.case.unique()):
        before, after = first.loc[case], last.loc[case]
        for metric in metrics:
            changes = after[metric].to_numpy(float) - before[metric].to_numpy(float)
            stats = _stats(changes)
            half_width = 1.96 * stats["mcse"]
            rows.append(dict(
                case=int(case), case_name=str(before.case_name.iloc[0]),
                start_epoch=start, end_epoch=end, metric=metric,
                start_mean=float(before[metric].mean()),
                end_mean=float(after[metric].mean()), mean_change=stats["mean"],
                change_sd=stats["sd"], change_mcse=stats["mcse"],
                change_mc_interval_low=stats["mean"] - half_width,
                change_mc_interval_high=stats["mean"] + half_width,
                repetitions=stats["repetitions"],
                mc_sd_available=stats["mc_sd_available"],
            ))
        train_down = after.train_check_loss.to_numpy() < before.train_check_loss.to_numpy()
        theta_up = after.theta_error_l2.to_numpy() > before.theta_error_l2.to_numpy()
        q_down = after.q_l2_error.to_numpy() < before.q_l2_error.to_numpy()
        test_up = after.test_check_loss.to_numpy() > before.test_check_loss.to_numpy()
        counts.append(dict(
            case=int(case), case_name=str(before.case_name.iloc[0]),
            start_epoch=start, end_epoch=end, repetitions=len(before),
            training_loss_down=int(train_down.sum()), theta_l2_up=int(theta_up.sum()),
            q_l2_down=int(q_down.sum()), test_loss_up=int(test_up.sum()),
            training_down_theta_up_q_down=int((train_down & theta_up & q_down).sum()),
            training_down_theta_up_test_up=int((train_down & theta_up & test_up).sum()),
            negative_cross_moment_start=int((before.cross_error_moment < 0).sum()),
            negative_cross_moment_end=int((after.cross_error_moment < 0).sum()),
        ))
    return pd.DataFrame(rows), pd.DataFrame(counts), start, end


def _make_figures(raw, theta, diagnostics, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(int(case) for case in raw.case.unique())
    epochs = sorted(int(epoch) for epoch in raw.epoch.unique())
    counts = raw.groupby("case").replicate.nunique()
    q_label = (f"Q = {int(counts.iloc[0])} per case" if counts.nunique() == 1
               else ", ".join(f"Case {int(c)}: Q = {int(q)}" for c, q in counts.items()))
    subtitle = (f"{q_label}; n = {int(raw.n.iloc[0])}; tau = {float(raw.tau.iloc[0]):g}; "
                "one continuous path per replicate; no early stopping")
    names = raw.groupby("case").case_name.first().to_dict()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    artifacts = []

    def finish(fig, filename, title, footnote):
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14, y=0.985)
        fig.text(0.5, 0.015, footnote, ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.915))
        for extension in ("png", "pdf"):
            path = figure_dir / f"{filename}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts.append(str(path))
        plt.close(fig)

    def style(ax, case, ylabel):
        ax.set_title(f"Case {case}: {names[case]}")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ticks = [e for e in (1, 10, 100, 1000) if epochs[0] <= e <= epochs[-1]]
        if epochs[-1] not in ticks:
            ticks.append(epochs[-1])
        ax.set_xticks(sorted(set(ticks)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(alpha=0.22)
        if epochs[0] != epochs[-1]:
            ax.set_xlim(epochs[0], epochs[-1])

    def diagnostic_line(ax, case, metric, label, color, bands=False):
        subset = diagnostics.loc[(diagnostics.case == case) & (diagnostics.metric == metric)]
        x = subset.epoch.to_numpy(float)
        y = subset["mean"].to_numpy(float)
        ax.plot(x, y, marker="o", markersize=3, label=label, color=color)
        if bands and subset.mc_sd_available.all():
            half = 1.96 * subset.mcse.to_numpy(float)
            ax.fill_between(x, y - half, y + half, color=color, alpha=0.12)

    fig, axes = plt.subplots(len(cases), 2, figsize=(11.8, 3.1 * len(cases)), squeeze=False)
    for row, case in enumerate(cases):
        for col, component in enumerate((1, 2)):
            ax = axes[row, col]
            subset = theta.loc[(theta.case == case) & (theta.theta_component == f"theta_{component}")]
            x, y = subset.epoch.to_numpy(float), subset.bias.to_numpy(float)
            color = COLORS[f"theta{component}"]
            ax.plot(x, y, marker="o", markersize=3, color=color)
            if subset.mc_sd_available.all():
                half = 1.96 * subset.bias_mcse.to_numpy(float)
                ax.fill_between(x, y - half, y + half, color=color, alpha=0.16)
            ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
            style(ax, case, rf"Bias of $\hat\theta_{component}$")
    finish(fig, "A_theta_bias", "A. Monte Carlo coefficient bias along training",
           "Shading: mean ± 1.96 Monte Carlo SE (pointwise normal approximation); absent when Q = 1.")

    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.6), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        for component in (1, 2):
            subset = theta.loc[(theta.case == case) & (theta.theta_component == f"theta_{component}")]
            ax.plot(subset.epoch, subset.rmse, marker="o", markersize=3,
                    color=COLORS[f"theta{component}"], label=rf"RMSE: $\theta_{component}$")
        diagnostic_line(ax, case, "theta_error_l2", "Mean coefficient L2 error", COLORS["total"])
        style(ax, case, "Coefficient estimation error")
        ax.legend(loc="best")
    finish(fig, "B_theta_error", "B. Coefficient error along training",
           "Component RMSE = sqrt(mean squared coefficient error); vector curve = mean Euclidean error across replicates.")

    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.6), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        diagnostic_line(ax, case, "train_check_loss", "Training", COLORS["theta1"])
        diagnostic_line(ax, case, "test_check_loss", "Independent test", COLORS["theta2"])
        style(ax, case, "Mean quantile check loss")
        ax.legend(loc="best")
    finish(fig, "C_check_loss", "C. Training and independent test loss",
           "Test data never influence training. Curves are Monte Carlo means; connected segments link recorded checkpoints only.")

    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.6), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        diagnostic_line(ax, case, "m_l2_error", "Nuisance m: L2 error", COLORS["theta2"], bands=True)
        diagnostic_line(ax, case, "q_l2_error", "Full quantile q: L2 error", COLORS["theta1"], bands=True)
        style(ax, case, "Mean out-of-sample L2 error")
        ax.legend(loc="best")
    finish(fig, "D_nuisance_and_quantile_error", "D. Nuisance and full conditional-quantile error",
           "Each replicate's error is sqrt(MSE) on its fixed evaluation sample. Shading: pointwise ± 1.96 Monte Carlo SE.")

    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.6), squeeze=False)
    for ax, case in zip(axes.flat, cases):
        subset = diagnostics.loc[diagnostics.case == case].pivot(index="epoch", columns="metric", values="mean")
        ax.plot(subset.index, subset.linear_error_mse + subset.nonlinear_error_mse,
                color=COLORS["theta2"], marker="o", markersize=3, label="Linear MSE + nuisance MSE")
        ax.plot(subset.index, 2 * subset.cross_error_moment, color=COLORS["total"],
                marker="o", markersize=3, label="2 × cross-error moment")
        ax.plot(subset.index, subset.q_mse, color=COLORS["theta1"], marker="o", markersize=3, label="Full quantile MSE")
        ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
        style(ax, case, "Mean evaluation-sample squared error")
        ax.legend(loc="best")
    finish(fig, "E_error_decomposition", "E. Linear–nonlinear error compensation",
           "Quantile MSE = linear MSE + nuisance MSE + 2 × cross-error moment. A negative cross term offsets component errors.")

    fixed_replicates = [rep for rep in (1, 2, 3) if rep in set(raw.replicate)]
    fig, axes = plt.subplots(len(cases), 2, figsize=(11.8, 3.1 * len(cases)), squeeze=False)
    palette = ("#0072B2", "#D55E00", "#009E73")
    for row, case in enumerate(cases):
        for col, component in enumerate((1, 2)):
            ax = axes[row, col]
            for replicate, color in zip(fixed_replicates, palette):
                subset = raw.loc[(raw.case == case) & (raw.replicate == replicate)].sort_values("epoch")
                ax.plot(subset.epoch, subset[f"theta_hat_{component}"], marker="o", markersize=3,
                        color=color, label=f"Replicate {replicate}")
            truth = raw.loc[raw.case == case, f"theta_true_{component}"].iloc[0]
            ax.axhline(truth, color="#444444", linewidth=1, linestyle="--", label=f"Truth = {truth:g}")
            style(ax, case, rf"Raw $\hat\theta_{component}$")
            ax.legend(loc="best")
    finish(fig, "F_fixed_replicate_paths", "F. Individual continuous training paths",
           "Replicates 1, 2, and 3, where available, are fixed in advance; no selection by observed drift. Monte Carlo summaries are primary.")
    return artifacts


def _format(value, digits=4):
    return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "NA"


def _markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(str(item) for item in row) + " |" for row in rows)])


def _write_results(raw, theta, diagnostics, paired, counts, start, end, config, output):
    per_case = raw.groupby("case").replicate.nunique()
    q_text = "; ".join(f"Case {int(case)}: {int(q)} repetitions" for case, q in per_case.items())
    intended = config.get("intended_repetitions", 100)
    lines = [
        "# Continued-training diagnostic results", "",
        "These results follow one initialized DPLQR model continuously on one training dataset per "
        "replicate. Training uses the fixed epoch ceiling, with no early stopping, no best-validation "
        "weight restoration, and no checkpoint selection. Validation, test and evaluation data do not "
        "affect optimization.", "",
        f"Completed data: **{q_text}**; n = {int(raw.n.iloc[0])}; tau = {float(raw.tau.iloc[0]):g}; "
        f"training/validation sizes = {int(raw.n_train.iloc[0])}/{int(raw.n_validation.iloc[0])}; "
        f"independent test size = {int(raw.n_test.iloc[0])}; independent evaluation size = {int(raw.n_eval.iloc[0])}.", "",
        f"Recorded epochs: {', '.join(str(int(epoch)) for epoch in sorted(raw.epoch.unique()))}. "
        f"There are {len(raw)} checkpoint rows and {int(per_case.sum())} distinct case–replicate trajectories.", "",
    ]
    if int(per_case.min()) < int(intended):
        lines += [f"**This is a reduced run. The intended design is {int(intended)} repetitions per case; "
                  "the current run does not complete that design.** Small-run differences and Monte Carlo "
                  "uncertainty intervals are exploratory.", ""]
    if int(per_case.min()) == 1:
        lines += ["With only one replicate, Monte Carlo SD and MCSE are undefined. They are saved as "
                  "NaN/blank with `mc_sd_available=False`, never replaced by zero.", ""]
    lines += [
        "## Paired endpoint comparisons", "",
        f"The following comparisons use epoch **{start} → {end}** on the same replicate. "
        "Changes are end minus start. The coefficient diagnostic is the Euclidean error "
        "`||theta_hat - theta_true||_2`; it is not the absolute value of average bias.", "",
    ]
    for metric, label in (
        ("theta_error_l2", "Coefficient L2 error"),
        ("train_check_loss", "Training check loss"),
        ("test_check_loss", "Independent test check loss"),
        ("m_l2_error", "Nuisance L2 error"),
        ("q_l2_error", "Full conditional-quantile L2 error"),
    ):
        subset = paired.loc[paired.metric == metric]
        rows = [[f"Case {int(row.case)}", _format(row.start_mean), _format(row.end_mean),
                 _format(row.mean_change), _format(row.change_mcse),
                 f"[{_format(row.change_mc_interval_low)}, {_format(row.change_mc_interval_high)}]"]
                for row in subset.itertuples()]
        lines += [f"### {label}", "", _markdown_table(
            ["Case", f"Mean at {start}", f"Mean at {end}", "Mean paired change", "MCSE of change", "Approx. MC interval"], rows), ""]
    lines += [
        "The intervals are mean paired change ± 1.96 × its Monte Carlo SE, treating a replicate as "
        "the independent unit. They are pointwise normal approximations, not simultaneous intervals "
        "or tests adjusted for looking across cases, coefficients and epochs. In small runs they can "
        "be unreliable. Evaluation-sample approximation error is also part of each replicate's measurement.", "",
        "### Counts of paired directional changes", "",
        _markdown_table(
            ["Case", "Q", "Train loss ↓", "Theta L2 ↑", "q L2 ↓", "Test loss ↑", "Train ↓, theta ↑, q ↓", "Train ↓, theta ↑, test ↑"],
            [[f"Case {int(row.case)}", int(row.repetitions), int(row.training_loss_down),
              int(row.theta_l2_up), int(row.q_l2_down), int(row.test_loss_up),
              int(row.training_down_theta_up_q_down), int(row.training_down_theta_up_test_up)]
             for row in counts.itertuples()]), "",
        "These are literal strict directional comparisons of finite-sample estimates, without a "
        "post hoc threshold for 'roughly stable'. Counts alone do not establish a population effect. "
        "A lower q error with a worse coefficient estimate is compatible with compensation by the "
        "nuisance network; a negative cross-error moment directly describes cancellation on the "
        "evaluation sample. Neither observation alone establishes a causal or asymptotic mechanism. "
        "Worsening test loss alongside lower training loss is evidence compatible with ordinary "
        "prediction overfitting, which can coexist with compensation.", "",
        "## Coefficient bias and RMSE", "",
    ]
    endpoint_rows = theta.loc[theta.epoch.isin([start, end])]
    lines += [_markdown_table(
        ["Case", "Component", "Epoch", "Mean estimate", "Truth", "Bias", "SD", "RMSE"],
        [[int(row.case), row.theta_component, int(row.epoch), _format(row.mean_theta),
          _format(row.theta_true), _format(row.bias), _format(row.sd), _format(row.rmse)]
         for row in endpoint_rows.itertuples()]), "",
        "Bias is the mean signed coefficient error. SD uses the sample standard deviation across "
        "replicates (ddof = 1). RMSE is sqrt(mean squared coefficient error). Small bias can coexist "
        "with large SD or RMSE because positive and negative errors cancel across replicates.", "",
        "## Figures", "",
    ]
    for filename, caption in (
        ("A_theta_bias", "A. Mean signed coefficient error; zero denotes no bias."),
        ("B_theta_error", "B. Component RMSE and mean vector coefficient L2 error."),
        ("C_check_loss", "C. Training and genuinely independent test check loss."),
        ("D_nuisance_and_quantile_error", "D. Nuisance and full quantile L2 error on fixed evaluation samples."),
        ("E_error_decomposition", "E. Squared-error decomposition and cancellation between linear and nonlinear errors."),
        ("F_fixed_replicate_paths", "F. Raw coefficient paths for pre-specified replicates 1–3, where available."),
    ):
        lines += [caption, "", f"![{caption}](figures/{filename}.png)", "",
                  f"[PDF version](figures/{filename}.pdf)", ""]
    maximum_residual = float(raw.decomposition_residual.abs().max())
    lines += [
        "## Definitions and reproducibility", "",
        "The nuisance and full-quantile L2 diagnostics are square roots of empirical mean squared "
        "errors against the known true functions on a fixed independent evaluation sample within "
        "each replicate. They are **not** the relative mean squared error used in the older paper-table "
        "replication output. The nuisance truth includes the error-distribution quantile shift when "
        "tau differs from 0.5. All reported network values are evaluated directly without post hoc "
        "coefficient clipping or nuisance recalibration.", "",
        "The decomposition is `q_mse = linear_error_mse + nonlinear_error_mse + "
        "2 * cross_error_moment`. The raw `cancellation_fraction` is "
        "`-2 * cross_error_moment / (linear_error_mse + nonlinear_error_mse)` "
        "(zero if the denominator is zero); a positive value indicates error cancellation. "
        f"The maximum absolute recorded decomposition residual is {maximum_residual:.3g}.", "",
        "Per-replicate data, initialization, shuffle/training, test and evaluation seeds are retained "
        "in the raw CSV. Optimizer-step counts, model-instance identifiers and fit-call counts are "
        "also retained to audit the continuous paths. Consult the run configuration and execution "
        "validation record for the architecture, optimizer, ceiling and checks actually used.", "",
        "- [Raw checkpoint trajectories](raw_epoch_trajectory.csv)",
        "- [Tidy coefficient summaries](monte_carlo_summary.csv)",
        "- [Tidy diagnostic summaries](diagnostic_summary.csv)",
        "- [Paired endpoint changes and Monte Carlo uncertainty](paired_endpoint_changes.csv)",
        "- [Paired directional counts](endpoint_pattern_counts.csv)",
        "- [Run configuration](run_config.json)", "",
    ]
    report = output / "RESULTS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return str(report)


def write_reports(raw: pd.DataFrame, output: Path, config: dict) -> dict:
    """Write summaries and figures for the supplied completed trajectory rows."""
    if raw.empty:
        raise ValueError("Cannot report an empty trajectory run.")
    if raw.duplicated(["case", "replicate", "epoch"]).any():
        raise ValueError("Duplicate case/replicate/epoch checkpoint rows.")
    required = {
        "case", "case_name", "replicate", "epoch", "n", "n_train", "n_validation",
        "n_test", "n_eval", "tau", "theta_hat_1", "theta_hat_2", "theta_true_1",
        "theta_true_2", "abs_theta_error_1", "abs_theta_error_2", *DIAGNOSTICS,
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing checkpoint columns: {sorted(missing)}")
    if (raw.epoch <= 0).any():
        raise ValueError("Checkpoint epochs must be positive for the log-scale plots.")
    numeric = raw[list(required - {"case_name"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite raw diagnostic values cannot be summarized as valid results.")
    epoch_sets = raw.groupby(["case", "replicate"]).epoch.apply(lambda values: tuple(sorted(values)))
    if epoch_sets.nunique() != 1:
        raise ValueError("All reported trajectories must have the same complete checkpoint schedule.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = raw.sort_values(["case", "replicate", "epoch"]).copy()
    theta, diagnostics = _summaries(raw)
    paired, counts, start, end = _paired_endpoints(raw)
    files = []
    for frame, filename in (
        (theta, "monte_carlo_summary.csv"), (diagnostics, "diagnostic_summary.csv"),
        (paired, "paired_endpoint_changes.csv"), (counts, "endpoint_pattern_counts.csv"),
    ):
        target = output / filename
        frame.to_csv(target, index=False, na_rep="NaN")
        files.append(str(target))
    figures = _make_figures(raw, theta, diagnostics, output)
    report = _write_results(raw, theta, diagnostics, paired, counts, start, end, config, output)
    return dict(summary_rows=len(theta), diagnostic_summary_rows=len(diagnostics),
                trajectories=int(raw.groupby(["case", "replicate"]).ngroups),
                checkpoint_rows=len(raw), endpoint_epochs=[start, end],
                artifacts=[*files, *figures, report])
