"""Monte Carlo diagnostics for the DPLQR theta1 experiment (no model fitting).

``generate_report`` consumes completed replication rows from the runner.  The
requested n is the total generated sample; n_train is the estimation sample
used in the retained DPLQR covariance estimator.  Both scalings are recorded.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


THETA1_TRUE = 1.0
Z975 = float(stats.norm.ppf(0.975))
PROBABILITIES = (0.025, 0.5, 0.975)
QUANTILE_NAMES = ("q025", "q500", "q975")
BLUE, ORANGE, GREEN = "#2563a5", "#cc6b28", "#26836c"


def _numeric(group: pd.DataFrame, column: str) -> np.ndarray:
    if column not in group:
        return np.full(len(group), np.nan)
    return pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if not count:
        return float("nan"), float("nan")
    p = successes / count
    denominator = 1 + Z975**2 / count
    center = (p + Z975**2 / (2 * count)) / denominator
    half_width = Z975 * math.sqrt(p * (1 - p) / count + Z975**2 / (4 * count**2)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _values(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = _numeric(group, "theta1")
    se = _numeric(group, "se_theta1")
    valid_se = np.isfinite(se) & (se > 0)
    valid_t = valid_se & np.isfinite(theta)
    studentized = (theta[valid_t] - THETA1_TRUE) / se[valid_t]
    # Prefer the actual saved interval bounds; fallback supports hand-built data.
    if "lower_theta1" in group and "upper_theta1" in group:
        lower, upper = _numeric(group, "lower_theta1"), _numeric(group, "upper_theta1")
        valid_ci = valid_t & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    else:
        lower, upper = theta - Z975 * se, theta + Z975 * se
        valid_ci = valid_t
    covered = ((lower[valid_ci] <= THETA1_TRUE) & (THETA1_TRUE <= upper[valid_ci])).astype(float)
    return _finite(theta), se[valid_se], studentized, covered


def _summarize(group: pd.DataFrame, case: Any, n: int, requested_repetitions: int) -> dict[str, Any]:
    theta, se, studentized, covered = _values(group)
    error = theta - THETA1_TRUE
    n_train_values = _finite(_numeric(group, "n_train"))
    n_train = _mean(n_train_values)
    if len(n_train_values) and not np.all(n_train_values == n_train_values[0]):
        raise ValueError(f"Mixed n_train values within case={case}, n={n}; separate incompatible runs.")
    n_val = _mean(_finite(_numeric(group, "n_val")))
    bias, sd = _mean(error), _sd(theta)
    mcse_bias = sd / math.sqrt(len(theta)) if len(theta) >= 2 else float("nan")
    coverage = _mean(covered)
    ci_low, ci_high = _wilson(int(covered.sum()), len(covered))
    t_sd = _sd(studentized)
    skew = float(stats.skew(studentized, bias=False)) if len(studentized) >= 3 and t_sd > 0 else float("nan")
    kurtosis = float(stats.kurtosis(studentized, fisher=True, bias=False)) if len(studentized) >= 4 and t_sd > 0 else float("nan")
    root_n_error = math.sqrt(n) * error
    root_skew = float(stats.skew(root_n_error, bias=False)) if len(error) >= 3 and sd > 0 else float("nan")
    root_kurtosis = float(stats.kurtosis(root_n_error, fisher=True, bias=False)) if len(error) >= 4 and sd > 0 else float("nan")
    row: dict[str, Any] = {
        "case": case, "n": int(n), "n_train": n_train, "n_val": n_val,
        "tau": _mean(_finite(_numeric(group, "tau"))),
        "requested_repetitions": requested_repetitions, "completed_repetitions": len(group),
        "q_theta1": len(theta), "q_se_theta1": len(se),
        "q_studentized": len(studentized), "q_coverage": len(covered),
        "bias_theta1": bias, "sqrt_n_bias_theta1": math.sqrt(n) * bias,
        "sqrt_n_train_bias_theta1": math.sqrt(n_train) * bias if n_train > 0 else float("nan"),
        "sd_theta1": sd, "sqrt_n_sd_theta1": math.sqrt(n) * sd,
        "sqrt_n_train_sd_theta1": math.sqrt(n_train) * sd if n_train > 0 else float("nan"),
        "mcse_bias_theta1": mcse_bias, "mcse_sqrt_n_bias_theta1": math.sqrt(n) * mcse_bias,
        "mcse_sqrt_n_train_bias_theta1": math.sqrt(n_train) * mcse_bias if n_train > 0 else float("nan"),
        "mean_se_theta1": _mean(se),
        "sqrt_n_mean_se_theta1": math.sqrt(n) * _mean(se),
        "sqrt_n_train_mean_se_theta1": math.sqrt(n_train) * _mean(se) if n_train > 0 else float("nan"),
        "mean_se_to_empirical_sd_theta1": _mean(se) / sd if sd > 0 else float("nan"),
        "rmse_theta1": math.sqrt(_mean(error**2)) if len(error) else float("nan"),
        "coverage_95_theta1": coverage,
        "mcse_coverage_95_theta1": math.sqrt(coverage * (1 - coverage) / len(covered)) if len(covered) else float("nan"),
        "coverage_95_wilson_lower": ci_low, "coverage_95_wilson_upper": ci_high,
        "t_mean": _mean(studentized), "t_sd": t_sd,
        "mcse_t_mean": t_sd / math.sqrt(len(studentized)) if len(studentized) >= 2 else float("nan"),
        "t_skewness": skew, "t_excess_kurtosis": kurtosis,
        "root_n_error_mean": _mean(root_n_error), "root_n_error_sd": _sd(root_n_error),
        "root_n_error_skewness": root_skew, "root_n_error_excess_kurtosis": root_kurtosis,
    }
    for probability, name in zip(PROBABILITIES, QUANTILE_NAMES):
        empirical = float(np.quantile(studentized, probability)) if len(studentized) else float("nan")
        reference = float(stats.norm.ppf(probability))
        row[f"t_{name}"] = empirical
        row[f"normal_{name}"] = reference
        row[f"t_{name}_minus_normal"] = empirical - reference
        row[f"root_n_error_{name}"] = float(np.quantile(root_n_error, probability)) if len(root_n_error) else float("nan")
    for column in ("fit_seconds", "inference_seconds", "total_seconds", "epochs_run", "best_epoch", "density_zero"):
        row[f"mean_{column}"] = _mean(_finite(_numeric(group, column)))
    return row


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _axes_grid(count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = min(2, max(count, 1))
    rows = math.ceil(max(count, 1) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 3.8 * rows), squeeze=False)
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return figure, axes.ravel()


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _distribution_figures(raw: pd.DataFrame, summary: pd.DataFrame, output: Path, case: Any) -> list[Path]:
    case_raw = raw[raw["case"] == case]
    case_summary = summary[summary["case"] == case].sort_values("n")
    distributions = {int(n): _values(group)[2] for n, group in case_raw.groupby("n", sort=True)}
    all_t = np.concatenate(list(distributions.values()))
    # Keep all observed values visible, including extreme tails; never trim outliers.
    extent = max(3.5, float(np.max(np.abs(all_t))) + 0.5) if len(all_t) else 3.5
    x = np.linspace(-extent, extent, 1200)
    histogram, axes = _axes_grid(len(case_summary))
    qq, qq_axes = _axes_grid(len(case_summary))
    for index, row in enumerate(case_summary.itertuples(index=False)):
        values = distributions[int(row.n)]
        q = len(values)
        axis, qq_axis = axes[index], qq_axes[index]
        axis.plot(x, stats.norm.pdf(x), color="black", lw=1.7, label="N(0,1)")
        if q >= 2 and np.std(values) > 0:
            bins = max(4, min(20, math.ceil(math.sqrt(q))))
            axis.hist(values, bins=bins, density=True, color=BLUE, alpha=0.32, edgecolor="white", label="Empirical histogram")
            if q >= 5:
                try:
                    axis.plot(x, stats.gaussian_kde(values)(x), color=BLUE, lw=1.7, label="Empirical KDE")
                except (ValueError, np.linalg.LinAlgError):
                    pass
        elif q:
            axis.axvline(values[0], color=BLUE, lw=1.7, label="Observed T")
        if q:
            probabilities = (np.arange(1, q + 1) - 0.5) / q
            theoretical, observed = stats.norm.ppf(probabilities), np.sort(values)
            qq_axis.scatter(theoretical, observed, color=BLUE, s=23, alpha=0.85)
            lo = min(-2.5, float(min(theoretical.min(), observed.min())))
            hi = max(2.5, float(max(theoretical.max(), observed.max())))
        else:
            lo, hi = -2.5, 2.5
        padding = 0.05 * (hi - lo)
        lo, hi = lo - padding, hi + padding
        qq_axis.plot([lo, hi], [lo, hi], color="black", lw=1.3, label="N(0,1): y=x")
        qq_axis.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Standard-normal theoretical quantile", ylabel="Observed T quantile")
        if q < 2:
            message = "Insufficient replications for distribution diagnostics"
            for current_axis in (axis, qq_axis):
                current_axis.text(0.5, 0.95, message, transform=current_axis.transAxes,
                                  va="top", ha="center", fontsize=8, wrap=True)
        else:
            axis.text(0.98, 0.95, f"Mean {row.t_mean:.2f}; SD {row.t_sd:.2f}\nCoverage {row.coverage_95_theta1:.1%}",
                      transform=axis.transAxes, va="top", ha="right", fontsize=9)
        axis.set(xlim=(-extent, extent), xlabel=r"$T=(\widehat\theta_1-1)/\widehat{\mathrm{se}}$", ylabel="Density")
        for current_axis in (axis, qq_axis):
            current_axis.set_title(f"n={int(row.n):,}; valid Q={q}")
            current_axis.grid(alpha=0.17)
        if index == 0:
            qq_axis.legend(loc="lower right", fontsize=8)
    histogram.suptitle(f"Case {case}: studentized theta1 distributions", fontsize=14)
    qq.suptitle(f"Case {case}: normal QQ diagnostics (reference is y=x)", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    histogram.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                     ncol=3, frameon=False, fontsize=9)
    histogram.tight_layout(rect=(0, 0.035, 1, 0.95))
    qq.tight_layout(rect=(0, 0, 1, 0.95))
    return [_save(histogram, output / f"case_{_slug(case)}_studentized_histograms.png"),
            _save(qq, output / f"case_{_slug(case)}_normal_qq.png")]


def _across_n_figure(summary: pd.DataFrame, output: Path, case: Any) -> Path:
    frame = summary[summary["case"] == case].sort_values("n")
    n = frame["n"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    bias_axis, sd_axis, t_axis, coverage_axis = axes.ravel()
    bias_axis.errorbar(n, frame["sqrt_n_bias_theta1"], yerr=Z975 * frame["mcse_sqrt_n_bias_theta1"],
                      fmt="o-", color=BLUE, capsize=4, label=r"$\sqrt{n}$ bias; 95% MC error bars")
    bias_axis.axhline(0, color="black", lw=1, ls="--")
    bias_axis.set(title="Root-n bias", ylabel=r"$\sqrt{n}\,\mathrm{Bias}(\widehat\theta_1)$")
    sd_axis.plot(n, frame["sqrt_n_sd_theta1"], "o-", color=BLUE, label=r"$\sqrt{n}$ empirical SD")
    sd_axis.plot(n, frame["sqrt_n_mean_se_theta1"], "s--", color=ORANGE, label=r"$\sqrt{n}$ mean estimated SE")
    sd_axis.set(title="Root-n spread and estimated SE", ylabel="Scaled SD / SE")
    t_axis.plot(n, frame["t_mean"], "o-", color=BLUE, label="T mean (target 0)")
    t_axis.plot(n, frame["t_sd"], "s-", color=ORANGE, label="T SD (target 1)")
    t_axis.axhline(0, color=BLUE, lw=1, ls="--", alpha=0.6)
    t_axis.axhline(1, color=ORANGE, lw=1, ls="--", alpha=0.6)
    t_axis.set(title="Studentized location and spread", ylabel="T moment")
    coverage = frame["coverage_95_theta1"].to_numpy(dtype=float)
    lower = frame["coverage_95_wilson_lower"].to_numpy(dtype=float)
    upper = frame["coverage_95_wilson_upper"].to_numpy(dtype=float)
    coverage_axis.errorbar(n, coverage, yerr=np.vstack((np.maximum(0, coverage - lower), np.maximum(0, upper - coverage))),
                          fmt="o-", color=GREEN, capsize=4, label="Coverage; 95% Wilson MC interval")
    coverage_axis.axhline(0.95, color="black", lw=1, ls="--", label="Nominal 0.95")
    coverage_axis.set(title="Empirical 95% CI coverage", ylabel="Coverage", ylim=(0, 1.04))
    ticks = [f"{int(row.n):,}\nQ={int(row.completed_repetitions)}" for row in frame.itertuples(index=False)]
    for axis in axes.ravel():
        axis.set_xticks(n, ticks, fontsize=9)
        axis.set_xlabel("Total generated sample size n")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"Case {case}: across-n diagnostics for theta1", fontsize=14)
    figure.text(0.5, 0.008, "MC error bars describe finite-repetition uncertainty; n_train scaling is also saved in summary.csv.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    return _save(figure, output / f"case_{_slug(case)}_across_n.png")


def _root_error_qq_figure(raw: pd.DataFrame, summary: pd.DataFrame, output: Path, case: Any) -> Path:
    frame = summary[summary["case"] == case].sort_values("n")
    figure, axes = _axes_grid(len(frame))
    for axis, row in zip(axes, frame.itertuples(index=False)):
        group = raw[(raw["case"] == case) & (raw["n"] == row.n)]
        error = math.sqrt(row.n) * (_finite(_numeric(group, "theta1")) - THETA1_TRUE)
        q = len(error)
        if q:
            theoretical = stats.norm.ppf((np.arange(1, q + 1) - 0.5) / q)
            axis.scatter(theoretical, np.sort(error), color=BLUE, s=23, alpha=0.85)
            limit = max(2.5, float(np.max(np.abs(theoretical))))
            if q >= 2:
                line_x = np.array([-limit, limit])
                axis.plot(line_x, _mean(error) + _sd(error) * line_x, color=ORANGE,
                          lw=1.4, label="Fitted normal: sample mean + SD × x")
                axis.legend(loc="best", fontsize=8)
                axis.text(0.03, 0.97, f"Skew {_format(row.root_n_error_skewness)}\nExcess kurtosis {_format(row.root_n_error_excess_kurtosis)}",
                          transform=axis.transAxes, ha="left", va="top", fontsize=9)
        if q < 2:
            axis.text(0.5, 0.95, "Insufficient replications for fitted normal reference", transform=axis.transAxes,
                      va="top", ha="center", fontsize=8)
        axis.set(title=f"n={int(row.n):,}; valid Q={q}", xlabel="Standard-normal theoretical quantile",
                 ylabel=r"Observed $\sqrt{n}(\widehat\theta_1-1)$ quantile")
        axis.grid(alpha=0.17)
    figure.suptitle(f"Case {case}: root-n estimator error QQ (fitted location and scale)", fontsize=14)
    figure.text(0.5, 0.006, "Reference fitted separately at each n: assesses shape only, not zero bias or variance calibration.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.025, 1, 0.95))
    return _save(figure, output / f"case_{_slug(case)}_root_n_error_qq.png")


def _shape_figure(summary: pd.DataFrame, output: Path, case: Any) -> Path:
    frame = summary[summary["case"] == case].sort_values("n")
    n = frame["n"].to_numpy(dtype=float)
    figure, (shape_axis, quantile_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    shape_axis.plot(n, frame["t_skewness"], "o-", color=BLUE, label="Skewness (normal: 0)")
    shape_axis.plot(n, frame["t_excess_kurtosis"], "s-", color=ORANGE, label="Excess kurtosis (normal: 0)")
    shape_axis.axhline(0, color="black", lw=1, ls="--")
    shape_axis.set(title="Studentized shape", ylabel="Bias-corrected sample moment")
    for probability, name, color, marker in zip(PROBABILITIES, QUANTILE_NAMES, (BLUE, GREEN, ORANGE), ("v", "o", "^")):
        quantile_axis.plot(n, frame[f"t_{name}"], marker=marker, color=color, label=f"Empirical {probability:.1%} quantile")
        quantile_axis.axhline(stats.norm.ppf(probability), color=color, lw=1, ls="--", alpha=0.8)
    quantile_axis.set(title="Studentized quantiles; dashed = N(0,1)", ylabel="T quantile")
    ticks = [f"{int(row.n):,}\nQ={int(row.q_studentized)}" for row in frame.itertuples(index=False)]
    for axis in (shape_axis, quantile_axis):
        axis.set_xticks(n, ticks, fontsize=9)
        axis.set_xlabel("Total generated sample size n")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    if frame["q_studentized"].max() < 3:
        shape_axis.text(0.5, 0.6, "Insufficient replications for shape moments", transform=shape_axis.transAxes,
                        ha="center", va="center", fontsize=9)
    figure.suptitle(f"Case {case}: studentized shape and tails", fontsize=14)
    figure.text(0.5, 0.01, "Small-Q shape moments and tail quantiles are noisy; Q=1 is execution evidence only.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.025, 1, 0.95))
    return _save(figure, output / f"case_{_slug(case)}_studentized_shape.png")


def _format(value: Any, decimals: int = 3) -> str:
    return f"{float(value):.{decimals}f}" if pd.notna(value) and np.isfinite(float(value)) else "NA"


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    headers = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, _ in columns:
            value = row[column]
            if column == "case":
                cells.append(str(int(value)) if isinstance(value, (int, float, np.number)) and np.isfinite(value) and float(value).is_integer() else str(value))
            else:
                cells.append(_format(value, 0 if column in {"n", "completed_repetitions"} else 3))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([headers, divider, *rows])


def _interpretation(summary: pd.DataFrame, requested_repetitions: int) -> str:
    lines = [
        "# DPLQR theta1 asymptotic-normality diagnostics", "",
        f"Primary target: theta1=1 at tau=0.5. Requested repetitions per case and n: **{requested_repetitions}**. "
        "Tables and figures use only completed replications; valid counts are recorded separately for estimates, SEs, studentization, and coverage.", "",
        _markdown_table(summary, [("case", "Case"), ("n", "n"), ("completed_repetitions", "Q"),
                                 ("bias_theta1", "Bias"), ("sqrt_n_bias_theta1", "sqrt(n) bias"),
                                 ("sqrt_n_sd_theta1", "sqrt(n) SD"), ("t_mean", "T mean"),
                                 ("t_sd", "T SD"), ("coverage_95_theta1", "Coverage")]), "",
        _markdown_table(summary, [("case", "Case"), ("n", "n"), ("sd_theta1", "Raw SD"),
                                 ("mean_se_theta1", "Mean SE"), ("rmse_theta1", "RMSE"),
                                 ("mean_se_to_empirical_sd_theta1", "SE / SD"),
                                 ("coverage_95_wilson_lower", "Coverage Wilson lower"),
                                 ("coverage_95_wilson_upper", "Coverage Wilson upper")]), "",
        "## How to read these diagnostics", "",
        "- **Consistency:** raw bias, empirical SD, and RMSE shrinking with n are relevant; a small raw bias alone is insufficient.",
        "- **Root-n behavior:** sqrt(n) SD should approach a finite positive constant and sqrt(n) bias should approach zero for the centered limit being assessed. "
        "Shrinking raw bias with persistent nonzero sqrt(n) bias is compatible with consistency but problematic for the centered claim in Theorem 3.3 if it persists beyond Monte Carlo and optimization noise.",
        "- **Asymptotic normality:** compare the whole distribution using QQ shape, skewness, excess kurtosis, and tails, in addition to location and scale. "
        "A nearly straight QQ curve with the wrong slope or intercept does not establish the desired N(0,1) studentized limit.",
        "- **Variance and CI calibration:** mean estimated SE / empirical SD should approach one; T mean/SD should approach 0/1, its quantiles should approach "
        "(-1.959964, 0, 1.959964), and coverage should approach 0.95. Studentization jointly checks the estimator and its estimated SE: "
        "a nonstandard T distribution can result from estimator bias, nonnormality, or a miscalibrated variance estimate. Coverage alone cannot distinguish these causes.", "",
        "The requested n is the total sample before the retained 80:20 training/validation split. The covariance uses n_train. "
        "summary.csv reports both sqrt(n) and sqrt(n_train) scalings; their fixed ratio changes the scale constant but not the studentized statistic or coverage.", "",
        "Separate root-n-error QQ plots compare sqrt(n)(theta1_hat-1) with a normal reference fitted to its empirical mean and SD. "
        "These plots and the saved root-n-error skewness/kurtosis help separate estimator shape from estimated-SE/studentization effects. "
        "Their fitted lines intentionally absorb location and scale: agreement with those lines does not validate zero root-n bias or the paper's estimated variance. "
        "The studentized QQ plots instead use the fixed N(0,1) reference y=x.", "",
        "## Observed pilot patterns", "",
    ]
    for case, frame in summary.groupby("case", sort=True):
        frame = frame.sort_values("n")
        first, last = frame.iloc[0], frame.iloc[-1]
        if len(frame) < 2:
            lines.append(f"- Case {case}: only n={int(first['n'])} is available. No across-n trend can yet be assessed.")
        else:
            lines.append(
                f"- Case {case}, n={int(first['n'])} to {int(last['n'])}: raw bias {_format(first['bias_theta1'])} to {_format(last['bias_theta1'])}; "
                f"sqrt(n) bias {_format(first['sqrt_n_bias_theta1'])} to {_format(last['sqrt_n_bias_theta1'])}; "
                f"sqrt(n) SD {_format(first['sqrt_n_sd_theta1'])} to {_format(last['sqrt_n_sd_theta1'])}; "
                f"T mean/SD {_format(first['t_mean'])}/{_format(first['t_sd'])} to {_format(last['t_mean'])}/{_format(last['t_sd'])}; "
                f"coverage {_format(first['coverage_95_theta1'])} to {_format(last['coverage_95_theta1'])}. "
                "These endpoint comparisons are descriptive; intermediate n values and Monte Carlo uncertainty must also be considered."
            )
    q_min, q_max = int(summary["q_studentized"].min()), int(summary["q_studentized"].max())
    lines.extend(["", f"Valid studentized repetitions range from {q_min} to {q_max} per group. "])
    if q_min < 2:
        lines.append("At least one group has fewer than two valid replications: this output is a smoke test only for those groups. "
                     "Empirical SD and inferential Monte Carlo precision are unavailable there; single-point quantiles and observed coverage are not distributional evidence.")
    if q_min < 50:
        lines.append("This is a small pilot. At Q=20, coverage moves in steps of 0.05 and its Monte Carlo SE is about 0.049 when true coverage is 0.95. "
                     "Tail quantiles, skewness, and kurtosis are particularly noisy; extending to Q=200 is needed before placing weight on apparent trends.")
    lines.extend([
        "", "Coverage uncertainty uses a 95% Wilson interval; the plug-in coverage MCSE is also saved, but can misleadingly be zero when observed coverage is 0 or 1. "
        "Bias MCSE is empirical SD/sqrt(Q). Across-n bias bars are approximate 95% Monte Carlo intervals, not confidence intervals for individual fitted coefficients. "
        "Empirical SD uses ddof=1. Skewness and excess kurtosis use SciPy's bias corrections and are unavailable below Q=3 and Q=4 respectively (or for a constant sample). "
        "Empirical quantiles use NumPy's linear interpolation; QQ plotting positions are (rank-0.5)/Q.", "",
        "This finite-sample experiment neither verifies nor disproves Theorem 3.3. A fixed network/training schedule can introduce approximation or optimization error "
        "that does not vanish at the rate required by the theorem. Persistent anomalies call for checking those errors and the variance estimator as well as adding repetitions and larger n.", "",
        "All detailed metrics are in summary.csv; quantiles.csv compares empirical and standard-normal quantiles. PNG figures show the distributions, QQ plots, and across-n patterns. "
        "No benchmark estimators or formal normality tests are used.", "",
    ])
    return "\n".join(lines)


def generate_report(raw: pd.DataFrame, output: Path, requested_repetitions: int) -> dict[str, Any]:
    """Save summary.csv, quantiles.csv, report.md, and theta1 PNG diagnostics.

    Returns paths in a dictionary. Missing case/n groups are omitted, never
    filled with artificial observations. A one-replication smoke test still
    produces readable artifacts with unavailable dispersions marked NA.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    required = {"case", "n", "theta1", "se_theta1"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing required reporting columns: {missing}")
    if raw.empty:
        raise ValueError("No completed replications are available for reporting.")
    if requested_repetitions < 1:
        raise ValueError("requested_repetitions must be positive")
    n_values = _numeric(raw, "n")
    if not np.all(np.isfinite(n_values) & (n_values > 0) & (n_values == np.floor(n_values))):
        raise ValueError("n must be a positive finite integer for each replication")
    if "tau" in raw and not np.allclose(_numeric(raw, "tau"), 0.5):
        raise ValueError("This report is restricted to tau=0.5")
    if "rep" in raw and raw.duplicated(["case", "n", "rep"]).any():
        raise ValueError("Duplicate case/n/rep rows would double-count replications")
    rows = [_summarize(group, case, int(n), requested_repetitions)
            for (case, n), group in raw.groupby(["case", "n"], sort=True)]
    summary = pd.DataFrame(rows).sort_values(["case", "n"]).reset_index(drop=True)
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False)
    quantile_rows = []
    for row in rows:
        for probability, name in zip(PROBABILITIES, QUANTILE_NAMES):
            quantile_rows.append({"case": row["case"], "n": row["n"], "q_studentized": row["q_studentized"],
                                  "probability": probability, "empirical_t_quantile": row[f"t_{name}"],
                                  "standard_normal_quantile": row[f"normal_{name}"],
                                  "difference": row[f"t_{name}_minus_normal"]})
    quantile_path = output / "quantiles.csv"
    pd.DataFrame(quantile_rows).to_csv(quantile_path, index=False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    figure_paths: list[Path] = []
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for case in summary["case"].drop_duplicates():
            figure_paths.extend(_distribution_figures(raw, summary, figures, case))
            figure_paths.append(_across_n_figure(summary, figures, case))
            figure_paths.append(_shape_figure(summary, figures, case))
            figure_paths.append(_root_error_qq_figure(raw, summary, figures, case))
    report_path = output / "report.md"
    report_path.write_text(_interpretation(summary, requested_repetitions), encoding="utf-8")
    return {"summary": summary_path, "quantiles": quantile_path, "report": report_path, "figures": figure_paths}


def timing_projection(raw: pd.DataFrame, output: Path, cases: Iterable[Any], sample_sizes: Iterable[int]) -> dict[str, Any]:
    """Project full Q=20/Q=200 wall time from measured completed replications.

    Exact case/n means are preferred. Unmeasured n is linearly interpolated
    between observed n values, or extrapolated proportionally to n from the
    nearest observed endpoint. An unmeasured case uses pooled observed cases.
    These assumptions are made explicit in both saved artifacts. The estimates
    assume sequential execution and unchanged fitting/inference settings.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases, sample_sizes = tuple(cases), tuple(sample_sizes)
    work = raw.copy()
    work["total_seconds"] = pd.to_numeric(work["total_seconds"], errors="coerce")
    work = work[np.isfinite(work["total_seconds"]) & (work["total_seconds"] > 0)]
    if work.empty:
        raise ValueError("At least one positive total_seconds measurement is required")
    rows = []
    for case in cases:
        own = work[work["case"] == case]
        source = own if not own.empty else work
        means = source.groupby("n", sort=True)["total_seconds"].mean()
        observed_n, observed_seconds = means.index.to_numpy(dtype=float), means.to_numpy(dtype=float)
        for n in sample_sizes:
            exact = own[own["n"] == n]
            prefix = "case-specific" if not own.empty else "pooled observed cases"
            if not exact.empty:
                seconds = float(exact["total_seconds"].mean())
                method = "measured case/n mean"
            elif n in means.index:
                seconds = float(means.loc[n])
                method = f"{prefix}, observed n"
            elif observed_n[0] < n < observed_n[-1]:
                seconds = float(np.interp(n, observed_n, observed_seconds))
                method = f"{prefix}, linear interpolation in n"
            else:
                index = 0 if n <= observed_n[0] else -1
                seconds = float(observed_seconds[index] * n / observed_n[index])
                method = f"{prefix}, proportional-to-n extrapolation"
            row = {"case": case, "n": int(n), "measured_repetitions": len(exact),
                   "estimated_seconds_per_replication": seconds, "method": method}
            for q in (20, 200):
                row[f"q{q}_full_seconds"] = q * seconds
                row[f"q{q}_remaining_seconds_if_reused"] = max(0, q - len(exact)) * seconds
            rows.append(row)
    frame = pd.DataFrame(rows)
    csv_path = output / "runtime_projection.csv"
    frame.to_csv(csv_path, index=False)
    full20, full200 = frame["q20_full_seconds"].sum(), frame["q200_full_seconds"].sum()
    remaining20 = frame["q20_remaining_seconds_if_reused"].sum()
    remaining200 = frame["q200_remaining_seconds_if_reused"].sum()
    lines = ["# Runtime projection from measured replications", "",
             f"Measured complete replications: {len(work)}. The projection covers {len(frame)} requested case/n groups.", "",
             f"- Q=20: **{full20 / 60:.1f} minutes ({full20 / 3600:.2f} hours)** for all fits; "
             f"{remaining20 / 60:.1f} minutes remaining if these checkpoints can be reused.",
             f"- Q=200: **{full200 / 60:.1f} minutes ({full200 / 3600:.2f} hours)** for all fits; "
             f"{remaining200 / 60:.1f} minutes remaining if these checkpoints can be reused.", "",
             "These are rough sequential wall-time estimates, not precision bounds. They require the same network, epoch cap, patience, optimizer, "
             "thread count, density routine, and auxiliary-projection settings as the measured fits. A deliberately shortened smoke run is not "
             "representative of a longer training configuration. Early stopping, process startup, and machine load can alter the result.", "",
             "runtime_projection.csv records each estimate and its source. Exact case/n mean timings are used when available. "
             "Missing n values use linear interpolation between measured n values, or proportional-to-n extrapolation from the nearest endpoint. "
             "Cases without measurements use pooled observed cases. Extrapolation from a single small n can be inaccurate. "
             "Report/figure generation overhead is excluded. Remaining-time estimates presume every measured row is a reusable compatible checkpoint.", "",
             "Q=200 is projected only; this function never launches fits.", ""]
    markdown_path = output / "runtime_projection.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "report": markdown_path,
            "q20_full_seconds": float(full20), "q200_full_seconds": float(full200),
            "q20_remaining_seconds_if_reused": float(remaining20), "q200_remaining_seconds_if_reused": float(remaining200)}
