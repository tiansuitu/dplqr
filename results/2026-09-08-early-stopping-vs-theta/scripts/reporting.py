"""Report paired stopping rules; never train a model or choose a checkpoint.

Inputs are already-selected checkpoint rows and independently computed oracle
diagnostics. A Monte Carlo replicate, not a stopping-rule row, is the sampling
unit. Epoch IQRs show between-replicate variation, not confidence intervals.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


RULE_KEYS = ["case", "analysis", "rule_id", "patience", "max_epochs"]
METRICS = (
    "selected_epoch", "stop_epoch", "theta_error_l2", "training_check_loss",
    "validation_check_loss", "test_check_loss", "m_l2_error", "q_l2_error",
    "theta_epoch_gap", "abs_theta_epoch_gap", "theta_regret",
    "theta_squared_regret", "q_regret", "best_validation_loss",
)
ORACLE_EPOCHS = ("e_theta", "e_q", "e_m", "e_test")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _stats(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return dict(mean=float(a.mean()), median=float(np.median(a)), sd=sd,
                mcse=sd / math.sqrt(len(a)), q25=float(np.quantile(a, .25)),
                q75=float(np.quantile(a, .75)))


def _summaries(raw):
    rows = []
    for keys, g in raw.groupby(RULE_KEYS, sort=True, dropna=False):
        row = dict(zip(RULE_KEYS, keys))
        row.update(case_name=str(g.case_name.iloc[0]), repetitions=len(g),
                   mc_sd_available=len(g) > 1,
                   restore_best=bool(g.restore_best.iloc[0]),
                   min_delta=float(g.min_delta.iloc[0]))
        for metric in METRICS:
            for statistic, value in _stats(g[metric]).items():
                row[f"{statistic}_{metric}"] = value
        if "empirical_validation_check_loss" in g:
            for statistic, value in _stats(g.empirical_validation_check_loss).items():
                row[f"{statistic}_empirical_validation_check_loss"] = value
        for j in (1, 2):
            truth = g[f"theta_true_{j}"].to_numpy(float)
            estimate = g[f"theta_hat_{j}"].to_numpy(float)
            error = estimate - truth
            if not np.all(truth == truth[0]):
                raise ValueError("Coefficient truth varies within a case/rule.")
            stat = _stats(estimate)
            row.update({f"theta_{j}_true": float(truth[0]),
                        f"theta_{j}_mean": stat["mean"],
                        f"theta_{j}_bias": float(error.mean()),
                        f"theta_{j}_sd": stat["sd"],
                        f"theta_{j}_bias_mcse": stat["mcse"],
                        f"theta_{j}_rmse": float(np.sqrt(np.mean(error ** 2)))})
        row["theta_vector_rmse"] = float(np.sqrt(np.mean(g.theta_error_l2 ** 2)))
        row["proportion_early_stopping"] = float(g.stopping_reason.eq("early_stopping").mean())
        row["proportion_max_epochs"] = float(g.stopping_reason.eq("max_epochs").mean())
        gap = g.theta_epoch_gap.to_numpy(float)
        row.update(proportion_before_theta_oracle=float(np.mean(gap < 0)),
                   proportion_equal_theta_oracle=float(np.mean(gap == 0)),
                   proportion_after_theta_oracle=float(np.mean(gap > 0)),
                   proportion_within_10_epochs=float(np.mean(np.abs(gap) <= 10)),
                   proportion_within_25_epochs=float(np.mean(np.abs(gap) <= 25)))
        rows.append(row)
    return pd.DataFrame(rows)


def _oracle_summary(oracles):
    rows = []
    optional = [c for c in oracles if c.startswith("oracle_") and
                pd.api.types.is_numeric_dtype(oracles[c])]
    for case, g in oracles.groupby("case", sort=True):
        row = dict(case=int(case), case_name=str(g.case_name.iloc[0]),
                   repetitions=len(g), mc_sd_available=len(g) > 1)
        for metric in [*ORACLE_EPOCHS, *optional]:
            for stat, value in _stats(g[metric]).items():
                row[f"{stat}_{metric}"] = value
        for first, second in (("e_q", "e_theta"), ("e_m", "e_theta"), ("e_test", "e_theta")):
            gap = g[first].to_numpy(float) - g[second].to_numpy(float)
            row[f"mean_{first}_minus_{second}"] = float(gap.mean())
            row[f"mean_abs_{first}_minus_{second}"] = float(np.abs(gap).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _cap_comparisons(raw):
    """Compare each finite cap to the largest cap on the same replicate."""
    rows = []
    caps = raw.loc[raw.analysis.eq("max_epochs")]
    for case, g in caps.groupby("case", sort=True):
        cap_values = sorted(g.max_epochs.unique())
        largest = g.loc[g.max_epochs.eq(cap_values[-1])].set_index("replicate").sort_index()
        for cap in cap_values:
            current = g.loc[g.max_epochs.eq(cap)].set_index("replicate").sort_index()
            if not current.index.equals(largest.index):
                raise ValueError("Maximum-epoch contrasts contain unpaired replicates.")
            same_epoch = current.selected_epoch.to_numpy() == largest.selected_epoch.to_numpy()
            coefficient_change = np.linalg.norm(
                current[["theta_hat_1", "theta_hat_2"]].to_numpy(float) -
                largest[["theta_hat_1", "theta_hat_2"]].to_numpy(float), axis=1)
            row = dict(case=int(case), case_name=str(g.case_name.iloc[0]),
                       max_epochs=int(cap), reference_max_epochs=int(cap_values[-1]),
                       repetitions=len(current), same_selected_epoch_count=int(same_epoch.sum()),
                       proportion_same_selected_epoch=float(same_epoch.mean()),
                       exactly_same_coefficients_count=int(np.count_nonzero(coefficient_change == 0)),
                       mean_coefficient_difference_l2=float(coefficient_change.mean()),
                       max_coefficient_difference_l2=float(coefficient_change.max()))
            for metric in ("selected_epoch", "theta_error_l2", "test_check_loss", "q_l2_error"):
                changes = current[metric].to_numpy(float) - largest[metric].to_numpy(float)
                stat = _stats(changes)
                row[f"mean_paired_{metric}_difference"] = stat["mean"]
                row[f"mcse_paired_{metric}_difference"] = stat["mcse"]
            rows.append(row)
    return pd.DataFrame(rows)


def _endpoint_comparisons(raw):
    """Descriptive selected-to-ceiling contrasts, without declaring stability."""
    rows = []
    refs = raw.loc[raw.analysis.eq("reference") & raw.rule_id.eq("no_early_stopping")]
    main = raw.loc[raw.analysis.eq("patience") & raw.patience.eq(15)]
    for case, g in main.groupby("case", sort=True):
        before = g.set_index("replicate").sort_index()
        after = refs.loc[refs.case.eq(case)].set_index("replicate").sort_index()
        if after.empty:
            continue
        if not before.index.equals(after.index):
            raise ValueError("No-stopping reference is not paired with patience 15.")
        delta = {m: after[m].to_numpy(float) - before[m].to_numpy(float) for m in
                 ("training_check_loss", "validation_check_loss", "test_check_loss",
                  "theta_error_l2", "m_l2_error", "q_l2_error")}
        train_down = delta["training_check_loss"] < 0
        val_up, test_up = delta["validation_check_loss"] > 0, delta["test_check_loss"] > 0
        theta_up, q_up = delta["theta_error_l2"] > 0, delta["q_l2_error"] > 0
        theta_close = before.abs_theta_epoch_gap.to_numpy(float) <= 25
        row = dict(case=int(case), case_name=str(g.case_name.iloc[0]), repetitions=len(g),
                   reference_epoch=int(after.selected_epoch.iloc[0]),
                   training_down=int(train_down.sum()), validation_up=int(val_up.sum()),
                   test_up=int(test_up.sum()), theta_up=int(theta_up.sum()), q_up=int(q_up.sum()),
                   train_down_val_up_test_up_theta_up=int((train_down & val_up & test_up & theta_up).sum()),
                   val_test_q_nonincreasing_theta_up=int((~val_up & ~test_up & ~q_up & theta_up).sum()),
                   within_25_oracle_then_theta_test_q_up=int((theta_close & theta_up & test_up & q_up).sum()))
        for metric, change in delta.items():
            row[f"selected_mean_{metric}"] = float(before[metric].mean())
            row[f"ceiling_mean_{metric}"] = float(after[metric].mean())
            row[f"mean_paired_change_{metric}"] = float(change.mean())
            row[f"mcse_paired_change_{metric}"] = _stats(change)["mcse"]
        rows.append(row)
    return pd.DataFrame(rows)


def _make_figures(raw, summary, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    files = []

    def save(fig, stem, title, case, name, q, note):
        fig.suptitle(f"{title}\nCase {case}: {name}; Q = {q}; n = {int(raw.n.iloc[0])}; "
                     f"tau = {float(raw.tau.iloc[0]):g}", fontsize=13, y=.985)
        fig.text(.5, .015, note, ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=(0, .075, 1, .89))
        for extension in ("png", "pdf"):
            path = destination / f"case_{case}_{stem}.{extension}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            files.append(str(path))
        plt.close(fig)

    def categorical(ax, values, label):
        x = np.arange(len(values))
        ax.set_xticks(x, [str(int(v)) for v in values])
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
        return x

    def epoch_curves(ax, s, x):
        median = s.median_selected_epoch.to_numpy(float)
        ax.errorbar(x, median, yerr=np.vstack((median - s.q25_selected_epoch,
                    s.q75_selected_epoch - median)), fmt="o-", color=COLORS[0],
                    capsize=4, label="Median and interquartile range", zorder=3)
        ax.plot(x, s.mean_selected_epoch, "s--", color=COLORS[1], markersize=4,
                label="Mean selected epoch", zorder=3)
        ax.set_ylabel("Selected checkpoint epoch")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")

    for case in sorted(raw.case.unique()):
        cr = raw.loc[raw.case.eq(case)]
        cs = summary.loc[summary.case.eq(case)]
        p = cs.loc[cs.analysis.eq("patience")].sort_values("patience")
        caps = cs.loc[cs.analysis.eq("max_epochs")].sort_values("max_epochs")
        name, q = str(cr.case_name.iloc[0]), int(cr.replicate.nunique())
        if p.empty or caps.empty:
            raise ValueError("Both patience and maximum-epoch analyses are required for figures.")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), squeeze=False)
        for j, ax in enumerate(axes.flat, start=1):
            x = categorical(ax, p.patience, "Patience (epochs without sufficient improvement)")
            ax.plot(x, p[f"theta_{j}_rmse"], "o-", color=COLORS[j - 1])
            ax.set_ylabel(rf"Monte Carlo RMSE of $\hat\theta_{j}$")
            ax.set_title(rf"Coefficient $\theta_{j}$")
            ax.set_ylim(bottom=0)
        save(fig, "A_patience_theta_rmse", "A. Patience and coefficient estimation", case, name, q,
             "RMSE = sqrt(mean squared coefficient error across independent replicates). Same path within each replicate.")

        fig, ax = plt.subplots(figsize=(8.5, 5))
        x = categorical(ax, p.patience, "Patience (epochs without sufficient improvement)")
        epoch_curves(ax, p, x)
        save(fig, "B_patience_selected_epoch", "B. Patience and returned checkpoint", case, name, q,
             "Bars show the middle 50% of replicate-selected epochs, not confidence intervals. Best validation weights are restored.")

        fig, ax = plt.subplots(figsize=(8.5, 5))
        x = categorical(ax, caps.max_epochs, "Maximum training epochs")
        for _, g in cr.loc[cr.analysis.eq("max_epochs")].groupby("replicate"):
            ax.plot(x, g.sort_values("max_epochs").selected_epoch, color="#A0A0A0", alpha=.3, linewidth=.8)
        epoch_curves(ax, caps, x)
        save(fig, "C_cap_selected_epoch", "C. Increasing the ceiling with patience fixed at 15", case, name, q,
             "Gray lines: the same replicate under different ceilings. Median with IQR and mean summarize selected checkpoints.")

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), squeeze=False)
        for ax, s, label, prefix, color in zip(axes.flat, (p, caps), ("Vary patience", "Vary maximum epochs"),
                                               ("p", "cap"), COLORS[:2]):
            ax.scatter(s.mean_q_l2_error, s.theta_vector_rmse, color=color, s=42, zorder=3)
            variable = "patience" if prefix == "p" else "max_epochs"
            # Exact estimator plateaus otherwise draw several labels on one point.
            labels = {}
            for _, row in s.iterrows():
                location = (round(float(row.mean_q_l2_error), 12), round(float(row.theta_vector_rmse), 12))
                labels.setdefault(location, []).append(int(row[variable]))
            for index, (location, values) in enumerate(labels.items()):
                offset = (7, 9 + (index % 2) * 9) if index % 2 == 0 else (7, -12 - (index % 3) * 7)
                label_text = f"{prefix}=" + ",".join(str(v) for v in values)
                ax.annotate(label_text, location, xytext=offset,
                            textcoords="offset points", fontsize=8,
                            arrowprops=dict(arrowstyle="-", color="#AAAAAA", linewidth=.6))
            ax.set_title(label)
            ax.set_xlabel("Mean full-quantile L2 error")
            ax.set_ylabel("Vector coefficient RMSE")
            ax.margins(.25)
            ax.grid(alpha=.2)
        save(fig, "D_prediction_theta_error", "D. Prediction and coefficient error by stopping rule", case, name, q,
             "Vector coefficient RMSE = sqrt(mean ||theta_hat − theta_true||²). Lower is better on both axes. Labels denote rules.")

        patience_values = sorted(p.patience.astype(int))
        columns = min(3, len(patience_values))
        row_count = math.ceil(len(patience_values) / columns)
        fig, axes = plt.subplots(row_count, columns, figsize=(4 * columns, 3.6 * row_count), squeeze=False)
        maximum = max(float(cr.e_theta.max()), float(cr.selected_epoch.max()), 1) * 1.04
        for ax, patience in zip(axes.flat, patience_values):
            g = cr.loc[cr.analysis.eq("patience") & cr.patience.eq(patience)]
            ax.plot([0, maximum], [0, maximum], "--", color="#555555", linewidth=1)
            ax.scatter(g.e_theta, g.selected_epoch, color=COLORS[0], s=34, alpha=.8)
            ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Infeasible theta-oracle epoch",
                   ylabel="Validation-selected epoch", title=f"Patience = {patience}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(patience_values):]:
            ax.set_visible(False)
        save(fig, "E_selected_theta_oracle", "E. Validation selection versus the theta oracle", case, name, q,
             "Each point is one replicate. Below the diagonal: validation selects earlier than the theta oracle. Oracles do not tune models.")
    return files


def _format(value, digits=3):
    return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "NA"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])


def _write_results(raw, summary, oracle_summary, caps, endpoints, output, metadata):
    counts = raw.groupby("case").replicate.nunique()
    default = summary.loc[summary.analysis.eq("patience") & summary.patience.eq(15)]
    ceiling = int(raw.loc[raw.analysis.eq("patience"), "max_epochs"].max())
    lines = ["# Validation early stopping versus theta estimation", "",
             "This experiment applies all stopping rules to the same continuous optimization path within each "
             "replicate. Only the stopping rule changes. The theta, prediction, nuisance and test-loss oracles "
             "are infeasible evaluation benchmarks and never determine the practical stopping rule.", "",
             "Completed analysis: **" + "; ".join(f"Case {int(c)}: {int(n)} replicates" for c, n in counts.items()) +
             f"**; n = {int(raw.n.iloc[0])}, tau = {float(raw.tau.iloc[0]):g}, common full-path ceiling = {ceiling}. "
             f"There are {len(raw)} selected-rule rows. Only these completed replicates support the summaries.", "",
             ("These are pilot Monte Carlo results. " if counts.min() < 100 else "These are Monte Carlo results. ") +
             "Replicate-level SD, MCSE and IQR are descriptive with this sample size. "
             "This study evaluates coefficient estimation, not confidence-interval coverage or "
             "the validity of inferential procedures.", "",
             "## Practical rule and oracle alignment", "",
             "The main rule restores the best validation weights. Its returned checkpoint (`selected_epoch`) "
             "can therefore precede the epoch at which training terminates (`stop_epoch`). The stopping decision "
             "uses only training-epoch order and validation check loss, with the repository's min_delta and "
             "patience convention recorded in run_config.json. Test loss and true functions are evaluation only.", "",
             "The repository's actual validation criterion averages batch check-loss means with equal batch "
             "weights. With 200 validation observations and batch size 128, that gives equal weight to the "
             "128-observation and 72-observation batch means. It differs from the empirical mean over all "
             "200 observations. `validation_check_loss` records this actual stopping criterion; "
             "`empirical_validation_check_loss` separately preserves the full-sample diagnostic. "
             "The first checkpoint attaining the qualifying best validation loss is retained when later "
             "epochs tie it; the configured improvement and patience-counting convention is preserved.", "",
             "The following table fixes patience at 15 and the common full-path ceiling. A negative epoch gap "
             "means validation selects earlier than the coefficient oracle.", "",
             _table(["Case", "Mean selected", "Mean stop", "Mean theta oracle", "Mean gap", "Within ±25", "Mean theta regret"],
                    [[int(r.case), _format(r.mean_selected_epoch, 1), _format(r.mean_stop_epoch, 1),
                      _format(oracle_summary.set_index("case").loc[r.case, "mean_e_theta"], 1),
                      _format(r.mean_theta_epoch_gap, 1), f"{r.proportion_within_25_epochs:.0%}",
                      _format(r.mean_theta_regret)] for r in default.itertuples()]), ""]
    for r in default.itertuples():
        lines += [f"Case {int(r.case)}: validation selects before the theta oracle in "
                  f"{r.proportion_before_theta_oracle:.0%} of replicates, exactly at it in "
                  f"{r.proportion_equal_theta_oracle:.0%}, and after it in {r.proportion_after_theta_oracle:.0%}. "
                  f"The mean absolute epoch gap is {r.mean_abs_theta_epoch_gap:.1f} epochs. "
                  "A distant epoch is not by itself a large estimation loss; the regret quantifies the actual "
                  "increase in coefficient error relative to the oracle.", ""]
    lines += ["## Coefficient and prediction performance", "",
              _table(["Case", "Patience", "Theta 1 bias", "Theta 1 RMSE", "Theta 2 bias", "Theta 2 RMSE", "Vector RMSE", "q L2", "Test loss"],
                     [[int(r.case), int(r.patience), _format(r.theta_1_bias), _format(r.theta_1_rmse),
                       _format(r.theta_2_bias), _format(r.theta_2_rmse), _format(r.theta_vector_rmse),
                       _format(r.mean_q_l2_error), _format(r.mean_test_check_loss)] for r in
                      summary.loc[summary.analysis.eq("patience")].sort_values(["case", "patience"]).itertuples()]), "",
              "Component RMSE is sqrt(mean squared signed coefficient error). Vector RMSE is "
              "sqrt(mean squared Euclidean coefficient error); it differs from the mean Euclidean error, "
              "which is also saved. SD uses ddof = 1 across replicates. The q and m errors are absolute L2 "
              "errors on the fixed independent evaluation sample, not relative MSE.", "",
              "## When does a higher epoch ceiling stop changing the estimator?", "",
              "Each row below compares the given ceiling with the largest ceiling on the same replicate, "
              "with patience fixed at 15. Identical selected epochs refer to exactly the same checkpoint.", "",
              _table(["Case", "Cap", "Reference cap", "Same selected checkpoint", "Largest coefficient difference (L2)"],
                     [[int(r.case), int(r.max_epochs), int(r.reference_max_epochs),
                       f"{int(r.same_selected_epoch_count)}/{int(r.repetitions)}",
                       _format(r.max_coefficient_difference_l2, 6)] for r in caps.itertuples()]), ""]
    for case, g in caps.groupby("case", sort=True):
        g = g.sort_values("max_epochs")
        plateau = next((int(r.max_epochs) for r in g.itertuples() if
                        g.loc[g.max_epochs.ge(r.max_epochs), "proportion_same_selected_epoch"].eq(1).all()), None)
        lines += [f"Case {int(case)}: among the tested ceilings, every replicate matches the largest-ceiling "
                  f"selected checkpoint from cap {plateau} onward. This is a statement about this run and "
                  "the tested caps, not a guarantee for new datasets.", ""]
    lines += ["## Distinguishing the scientific patterns", "",
              "1. **Ordinary prediction overfitting:** later training lowers training loss while validation/test "
              "loss and coefficient error rise.",
              "2. **Prediction–coefficient tension:** prediction remains good or improves while coefficient "
              "error rises. This shows that a prediction criterion need not optimize theta estimation.",
              "3. **Early stopping protects both:** the selected epoch is close to the theta oracle and later "
              "training worsens both prediction and coefficient error.", "",
              "These patterns can coexist within a case or differ across replicates. They are not exhaustive "
              "and the results are not assigned to a pattern by assumption.", ""]
    if not endpoints.empty:
        lines += [f"For a concrete paired comparison, the table below follows each patience-15 selected checkpoint "
                  f"to epoch {ceiling} on that same path. These are endpoint comparisons and do not imply "
                  "monotonic change at every intervening epoch.", "",
                  _table(["Case", "Q", "Train↓, val↑, test↑, theta↑", "Val/test/q no worse, theta↑", "Within ±25 then theta/test/q↑"],
                         [[int(r.case), int(r.repetitions), int(r.train_down_val_up_test_up_theta_up),
                           int(r.val_test_q_nonincreasing_theta_up), int(r.within_25_oracle_then_theta_test_q_up)]
                          for r in endpoints.itertuples()]), "",
                  "Arrows are literal directional comparisons. 'No worse' means a nonpositive measured change "
                  "in every listed metric; no data-dependent tolerance for 'stable' is introduced. The ±25 epoch "
                  "window is a prespecified descriptive proximity summary, not an optimality test. These counts "
                  "illustrate the three possible patterns without treating an endpoint sign as proof of a mechanism.", ""]
        for r in endpoints.itertuples():
            lines += [f"Case {int(r.case)}, selected → {ceiling}: mean coefficient L2 error "
                      f"{r.selected_mean_theta_error_l2:.3f} → {r.ceiling_mean_theta_error_l2:.3f}; "
                      f"mean q L2 error {r.selected_mean_q_l2_error:.3f} → {r.ceiling_mean_q_l2_error:.3f}; "
                      f"mean test loss {r.selected_mean_test_check_loss:.3f} → {r.ceiling_mean_test_check_loss:.3f}.", ""]
            if (r.mean_paired_change_theta_error_l2 < 0 and r.mean_paired_change_q_l2_error > 0
                    and r.mean_paired_change_test_check_loss > 0):
                lines += ["Here the endpoint means show a timing mismatch in the reverse direction from "
                          "Pattern 2: continued training improves coefficient estimation while worsening "
                          "prediction. Validation-based selection may favor a substantially earlier checkpoint "
                          "than coefficient estimation favors. This is a descriptive paired result, not a "
                          "reason to use the infeasible oracle in practical training.", ""]
    lines += ["The infeasible theta oracle minimizes error over all recorded epochs of the full path. Its "
              "advantage is optimistic by construction and cannot be achieved by consulting unknown theta in "
              "real data. A positive oracle regret demonstrates a gap to that benchmark; it alone does not "
              "prove early stopping is ineffective, nor does good quantile prediction establish valid theta inference.", "",
              "## Figures", "",
              "Each case has separate PNG and vector PDF versions. Epoch bars are replicate IQRs. "
              "There are no confidence bands on RMSE curves.", ""]
    captions = (("A_patience_theta_rmse", "A. Patience versus component RMSE"),
                ("B_patience_selected_epoch", "B. Patience versus selected epoch"),
                ("C_cap_selected_epoch", "C. Maximum epochs versus selected epoch"),
                ("D_prediction_theta_error", "D. Quantile prediction error versus vector coefficient RMSE"),
                ("E_selected_theta_oracle", "E. Selected epoch versus the theta oracle, by patience"))
    for case in sorted(raw.case.unique()):
        lines += [f"### Case {int(case)}", ""]
        for stem, caption in captions:
            base = f"figures/case_{int(case)}_{stem}"
            lines += [caption, "", f"![{caption}]({base}.png)", "", f"[Vector PDF]({base}.pdf)", ""]
    lines += ["## Machine-readable outputs", "",
              "- [Selected-rule estimates and stopping decisions](early_stopping_results.csv)",
              "- [Patience Monte Carlo summaries](monte_carlo_summary_patience.csv)",
              "- [Maximum-epoch Monte Carlo summaries](monte_carlo_summary_max_epochs.csv)",
              "- [Oracle epoch summaries](oracle_epoch_summary.csv)",
              "- [Per-replicate oracle diagnostics](oracle_epochs.csv)",
              "- [Paired ceiling comparisons](paired_max_epoch_comparisons.csv)",
              "- [Selected-to-ceiling directional comparisons](selected_to_ceiling_comparisons.csv)",
              "- [Run configuration and provenance](run_config.json)", ""]
    path = output / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_reports(raw: pd.DataFrame, oracles: pd.DataFrame, output: Path, metadata: dict) -> dict:
    """Write analysis-only CSV summaries and A–E PNG/PDF figures per case."""
    if raw.empty or oracles.empty:
        raise ValueError("Cannot summarize an empty stopping-rule or oracle table.")
    required = {"case", "case_name", "replicate", "analysis", "rule_id", "patience", "max_epochs",
                "n", "tau", "stopping_reason", "restore_best", "min_delta", *METRICS, *ORACLE_EPOCHS,
                "theta_hat_1", "theta_hat_2", "theta_true_1", "theta_true_2"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing stopping-rule report columns: {sorted(missing)}")
    if raw.duplicated(["case", "replicate", "analysis", "rule_id"]).any():
        raise ValueError("Duplicate case/replicate/stopping-rule rows.")
    if oracles.duplicated(["case", "replicate"]).any():
        raise ValueError("Oracle rows must contain each independent replicate exactly once.")
    nonnumeric = {"case_name", "analysis", "rule_id", "stopping_reason", "restore_best"}
    if not np.isfinite(raw[list(required - nonnumeric)].to_numpy(float)).all():
        raise ValueError("Nonfinite required diagnostics cannot be reported as valid results.")
    if not raw.stopping_reason.isin(["early_stopping", "max_epochs"]).all():
        raise ValueError("Unrecognized stopping reason.")
    if (raw.selected_epoch > raw.stop_epoch).any() or (raw.stop_epoch > raw.max_epochs).any():
        raise ValueError("Selected, stop and maximum epoch ordering is invalid.")
    if not raw.loc[raw.analysis.ne("reference"), "restore_best"].all():
        raise ValueError("This main-analysis report expects restored best validation weights.")
    for _, g in raw.groupby("case"):
        replicate_sets = g.groupby(["analysis", "rule_id"]).replicate.apply(lambda v: tuple(sorted(v)))
        if replicate_sets.nunique() != 1:
            raise ValueError("Stopping-rule comparisons are not paired on the same replicates.")
    raw = raw.sort_values(["case", "analysis", "patience", "max_epochs", "replicate"]).copy()
    oracles = oracles.sort_values(["case", "replicate"]).copy()
    if "case_name" not in oracles:
        oracles["case_name"] = oracles.case.map(raw.groupby("case").case_name.first())
    if not set(ORACLE_EPOCHS).issubset(oracles):
        raise ValueError("Missing oracle epoch definitions.")
    if not set(map(tuple, raw[["case", "replicate"]].drop_duplicates().to_numpy())).issubset(
            set(map(tuple, oracles[["case", "replicate"]].to_numpy()))):
        raise ValueError("A selected-rule replicate is missing from the oracle table.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = _summaries(raw)
    oracle_summary = _oracle_summary(oracles)
    caps, endpoints = _cap_comparisons(raw), _endpoint_comparisons(raw)
    files = []
    tables = [(summary.loc[summary.analysis.eq("patience")], "monte_carlo_summary_patience.csv"),
              (summary.loc[summary.analysis.eq("max_epochs")], "monte_carlo_summary_max_epochs.csv"),
              (summary.loc[summary.analysis.eq("reference")], "monte_carlo_summary_reference.csv"),
              (oracle_summary, "oracle_epoch_summary.csv"), (oracles, "oracle_epochs.csv"),
              (caps, "paired_max_epoch_comparisons.csv"), (endpoints, "selected_to_ceiling_comparisons.csv")]
    for frame, filename in tables:
        destination = output / filename
        frame.to_csv(destination, index=False, na_rep="NaN")
        files.append(str(destination))
    figures = _make_figures(raw, summary, output)
    report = _write_results(raw, summary, oracle_summary, caps, endpoints, output, metadata)
    return dict(selected_rule_rows=len(raw), independent_trajectories=len(oracles),
                summary_rows=len(summary), figure_pairs=len(figures) // 2,
                artifacts=[*files, *figures, report])
