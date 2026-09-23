from pathlib import Path
import nbformat as nbf

base = Path('results/2026-09-23_psi_n_thm33')
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip() + '\n'))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + '\n'))


md(r'''
# Fitted scores, asymptotic expansion, and scaled bias

This notebook asks whether the fitted-score condition, the two expansion remainders,
and the **root-$n$-scaled signed bias** become small as sample size increases.
It also compares the original Adam protocol with additional full-batch optimization
on the **same data and starting fit**.

Your original `run/` results are read below without alteration. The previous executed
notebook is preserved as `psi_n_diagnostic_before_expansion_v2.ipynb`.
New runs use a separate configuration- and code-specific `run_expansion_v2/` folder.

**Run All is safe by default:** it analyzes the old results but does not start new training.
Set `RUN_EXPERIMENT = True` when ready. `PROFILE = "pilot"` is a pipeline/sensitivity
check; use `"main"` and increase replications to resolve small biases.
Computational functions are in [diagnostic_helpers.py](diagnostic_helpers.py).
''')

md(r'''
## The bias question

The requested target is

$$
b_n=\sqrt n\,E[\hat\theta-\theta_0]\longrightarrow0.
$$

With $B$ independent Monte Carlo replications, estimate it by

$$
\hat b_n=\frac1B\sum_{b=1}^B\sqrt n(\hat\theta_b-\theta_0),\qquad
\widehat{\mathrm{SE}}(\hat b_n)=
\frac{\mathrm{SD}_b[\sqrt n(\hat\theta_b-\theta_0)]}{\sqrt B}.
$$

The tables and plots include Student-$t$ 95% Monte Carlo intervals. These describe uncertainty
in **the estimated bias**, not confidence intervals for $\theta_0$ in one dataset.
An interval containing zero is not evidence that the bias is negligible. The plot includes
a configurable practical tolerance; an interval wholly inside that band gives stronger
finite-sample evidence of small bias, but still does not prove a limit.

By contrast, $\sqrt nE|\hat\theta-\theta_0|$, $\sqrt{nE(\hat\theta-\theta_0)^2}$,
and the SD of $\sqrt n(\hat\theta-\theta_0)$ generally have **nonzero** limits.
They are reported as calibration measures, not vanishing targets.
Convergence in distribution to a centered normal alone does not imply $b_n\to0$;
convergence of means additionally needs moment control, such as uniform integrability.
''')

code('''
from pathlib import Path
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

# Locate this experiment from either the repository root or the notebook directory.
BASE = None
for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    for candidate in [parent, parent / "results" / "2026-09-23_psi_n_thm33"]:
        if (candidate / "diagnostic_helpers.py").exists():
            BASE = candidate
            break
    if BASE is not None:
        break
if BASE is None:
    raise FileNotFoundError("Open the notebook from its folder or the dplqr repository")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
import diagnostic_helpers as diag
diag = importlib.reload(diag)
pd.set_option("display.max_columns", 20)
print("Experiment:", BASE)
print(f"J = {diag.J:.8f}; Gaussian-model root-n SD benchmark = {diag.ASYMPTOTIC_SD:.6f}")
''')

code('''
RUN_EXPERIMENT = False
PROFILE = "pilot"       # "smoke", "pilot", or "main"
WIDTH_MODE = "fixed"    # "growing" is an exploratory network-size comparison
CONFIG = diag.make_config(PROFILE, WIDTH_MODE)

# Editable examples (edit before running):
# CONFIG["reps"] = 200
# CONFIG["n_values"] = [500, 1000, 2000, 4000, 8000]
# CONFIG["polish_epochs"] = 1500
# CONFIG["integration_power"] = 14  # points per scramble = 2**power
diag.validate_config(CONFIG)
display(pd.Series(CONFIG, name="value").to_frame())
RUN_DIR = diag.run_directory(BASE, CONFIG)
print("New run directory:", RUN_DIR)
print("Planned datasets:", len(CONFIG["n_values"]) * CONFIG["reps"])
print("Each dataset has baseline and polished diagnostics; models are saved.")
''')

md(r'''
## What the existing 30-replication run already says

The old CSV contains $\hat\theta$ and the data-generation seed. Its DGP is validated before
reconstructing the true errors and oracle score. Thus we can recover the scaled signed bias
and the **total linear-expansion residual** without retraining.
The fitted neural functions were not saved, so we cannot retrospectively separate
the empirical-process and Taylor remainders. The old files remain unchanged.
''')

code('''
legacy = diag.legacy_diagnostics(BASE)
if legacy.empty:
    print("No legacy run found. New-run cells below are still usable.")
else:
    legacy_bias = diag.bias_summary(legacy)
    display(legacy_bias[["n", "completed_reps", "root_n_bias", "mc_se", "ci_low", "ci_high",
                         "root_n_sd", "root_n_mae", "root_n_rmse", "oracle_wald_coverage"]])
    legacy_terms = []
    for n, group in legacy.groupby("n"):
        for metric in ["S", "R_score", "R_theta", "E_plus_T", "W"]:
            mean, se, low, high = diag.mean_interval(group[metric].abs())
            legacy_terms.append(dict(n=n, term=metric, mean_abs=mean, mc_se=se,
                                     ci_low=max(0, low), ci_high=high,
                                     q90_abs=group[metric].abs().quantile(.9)))
    display(pd.DataFrame(legacy_terms))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g = legacy_bias.sort_values("n")
    ax.errorbar(g.n, g.root_n_bias, yerr=[g.root_n_bias-g.ci_low, g.ci_high-g.root_n_bias],
                marker="o", capsize=4, label="legacy baseline: 95% Monte Carlo interval")
    ax.axhline(0, color="black", linewidth=.8)
    ax.axhspan(-CONFIG["bias_tolerance"], CONFIG["bias_tolerance"], color="gray", alpha=.15)
    ax.set(xlabel="n", ylabel=r"$\\sqrt{n}\\,\\widehat{E}(\\hat\\theta-\\theta_0)$",
           title="Existing run: scaled signed bias")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
''')

md(r'''
## Exact decomposition and the vanishing targets

Keep the score normalization

$$
\Psi_n(\zeta,h)=-\frac1n\sum_i
\left[\tau-\mathbf1\{\epsilon_i-\zeta\tilde X_i-h(Z_i)<0\}\right]\tilde X_i,
\qquad \tilde X=X-\varphi^*(Z).
$$

Write $\hat\zeta=\hat\theta-\theta_0$ and
$\hat h=\hat m-m_0+\hat\zeta\varphi^*$. Then the fitted residual is exactly
$\hat r_i=\epsilon_i-\hat\zeta\tilde X_i-\hat h(Z_i)$.
Let $\Psi_0$ be the population score and $J=E[f(0\mid X,Z)\tilde X^2]$.

$$
\begin{aligned}
S_n &= \sqrt n\,\Psi_n(\hat\zeta,\hat h),\\
W_n &= \sqrt n\,\Psi_n(0,0),\\
E_n &= \sqrt n\left\{(\Psi_n-\Psi_0)(\hat\zeta,\hat h)
                         -(\Psi_n-\Psi_0)(0,0)\right\},\\
T_n &= \sqrt n\left\{\Psi_0(\hat\zeta,\hat h)-J\hat\zeta\right\},\\
R_n^{\mathrm{score}} &= J\sqrt n(\hat\theta-\theta_0)+W_n
                       =S_n-E_n-T_n,\\
R_n^{\theta} &= \sqrt n(\hat\theta-\theta_0)+J^{-1}W_n.
\end{aligned}
$$

| Saved column | Meaning | Target |
|---|---|---|
| `S` | Fitted projected score | $o_p(1)$ |
| `E` | Empirical-process increment | $o_p(1)$ |
| `T` | Population Taylor remainder | $o_p(1)$ |
| `R_score`, `R_theta` | Total linear-expansion residual, in two units | $o_p(1)$ |
| `W` | Oracle empirical score at the truth | Nondegenerate; should **not** vanish |
| `root_n_G` | Ordinary parameter score | Optimization diagnostic |
| `root_n_phi_score` | Difference between projected and ordinary scores | Directional diagnostic |
| `sqrt_n_prediction_error_squared` | $\sqrt n\,d^2(\hat\beta,\beta_0)$ | Sufficient rate diagnostic in the usual remainder bound |
| `n_quarter_h_l2`, `n_quarter_m_l2` | Nuisance errors on an independent integration design | Rate diagnostics |

For this DGP, $J=0.25/\sqrt{2\pi}$ and the usual linear representation is

$$
\sqrt n(\hat\theta-\theta_0)=-J^{-1}W_n+o_p(1).
$$

**Normalization note:** this follows by differentiating the score defined above.
The supplement's final page prints a factor $2$ in the population derivative and a positive
sign in its final representation; these appear inconsistent with its stated score and with
the main paper's $\Sigma_2=E[f(0\mid U)\tilde X\tilde X^\top]$.
The code uses the derivative of the explicit score, and tests it numerically.

The decomposition check is an **algebraic implementation check**, not empirical evidence
for the expansion: separate magnitudes and tail probabilities are the substantive tests.
`R_score` and `R_theta` are computed directly from the fit and the oracle score and do not
depend on numerical population integration.
''')

md(r'''
## Accurate population integration for this DGP

Here $\tilde X=V\sim N(0,\sigma_V^2)$, $\sigma_V=0.5$, and
$\epsilon\sim N(0,1)$ are mutually independent and independent of $Z$.
With $\Phi,\phi$ denoting the standard normal CDF and density, Gaussian integration gives

$$
\begin{aligned}
\Psi_0(\zeta,h)
&=E_ZE_V\!\left[V\{\Phi(\zeta V+h(Z))-1/2\}\right]\\
&=\frac{\sigma_V^2\zeta}{\sqrt{1+\sigma_V^2\zeta^2}}
E_Z\!\left[\phi\!\left(\frac{h(Z)}{\sqrt{1+\sigma_V^2\zeta^2}}\right)\right].
\end{aligned}
$$

Only the two-dimensional expectation over $Z$ needs numerical integration. Each fitted model
is evaluated on independent, scrambled Sobol designs that are separate from training.
The same integration designs are shared between the paired baseline/polished fits.
Variation across independent scrambles gives a numerical integration standard error;
a nested half-size design gives an additional refinement comparison.

The `integration` table reports errors **after multiplying by $\sqrt n$**.
Increase `integration_power` and `integration_scrambles` if the integration flag fails,
or if a claimed small remainder is comparable to these numerical errors. These checks are
diagnostics, not deterministic integration-error bounds. Integration error shifts `E` and `T`
in opposite directions and cancels from their sum; it cannot create or erase `R_score`.

This DGP is unusually favorable: $\Psi_0(0,h)=0$ for **every** $h$.
Consequently, there is no nuisance-only population drift, even away from $h=0$.
A small `T` here therefore does not establish the same behavior for a general
heteroskedastic model. The population prediction discrepancy is evaluated as

$$
d^2(\hat\beta,\beta_0)=\sigma_V^2\hat\zeta^2+E_Z[\hat h(Z)^2].
$$
''')

md(r'''
## Paired optimization comparison

- **Baseline:** the original joint Adam protocol: 500 epochs, batch size 128, learning rate 0.001
  in the pilot/main profiles. Return the final iterate.
- **Polished:** continue from that fit using a fresh full-batch Adam optimizer and a learning
  rate decaying from $10^{-4}$ to $10^{-6}$. Retain the lowest **actual empirical check loss**
  among the baseline and all polishing iterates. Never select using the true parameter,
  remainder size, or projected score.
- **Frozen-$h$ adjustment:** solve
  $\hat a=\arg\min_a P_n\rho_{1/2}(\hat r-a\tilde X)$ as a diagnostic only.
  Record $\sqrt n\hat a$ and $n$ times the loss improvement. The original estimate is not replaced.

Paired confidence intervals describe the change in absolute diagnostic size on each dataset.
Smaller empirical loss need not imply smaller projected scores, remainders, or bias.
Polishing is a sensitivity check; it does not certify a global or frozen-$h$ minimum.

`WIDTH_MODE = "fixed"` isolates optimization sensitivity. `"growing"` uses
$\lceil16(n/500)^{1/4}\rceil$ units per layer for $n\ge500$ as an exploratory comparison.
This is not a claim that all network-class conditions of the theorem are met.
''')

code('''
if RUN_EXPERIMENT:
    RUN_DIR, results = diag.run_experiment(BASE, CONFIG)
else:
    results = diag.collect_results(RUN_DIR, CONFIG)
    print("Training disabled. Set RUN_EXPERIMENT = True and rerun this cell to start/resume.")
    print("Previously saved rows for this exact configuration:", len(results))

if not results.empty:
    progress = results.groupby(["n", "stage"]).size().rename("completed_reps").reset_index()
    progress["planned_reps"] = CONFIG["reps"]
    display(progress)
    expected_rows = 2 * len(CONFIG["n_values"]) * CONFIG["reps"]
    print("Complete:", len(results) == expected_rows)
''')

md(r'''
## Bias and distribution summaries

Intervals for means use between-replication uncertainty. Oracle Wald coverage uses the known
Gaussian-model variance as a calibration benchmark; it is not a feasible estimated-variance
procedure. A near-zero average score is insufficient: opposite signs can cancel.
Inspect absolute magnitudes, upper quantiles, and exceedance probabilities as well.
''')

code('''
tables = diag.summarize(results, CONFIG)
if tables:
    diag.save_summaries(RUN_DIR, tables)
    display(tables["bias"])
    for name in ["terms", "tails", "integration", "paired"]:
        print("\\n" + name.upper())
        display(tables[name])
else:
    print("No new results yet. The legacy analysis above is available immediately.")
''')

code('''
if tables:
    for path in diag.plot_diagnostics(tables, CONFIG, RUN_DIR):
        display(Image(filename=str(path)))
''')

md(r'''
## Interpretation checks, without automatic convergence verdicts

1. **Numerical accuracy first:** inspect integration uncertainty before interpreting `E` or `T`.
   An exactly reconstructed decomposition is necessary for implementation correctness, but is
   not a test of the asymptotic claim.
2. **Bias:** inspect the signed root-$n$ bias and its Monte Carlo interval. To resolve an effect
   of size $\eta$, the simulation standard error must be much smaller than $\eta$.
   Under the nominal root-$n$ SD $\sqrt{2\pi}$, a 95% interval half-width of 0.25 needs roughly
   $(1.96\sqrt{2\pi}/0.25)^2\approx387$ replications, not 30.
3. **Vanishing in probability:** for `S`, `E`, `T`, and the total residual, inspect
   $P(|A_n|>\varepsilon)$ for several fixed $\varepsilon>0$, with Wilson intervals.
   Zero observed exceedances in a small Monte Carlo sample is not a zero probability.
4. **Optimization versus statistical error:** if polishing reduces `S` and the total residual,
   optimization error is implicated. If ordinary scores improve but projected scores do not,
   inspect the missing projection direction and network approximation. These patterns identify
   candidates for further checks, not definitive causal explanations.
5. **Sampling noise should remain:** `W`, root-$n$ sampling SD, scaled MAE, and scaled RMSE
   generally stay nonzero. Their failure to vanish does not contradict asymptotic normality.
6. **Limits of the experiment:** a finite grid cannot prove convergence. The Gaussian regressor
   remains outside the main theorem's compact-support assumption; the fixed network does not
   reproduce its growing class. Even the growing-width option does not enforce all bounds,
   sparsity, or approximation-rate assumptions. This tests the numerical estimator in the
   stated toy DGP, not the theorem under all its conditions.
''')

code('''
if tables:
    integration = tables["integration"]
    print("Integration flags (rerun with more integration points if nonzero):",
          int(integration.inadequate_integrations.sum()))
    print("Maximum algebraic reconstruction error:", integration.max_decomposition_error.max())
    bias = tables["bias"].copy()
    tolerance = CONFIG["bias_tolerance"]
    bias["CI_inside_chosen_bias_band"] = (bias.ci_low > -tolerance) & (bias.ci_high < tolerance)
    print("Finite-sample bias assessment for tolerance", tolerance)
    display(bias[["n", "stage", "completed_reps", "root_n_bias", "ci_low", "ci_high",
                  "CI_inside_chosen_bias_band"]])
    print("No YES/NO conclusion about an asymptotic limit is inferred from these finite samples.")
''')

md(r'''
## Reproducibility and sources

Every new run is keyed by its full configuration, helper-code hash, and Python/package versions.
Each replication saves baseline and polished model checkpoints plus both diagnostic rows;
only complete pairs are skipped on restart. Errors stop the run rather than silently dropping
failed fits. Summaries always report the number of completed replications.
Changing the configuration or helper code creates a separate run directory.
Do not run two writers against the same run directory concurrently.

New outputs: `replication_results.csv`, `bias_summary.csv`, `terms_summary.csv`,
`tails_summary.csv`, `integration_summary.csv`, `paired_summary.csv`, three figures,
and per-replication `.pt` model checkpoints and `result.json` files.
The old `run/` folder is never used for new training or new summary writes.

Sources: [Zhong and Wang, main paper](../../Zhong%20and%20Wang%202024%20JBES.pdf),
Section 3 and Theorem 3.3; [supplement](../../Zhong%20and%20Wang%202024%20JBES%20supplement.pdf),
pp. 4–6. The Gaussian population-score formula is derived here from the stated DGP
and the explicitly displayed score normalization.
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': '.venv (Python 3.11)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
})
for i, cell in enumerate(nb.cells):
    cell.id = f'expansion-v2-{i:02d}'
nbf.validate(nb)
nbf.write(nb, base / 'psi_n_diagnostic.ipynb')
print('Wrote', len(cells), 'cells')
