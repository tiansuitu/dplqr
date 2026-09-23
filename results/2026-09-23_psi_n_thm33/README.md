# Score, expansion, and root-n bias diagnostics

Open `psi_n_diagnostic.ipynb`. The notebook separates the fitted-score condition,
empirical-process increment, population Taylor remainder, total linear-expansion
residual, and root-n-scaled **signed** bias.

The previous executed notebook is saved as
`psi_n_diagnostic_before_expansion_v2.ipynb`. Existing `run/` files are preserved.
The rewritten notebook analyzes their signed bias and reconstructs the total
expansion residual from saved data-generation seeds without retraining. It cannot
recover separate empirical-process/Taylor terms without the old fitted neural functions.

## Run controls

`RUN_EXPERIMENT = False` by default: Run All displays the legacy analysis and
any compatible saved new results without starting training.

Set `RUN_EXPERIMENT = True` to run or resume the selected profile:

| Profile | Sample sizes | Replications per size | Baseline / polishing epochs |
|---|---|---:|---:|
| smoke | 32, 48 | 2 | 2 / 3 |
| pilot | 500, 1000, 2000 | 10 | 500 / 500 |
| main | 500, 1000, 2000, 4000 | 100 | 500 / 1000 |

The pilot is not large enough to resolve small biases. Replication counts are
editable; a nominal 95% bias interval half-width of 0.25 on the root-n scale
requires roughly 387 independent replications under the Gaussian benchmark.

`WIDTH_MODE = "fixed"` retains `[16, 16]` for a controlled optimizer comparison.
`"growing"` provides an exploratory increasing-width alternative. Neither mode
is claimed to enforce all theorem assumptions.

## Diagnostics

- `S`: fitted projected score times sqrt(n).
- `E`: empirical-process increment times sqrt(n).
- `T`: population Taylor remainder times sqrt(n).
- `R_score = J*sqrt(n)*(theta_hat-theta0) + W = S-E-T`.
- `R_theta = R_score/J`: total expansion residual in parameter units.
- `W`: true oracle score times sqrt(n); this should remain nondegenerate.
- Root-n signed bias with Monte Carlo standard errors and Student-t intervals.
- Root-n sampling SD, MAE, RMSE, and oracle-variance Wald coverage as calibration
  measures, not vanishing targets.
- Scaled nuisance/prediction errors, ordinary versus projected scores, frozen-h
  adjustments, objective gains, and residual ties.
- Paired baseline-versus-polished comparisons on the same data and initial fit.

The population-score calculation analytically integrates the independent Gaussian
error and residualized regressor. Independent scrambled Sobol designs integrate
only over two-dimensional Z. Replicate-scramble errors and a half-size refinement
comparison are reported after sqrt(n) scaling. Increase integration settings when
these are material relative to a claimed small term.

The exact decomposition is an algebra check, not evidence of convergence. The
total residual does not depend on numerical population integration. This toy DGP
has exact cancellation of nuisance-only population drift, making the population
remainder check especially favorable. See the notebook for the derivation, the
score-normalization issue in the supplement, and theorem-scope limitations.

## Outputs and restart safety

New results are written to `run_expansion_v2/<profile>_<hash>/`. The hash covers
configuration, helper code, and runtime versions. Changing settings creates a
separate run; incompatible results are not appended to old CSVs.

Each dataset saves two model checkpoints and two diagnostic rows. Completion is
marked only after both stages are saved. Failures stop execution rather than
silently removing replications. Restart skips complete pairs. Do not launch
concurrent writers for the same directory.

Outputs include `replication_results.csv`, five summary CSVs (bias, terms, tails,
integration, paired), three diagnostic figures, a manifest, and per-replication
checkpoints. The old `run/` folder is never modified by this code.

## Validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe results/2026-09-23_psi_n_thm33/test_diagnostics.py
```

Tests check the analytic score against independent numerical integration and its
derivative, LP sign/objective, baseline equivalence, signed-bias summaries,
end-to-end smoke execution, model reloads, paired optimization, summaries/plots,
restart integrity, configuration isolation, and preservation of old results.
These are implementation checks, not a production simulation or a theorem proof.
