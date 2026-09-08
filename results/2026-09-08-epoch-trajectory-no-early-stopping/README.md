# DPLQR continued-training diagnostic — 2026-09-08

**Question:** on the same simulated data and the same optimization path, does continued training move the coefficient estimate away from its truth while empirical quantile loss improves? This is a diagnostic of training duration, not hyperparameter tuning or a reproduction of the paper's Tables 1–3.

The intended design remains **100 repetitions per case, 1,000 epochs**. Completed here are a two-repetition smoke test, a two-repetition timing benchmark, and a **10-repetition-per-case pilot through all 1,000 epochs**. The pilot took 210.9 seconds with six CPU workers. The benchmark projects approximately 30–35 minutes for the intended 300 paths on this machine; that full experiment has **not** been run. These runs share some seeds and must not be pooled as independent repetitions.

Start with [the pilot report and figures](pilot-10/RESULTS.md), [raw trajectories](pilot-10/raw_epoch_trajectory.csv), and [coefficient summaries](pilot-10/monte_carlo_summary.csv). The pilot has 30 continuous paths and 450 checkpoint rows. Ten repetitions support exploratory interpretation, not a general robustness claim.

## What was reused and what changed

The existing DGP was inspected in `../2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb`. The runner loads only its existing `seed_all`, `nonlinear_truth`, `generate_covariates`, `generate_dataset`, and `tensor_pair` function definitions through Python's AST; it does not execute the replication notebook's run cells. It directly imports `dqNetSparse`, `checkLoss`, and `checkErrorMean` from the root `dqAux.py`. DPLQR is not reimplemented.

The original first-pass network fit uses validation early stopping with patience 15 and restores the best validation weights under a 100-epoch ceiling. That is appropriate for a selected fitted model, but would censor or overwrite the training path being studied here. This diagnostic removes that callback and supplies no validation data to `Model.fit`. A recording-only callback never stops training, selects a checkpoint, or restores weights. Each replicate constructs one network and one optimizer and calls `fit` exactly once for the requested ceiling. The optimizer state, architecture and learning rate remain continuous across checkpoints.

The original first-pass code also clips nonlinear weights **after** fitting. Inspection of the earlier epoch study found that its checkpoint evaluator clipped an evaluation clone while recording the live, unclipped coefficient estimate. Consequently, its displayed losses and nuisance errors did not describe the actual network being optimized. This diagnostic evaluates the current network directly, **without clipping or nuisance recalibration**. This is a correction to the diagnostic measurement; it does not add or change training regularization. Earlier prediction-loss plots should not be interpreted as losses along this clean optimization path.

No original simulation notebook, R script, root helper, or earlier result was edited. `preserved_inputs.json` records SHA-256 fingerprints of 151 pre-existing files. The deleted `simulate_homoscedastic.py` is not imported or required. R is unnecessary for this DPLQR-only diagnostic. This new experiment provides a notebook launcher and separate scripts for the requested command-line replication control.

## Exact configuration

| Setting | Value |
| --- | --- |
| Cases | 1: linear; 2: additive; 3: deep, using the existing `nonlinear_truth` unchanged |
| Model | `Y = X theta_0 + m_0(Z) + epsilon`, `theta_0 = (1, -1)`, independent Student-t errors with 3 degrees of freedom |
| Covariates | Existing 10-dimensional Gaussian copula with correlation 0.5; uniform margins on [0, 2]; eight nonlinear covariates, one binary linear covariate, one continuous linear covariate |
| Sample and quantile | `n = 1000`, `tau = 0.5`; original 800/200 estimation/holdout split |
| Preprocessing | Standardize Z using the 800 estimation observations only; leave X on its original scale |
| Independent test data | 10,000 fresh (X, Z, Y) observations per replicate, fixed across checkpoints |
| Independent evaluation data | A separate 10,000 fresh (X, Z) observations per replicate, with known truth; fixed across checkpoints |
| Network | Original `dqNetSparse(2, 8, ..., [2, 32], sparseRatio=0.5)`; ReLU, original dropout probability `1e-5` |
| Depth convention | `[2, 32]` means an initial hidden layer followed by two masked hidden layers, all width 32 |
| Coefficients | `net.linLinear.weight`, no linear intercept; the nonlinear branch contains its original biases |
| Initialization | Original network initialization and masks, followed by `linLinear.reset_parameters()` as in the existing fit |
| Optimizer | Existing torchtuples `AdamW` default wrapper over `torch.optim.Adam`; betas `(0.9, 0.99)`, epsilon `1e-8`, zero decoupled weight decay |
| Learning rate and batches | Constant `0.005`; shuffled minibatches of 128, seven optimizer updates per epoch |
| Stopping and selection | Fixed 1,000 epochs; no early stopping, restoration, validation selection, scheduler, or added regularization |
| Computation | CPU; deterministic Torch algorithms; one Torch thread per worker; six independent workers in the pilot |

Checkpoints are exactly:

```text
1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000
```

`n` denotes the original total simulation sample size. Training loss averages over the **800 observations actually used for estimation**, preserving the first-pass split. The 200 holdout observations only supply an optional diagnostic loss. Test/evaluation observations never enter fitting or standardization. For a nonmedian `--tau`, the known nuisance truth includes the Student-t quantile shift; it is zero at the default median.

## Reproducibility

For case `c` and one-based repetition `r`, with base seed `20260904`:

```text
data_seed     = base_seed + c * 10_000_000 + n * 1000 + r
init_seed     = data_seed + round(tau * 1_000_000)
training_seed = init_seed + 1_000_000_000
test_seed     = data_seed + 2_000_000_000
eval_seed     = data_seed + 3_000_000_000
```

The original data RNG also creates the estimation/holdout permutation. Available first-pass training exports are checked against the regenerated data. Network initialization and sparsity masks use `init_seed`; subsequent minibatch shuffling and original dropout use `training_seed`. Test and evaluation draws use separate NumPy generators. Workers own independent data, networks and optimizers, so their completion order does not define any seed.

Removing the old validation DataLoader also removes its consumption of Torch's global RNG. The explicit separate training seed makes the new experiment reproducible, but the trajectory is not expected to reproduce an older validation-enabled path byte for byte. This is not a substantive change to the DGP or optimizer.

At every checkpoint, evaluation uses inference mode and temporarily disables dropout. It then restores all training flags and asserts that neither parameters nor Python, NumPy or Torch RNG states have changed. Network/optimizer identities and cumulative optimizer steps are checked throughout fitting. Each checkpoint archive contains model and optimizer state, preprocessing parameters, seeds and Torch RNG state; the completion manifest links checkpoint hashes and verifies that the final weights are the 1,000-epoch weights.

Each run's `run_config.json` records its actual design, software versions, source fingerprints, counts, status and elapsed time. A completed replicate can be reused only under the same run identity. An interrupted replicate restarts from its seeds; partial trajectories are not spliced together. Changing repetitions or scientific settings requires a fresh output directory, preserving earlier runs. Exact floating-point reproduction across different library versions or hardware is not guaranteed.

## Running from PowerShell

Run these commands from the repository root using the existing `.venv`. No package downgrade or new dependency is required in the supplied environment.

Smoke test, including a check that observation frequency does not affect training:

```powershell
.\.venv\Scripts\python.exe results/2026-09-08-epoch-trajectory-no-early-stopping/scripts/run_trajectory.py --repetitions 2 --max-epochs 50 --workers 3 --verify-observer --output-dir results/2026-09-08-epoch-trajectory-no-early-stopping/smoke-test
```

The smoke test retains checkpoints up to epoch 50. Its uncertainty intervals are software checks, not useful statistical evidence.

Reproduce or resume the completed pilot:

```powershell
.\.venv\Scripts\python.exe results/2026-09-08-epoch-trajectory-no-early-stopping/scripts/run_trajectory.py --repetitions 10 --max-epochs 1000 --workers 6 --output-dir results/2026-09-08-epoch-trajectory-no-early-stopping/pilot-10
```

Intended full experiment, **100 repetitions per case**:

```powershell
.\.venv\Scripts\python.exe results/2026-09-08-epoch-trajectory-no-early-stopping/scripts/run_trajectory.py --repetitions 100 --max-epochs 1000 --workers 6 --output-dir results/2026-09-08-epoch-trajectory-no-early-stopping/full-100
```

The CLI also exposes `--n`, `--tau`, `--cases`, `--seed`, `--test-size`, `--eval-size`, `--threads`, and `--checkpoints`. Defaults retain the intended 100-repetition design. Workers affect parallel execution only; use fewer workers if necessary.

Verify a completed run independently:

```powershell
.\.venv\Scripts\python.exe results/2026-09-08-epoch-trajectory-no-early-stopping/scripts/verify_outputs.py results/2026-09-08-epoch-trajectory-no-early-stopping/pilot-10
```

For Jupyter, open [diagnostic.ipynb](diagnostic.ipynb), select the repository `.venv` kernel, and run cells in order. Its editable settings default to the two-repetition smoke test. To view/reuse the pilot, set `REPETITIONS = 10`, `MAX_EPOCHS = 1000`, `WORKERS = 6`, and `OUTPUT_DIR = EXPERIMENT_DIR/'pilot-10'`. For the full design use 100 and a new `full-100` output directory. No source file is generated or deleted by the notebook.

## Output definitions and interpretation

Each run directory contains:

| File or folder | Contents |
| --- | --- |
| `run_config.json` | Actual run identity, configuration, execution status and counts |
| `raw_epoch_trajectory.csv` | One row per case × repetition × checkpoint, including all five seeds |
| `monte_carlo_summary.csv` | Case × epoch × coefficient: mean estimate, signed bias, sample SD, RMSE, MAE and bias MCSE |
| `diagnostic_summary.csv` | Tidy means, sample SDs and MCSEs for vector coefficient error, losses, L2 errors and cancellation diagnostics |
| `paired_endpoint_changes.csv` | Within-replicate changes from epoch 100 to 1000 when available; otherwise the first and last recorded epochs |
| `endpoint_pattern_counts.csv` | Counts of directional changes, including train loss down / coefficient error up / quantile error down |
| `figures/` | Plots A–F, each as PNG and PDF |
| `replicates/` | Checkpoint states, raw path CSVs, data fingerprints and completion audits |
| `replicate_timings.csv` | Per-path timing and whether a completed result was reused |
| `verification.json` | Independent verification of completed outputs and reconstructed checkpoints |
| `RESULTS.md` | Computed results, paired endpoint tables and plots for the actual run |

`theta_hat_1/2`, `theta_true_1/2`, `theta_error_1/2` and `abs_theta_error_1/2` describe each coefficient. `theta_error_l2` is its vector Euclidean error. Losses are **full-sample empirical check losses of the current checkpoint**, not a minibatch training-log average. `m_l2_error` and `q_l2_error` are square roots of absolute empirical mean squared errors on the independent evaluation sample. They are not the relative-MSE metric in the older Table 3 replication.

The decomposition uses `linear_error = X(theta_hat - theta_0)` and `nonlinear_error = m_hat - m_0`:

```text
q_mse = linear_error_mse + nonlinear_error_mse + 2 * cross_error_moment
cancellation_fraction = -2 * cross_error_moment / (linear_error_mse + nonlinear_error_mse)
```

A negative cross moment means the two errors partially cancel on the evaluation draw. Positive cancellation fraction reports the fraction of their summed squared errors removed by the cross term (defined as zero if both errors vanish). This is an uncentered cross moment, not a correlation. A negative value alone does not show that coefficient deterioration is being hidden by improved prediction.

- **A:** mean signed coefficient error; zero means no bias. Small average bias can hide dispersion, so read B too.
- **B:** component RMSE and mean vector coefficient error. Examine the entire path, rather than choosing the most favorable checkpoint.
- **C:** training and independent test check loss. Training improvement with test deterioration is compatible with prediction overfitting.
- **D:** nuisance and full conditional-quantile L2 errors. Coefficient error increasing while q error stays stable or improves would motivate a compensation interpretation; deterioration in both is a different pattern.
- **E:** squared-error decomposition and cross term, directly showing cancellation.
- **F:** raw coefficient trajectories for repetitions 1–3, selected in advance as illustrations; Monte Carlo summaries remain the primary evidence.

The replicate is the Monte Carlo unit. SD uses `ddof=1`, RMSE uses mean squared error against truth, and shaded mean bands are pointwise normal approximations `mean ± 1.96 × MCSE`. They are not simultaneous intervals, and with ten repetitions they should be treated cautiously. The paired endpoint comparison is a fixed descriptive comparison, not a selected optimum. A lack of coefficient drift is a valid outcome.

## What the 10-repetition pilot shows

The dominant late-training pattern is **prediction overfitting**, with limited and inconsistent coefficient drift. From epoch 100 to 1000, all 30 paths lower training loss but increase independent test loss and full-quantile L2 error. Mean vector coefficient error improves in all three cases: 0.311 to 0.167 (linear), 0.332 to 0.201 (additive), and 0.294 to 0.186 (deep).

The entire trajectory gives a more nuanced picture than this endpoint comparison. From epoch 500 to 1000:

| Case | Mean coefficient L2 at 500 | At 1000 | Paired change | MCSE of change | Paths with increased error |
| --- | --- | --- | --- | --- | --- |
| 1: linear | 0.1462 | 0.1669 | +0.0208 | 0.0146 | 8/10 |
| 2: additive | 0.1940 | 0.2005 | +0.0065 | 0.0181 | 6/10 |
| 3: deep | 0.2141 | 0.1864 | -0.0276 | 0.0188 | 3/10 |

Case 1 also has component-specific drift: theta_1 RMSE rises from 0.0488 at epoch 200 to 0.0959 at 1000. However, full-quantile error worsens at the same time; it increases in all 30 paths between 500 and 1000. Thus this is not a sustained example of worse coefficients hidden by preserved prediction accuracy. Negative cross terms show some cancellation, but mean cancellation fractions fall to approximately 2–4% at epoch 1000, insufficient to maintain q accuracy.

These later comparisons describe the observed path; they are not pre-specified hypothesis tests. With only ten repetitions, small coefficient changes remain uncertain. There is no basis here for a blanket claim that estimates are robust to arbitrary training duration, or that prolonged training necessarily causes coefficient drift.

## Verification

The smoke test completed all six 50-epoch paths (42 finite checkpoint rows), with 350 optimizer steps per path. A separate observation-frequency check produced exactly identical final weights and diagnostics after 12 epochs when recording every epoch versus recording only the last epoch. Independent verification reloads saved checkpoints and reconstructs their data to check coefficient extraction, losses, nuisance/quantile errors, decomposition and Monte Carlo formulas.

The pilot passed verification of all 450 finite rows and checkpoint archives, 30 complete 1,000-epoch paths with 7,000 optimizer updates each, and all coefficient/diagnostic summaries. Nine independently reconstructed checkpoints reproduced their saved metrics. The maximum squared-error decomposition discrepancy was `7.11e-15`. See [pilot verification](pilot-10/verification.json). All 151 preserved original-file fingerprints still match.

The Jupyter launcher was also executed from start to finish in a fresh repository `.venv` kernel, successfully reusing the smoke-test paths, verifying them, and displaying the tables and all six figures. Its executed copy is [smoke-test/executed_diagnostic.ipynb](smoke-test/executed_diagnostic.ipynb).

The new implementation consists of `scripts/run_trajectory.py` (training and checkpoints), `scripts/reporting.py` (summaries and figures), `scripts/verify_outputs.py` (independent numerical audit), and `diagnostic.ipynb` (Jupyter launcher). All additions and generated outputs are confined to this experiment directory.
