# Validation early stopping versus coefficient estimation — 2026-09-08

**Question:** does the epoch selected by validation quantile loss also give a good estimate of the interpretable coefficient vector theta? This experiment compares stopping rules on the same optimization path and contrasts their selected epochs with infeasible simulation oracles. Truth and test data never enter the stopping decision.

Start with [RESULTS.md](RESULTS.md), [patience summaries](monte_carlo_summary_patience.csv), [maximum-epoch summaries](monte_carlo_summary_max_epochs.csv), and [oracle summaries](oracle_epoch_summary.csv). Results use every available path from Experiment 1's `pilot-10`: **10 repetitions in each of three cases**, not 100 repetitions per case. This is an estimation diagnostic; it does not compute confidence-interval coverage or establish validity of inference.

The completed analysis contains **30,000 per-epoch rows and 390 stopping-rule rows**, with five PNG/PDF figures per case. The two-path smoke run took 46.8 seconds; the subsequent all-path run, reusing those two paths, took 388.0 seconds.

With patience 15 and cap 1000, validation selects earlier than the theta oracle in **all 30 paths**, and none is within 25 epochs of that oracle:

| Case | Mean selected epoch | Mean theta-oracle epoch | Mean theta regret |
| --- | --- | --- | --- |
| 1: linear | 54.5 | 353.3 | 0.563 |
| 2: additive | 65.9 | 583.4 | 0.430 |
| 3: deep | 41.3 | 605.8 | 0.564 |

At patience 15, caps **200, 500 and 1000 return exactly the same selected fit in all 30 paths**. Cap 100 already matches in 29 of 30 paths. Mean coefficient error improves with continued training to epoch 1000, while mean quantile-prediction error and test loss worsen. The observed mismatch is therefore early prediction-based selection versus later coefficient improvement; it is not systematic selection after the theta oracle. These are pilot results, and the infeasible oracle is optimistic by construction.

## Why dense recording was necessary

Experiment 1 saved validation loss and model quantities at only 15 epochs. Those sparse checkpoints cannot determine whether patience expired between checkpoints, or locate an oracle over every epoch from 1 through 1000. No interpolation or sparse-grid approximation is used here.

An optional observer extension, `../2026-09-08-epoch-trajectory-no-early-stopping/scripts/record_dense.py`, records every epoch while retaining the original weight archives at the original 15 checkpoints. It reuses the original `fit_one` function's exact code with a private callback binding, the original DGP, model, optimizer, initialization and RNG scheme. The original runner file and its source fingerprints are unchanged.

Each existing case–replicate path is **deterministically replayed once** to obtain the missing observations, then every stopping rule is applied retrospectively to that single dense path. This is not derived entirely from the old CSVs, and models are not refitted for different patience values or epoch caps. The replay must match every original checkpoint metric exactly, match the model-state fingerprints at all 15 original epochs, and match the initial and final state fingerprints. A mismatch aborts the analysis rather than treating a new path as equivalent.

The first two paths were recorded and checked before processing the remaining paths; they are reused in the final analysis. The experiment does not create additional independent Monte Carlo repetitions by replaying old seeds.

## Source files and preserved design

The source run is `../2026-09-08-epoch-trajectory-no-early-stopping/pilot-10/`. The following files are used directly:

- `run_config.json`: original configuration, numerical environment and source identities.
- `raw_epoch_trajectory.csv`: identifies all available case–replicate paths.
- `replicates/case_C_rep_RRRR/trajectory.csv`, `completion.json`, `data_manifest.json`, and `epoch_EEEE.pt`: checkpoint, state, data and reproducibility comparisons.
- Experiment 1's `scripts/run_trajectory.py`: exact fitting code, original metric calculations, preprocessing, seed scheme, and AST-loaded DGP functions.
- `../2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb`: original DGP definitions and the authoritative `EarlyStopInMemory` callback used for validation tests.
- Root `dqAux.py`: original `dqNetSparse`, `checkLoss`, and `checkErrorMean`.

The dense-input manifest is [dense_identity.json](dense_identity.json); the actual execution and analysis settings are in [run_config.json](run_config.json). Each dense path has its own data and completion manifest.

| Setting | Preserved value |
| --- | --- |
| Cases | 1 linear, 2 additive, 3 deep; original homoscedastic DGP with independent Student-t(3) errors |
| True coefficients | `(1, -1)`; linear branch has no intercept |
| n and tau | `1000`, `0.5` |
| Estimation / validation | Original fixed `800 / 200` split within each replicate |
| Test / evaluation | Separate independent samples of 10,000 observations each, fixed within a replicate and identical to Experiment 1 |
| Network | Original sparse DPLQR `[depth=2, width=32]`, sparse ratio 0.5, ReLU, original dropout `1e-5` |
| Scaling | Train-only standardization of Z; original scale of X |
| Optimization | Original torchtuples AdamW wrapper, learning rate `0.005`, batch size `128`, no added regularization or scheduler |
| Trajectory | One fit through epoch 1000; diagnostics at every integer epoch |
| Main analysis 2A | Patience `5, 15, 30, 50, 100, 200`; common cap 1000 |
| Main analysis 2B | Patience `15`; caps `25, 50, 100, 200, 500, 1000` |
| Reference | No early stopping, selected epoch 1000, no restoration |

The original seeds are reused exactly. For case `c`, one-based repetition `r`, and base seed `20260904`:

```text
data_seed     = base_seed + c * 10_000_000 + n * 1000 + r
init_seed     = data_seed + round(tau * 1_000_000)
training_seed = init_seed + 1_000_000_000
test_seed     = data_seed + 2_000_000_000
eval_seed     = data_seed + 3_000_000_000
```

Workers own separate models and RNG streams. More frequent observations do not consume the training RNG or change live weights. No validation DataLoader is inserted into training, because creating it would consume Torch RNG and change the Experiment 1 path. Instead, its scoring calculation is reproduced directly in inference mode after each epoch.

## Exact practical early-stopping convention

The authoritative callback is the main homoscedastic notebook's **custom `EarlyStopInMemory`**, with:

1. One-based epochs and initial best loss of positive infinity.
2. Strict improvement: `validation_loss < best_loss`, equivalent to `min_delta = 0`.
3. An improvement resets the non-improvement counter to zero and saves that epoch.
4. A tie counts as non-improvement and retains the **earliest** best checkpoint.
5. Training stops when consecutive non-improving epochs reach patience.
6. Best weights are restored, including when the maximum epoch cap is reached.

`stop_epoch` is when the patience rule or cap terminates the fit. `selected_epoch` is the best validation epoch encountered through `stop_epoch`, whose weights would be returned. If patience triggers exactly at the cap, `stopping_reason` is `early_stopping`, because the callback actually triggers then.

There are two important implementation details:

**Validation batching.** The current fit specifies `val_batch_size=128`. Installed torchtuples averages the two validation batch means equally: `(loss_for_128 + loss_for_72) / 2`, in float32. This differs from the ordinary empirical mean over all 200 observations. For example, batch losses 0.5 and 2.0 produce 1.25 under the actual implementation, versus the sample-weighted mean 1.04. This experiment faithfully uses the existing batching convention for selection and also retains the ordinary empirical validation loss for transparency. No silent correction changes the rule being studied.

**Which callback is replicated.** Installed torchtuples' built-in `EarlyStopping` can overwrite its saved state on an exact tie, returning the latest tied best state. The notebook's custom callback returns the earliest tied best state. The latter is replicated here; these behaviors should not be conflated.

The no-stopping reference has `patience=0` as a CSV sentinel, `rule_id=no_early_stopping`, and `restore_best=False`. Zero is not a valid patience supplied to the practical rule. Its best validation epoch is still recorded diagnostically, but its selected epoch is always 1000.

The original main replication applies nonlinear weight clipping after fitting. Experiment 2 preserves Experiment 1's **unclipped** trajectory and reads quantities directly from the selected epoch. It isolates stopping and restoration; it does not add post-fit clipping or reproduce every step of the separate main replication pipeline.

## Oracles and measurements

Oracles search **all 1000 integer epochs** separately within each replicate, choosing the earliest exact minimum:

- `e_theta`: minimum vector coefficient error `||theta_hat - theta_0||_2`.
- `e_q`: minimum full conditional-quantile L2 error.
- `e_m`: minimum nuisance-function L2 error.
- `e_test`: minimum independent test check loss.

Component-specific coefficient oracles and the full-path validation minimum are also recorded. Test, nuisance and quantile oracles use the same fixed samples as Experiment 1. These are infeasible, sample-dependent benchmarks; looking across 1000 noisy observations makes their reported minima optimistic. They are not implementable tuning recommendations, and the theta oracle may be later than a finite cap permits.

`theta_regret` is selected vector error minus its full-path minimum. `theta_squared_regret` is the analogous difference of squared vector errors; `q_regret` compares full-quantile L2 error to its oracle. These quantities and signed/absolute epoch gaps are computed only **after** the validation-only selector returns its decision. The selector accepts just validation losses, patience and a cap: it has no access to truth, test losses or oracle columns.

The selected-row CSV records coefficient estimates, truths, signed and absolute errors, vector error, losses, nuisance/quantile errors, seeds, state fingerprints, stopping decisions, oracle epochs and regrets. `training_check_loss` is the full empirical mean on the 800 estimation observations. In `early_stopping_results.csv`, `validation_check_loss` and `stopping_validation_check_loss` refer to the practical batch-weighted stopping criterion; `empirical_validation_check_loss` is the ordinary empirical mean. In the dense trajectory CSV, the original Experiment 1 column `validation_check_loss` retains its empirical meaning, and the stopping criterion has its separate explicit name.

All L2 errors mean **square root of absolute mean squared error**, not the relative-MSE metric from the paper's Table 3 replication. Summaries use the replicate as the Monte Carlo unit: sample SD (`ddof=1`), coefficient RMSE against truth, and paired changes for comparisons of rules. The six patience rows and six cap rows are alternative decisions on the same replicate, not additional independent observations.

## Running and reproducing

From the repository root, in PowerShell with the existing `.venv`:

```powershell
# Record and verify the first two existing paths before the full analysis.
.\.venv\Scripts\python.exe results/2026-09-08-early-stopping-vs-theta/scripts/run_diagnostic.py --stage smoke --workers 2

# Compare retrospective decisions to the actual notebook callback on those paths.
.\.venv\Scripts\python.exe results/2026-09-08-early-stopping-vs-theta/scripts/validate_stopping_rules.py --trajectory-csv results/2026-09-08-early-stopping-vs-theta/smoke-test/dense_epoch_trajectory.csv --output results/2026-09-08-early-stopping-vs-theta/validation/stopping_rules_dense_smoke.json

# Process ALL available paths from the source run; reuse completed dense replays.
.\.venv\Scripts\python.exe results/2026-09-08-early-stopping-vs-theta/scripts/run_diagnostic.py --stage all --workers 6

# Rebuild tables and figures from complete dense paths, without training.
.\.venv\Scripts\python.exe results/2026-09-08-early-stopping-vs-theta/scripts/run_diagnostic.py --stage analyze

# Independently audit dense data, stopping decisions, oracles and summaries.
.\.venv\Scripts\python.exe results/2026-09-08-early-stopping-vs-theta/scripts/verify_outputs.py
```

`--source` defaults to Experiment 1's `pilot-10`. After producing a 100-repetition Experiment 1 run, provide that directory with `--source` and choose a fresh `--output-dir`; all repetitions in that source are processed. There is no silent reduction to ten. Input fingerprints prevent mixing trajectories from different configurations or code. An interrupted incomplete replay restarts from its fixed seeds; completed verified paths are reused. Existing outputs are preserved when changing the source by using a new output directory.

For Jupyter, open [diagnostic.ipynb](diagnostic.ipynb) with the repository `.venv` kernel and run cells in order. It displays completed summaries and figures without training. Change `CASE_TO_DISPLAY` to 1, 2 or 3. `RUN_ANALYSIS=True` regenerates reports from existing dense trajectories; it does not fit networks.

## Output map

| File or directory | Purpose |
| --- | --- |
| `run_config.json`, `dense_identity.json` | Actual source, settings, source fingerprints and completion status |
| `dense_epoch_trajectory.csv` | Every epoch on each continuous path |
| `early_stopping_results.csv` | One row per case × repetition × stopping-rule configuration |
| `monte_carlo_summary_patience.csv` | Patience comparisons, coefficient bias/SD/RMSE, epoch distributions, stopping proportions and oracle regret |
| `monte_carlo_summary_max_epochs.csv` | Corresponding epoch-cap comparisons |
| `oracle_epochs.csv`, `oracle_epoch_summary.csv` | Replicate-level oracles and their Monte Carlo summaries |
| `paired_max_epoch_comparisons.csv` | Paired comparisons showing when larger caps return exactly the same coefficients |
| `figures/` | Figures A–E, separately for each case, as PNG and PDF |
| `dense-trajectories/` | Per-path epoch metrics, original-frequency weight archives, data fingerprints and completion checks |
| `smoke-test/`, `validation/`, `pairing_validation.json` | Small-run evidence, callback checks and paired-selection verification |
| `RESULTS.md` | Data-derived interpretation, figures and numerical comparisons |

## Reading the figures

**A** compares coefficient RMSE across patience settings. **B** shows selected epochs across patience settings, with between-replicate variation. **C** compares epoch caps with actual selected epochs: a flat segment means higher caps have stopped changing the selected fit. **D** compares prediction error with coefficient RMSE for the same stopping rules. **E** compares the theta oracle (horizontal axis) with validation-selected epochs (vertical axis), with the diagonal indicating equality. Points below the diagonal stop earlier than the theta oracle; points above stop later.

Three distinct patterns motivate the diagnostic:

1. **Ordinary prediction overfitting:** later training lowers training loss but raises validation/test loss and coefficient error.
2. **Prediction–estimation tension:** prediction remains stable or improves while coefficient error increases. Prediction-based selection may then miss deterioration in the coefficient estimate.
3. **Protection of both:** the validation-selected epoch is close to the theta oracle, and both errors deteriorate afterward.

These are possibilities, not required findings. The opposite timing mismatch is also possible: validation stopping may prefer an early model whose prediction is good while coefficients need longer to approach truth. Epoch gaps alone are insufficient; assess regret and prediction error as well. Reported closeness within 10 or 25 epochs is descriptive, and exact equality should not be overinterpreted. No pattern establishes inference validity without a separate coverage study.

## Files added and validation

This experiment adds `scripts/run_diagnostic.py`, `scripts/stopping_rules.py`, `scripts/validate_stopping_rules.py`, `scripts/reporting.py`, `scripts/verify_outputs.py`, `diagnostic.ipynb`, this README, and the generated outputs. Experiment 1 gains only the optional `scripts/record_dense.py` observer module. Existing experiment sources and outputs are preserved.

The stopping-rule smoke checks compare against the actual notebook class, including ties, tiny strict improvements, patience one, counter resets, cap truncations, and a patience trigger coinciding with the cap. A frozen-network test verifies the installed library's validation-batch calculation. The first two complete dense paths also passed callback comparisons for every requested rule. Source checkpoint matching, selected-state equality and cap-plateau checks must pass before reporting results.

The full independent audit passed: 30 continuous paths, 30,000 finite epoch rows, exact matches to all 450 original checkpoint rows and 450 weight-state fingerprints, and identical data manifests. It independently recomputed all 390 stopping decisions, every full-path oracle and regret, 194 cap-plateau comparisons, and all coefficient/epoch/oracle summaries. Nine reloaded checkpoints reproduced their metrics, including exact agreement with the installed torchtuples validation scorer. See [verification.json](verification.json).

The notebook executed successfully in a fresh `.venv` kernel, with no additional training; its executed copy is [validation/executed_diagnostic.ipynb](validation/executed_diagnostic.ipynb). The pre-existing 151-file preservation check also passed in [validation/preservation.json](validation/preservation.json).
