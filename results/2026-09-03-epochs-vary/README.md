# Concrete DPLQR epoch sensitivity — 03 September 2026 (03092026)

`epochs_vary.py` adapts `../../demo.ipynb` to compare epoch ceilings of
50, 100, 250, 500 (the demo setting), and 1,000. Both the main DPLQR fit and
the auxiliary standard-error projection receive the varied ceiling, matching
the demo's use of one `epochs` variable.

## Controlled setup

- Same concrete data, 49 quantiles (0.02–0.98), `nodes=[5,128]`, 50% sparsity,
  batch size 128, learning rate 0.001, and default torchtuples AdamW optimizer.
- One fixed split (`20260903`): 742 training, 82 validation, and 206 test rows.
  Scaling is fitted only on training rows. Saved split CSVs include the
  zero-based source row index.
- Three training seeds (11, 22, 33), with paired initialization, masks, and
  batch order across ceilings. Separate deterministic seed streams are used
  for each quantile and the two networks; the formula is in `run_config.json`.
- The demo's validation early stopping remains active (patience 10), with
  best validation weights restored. Thus the varied setting is a **ceiling**,
  not a requirement to complete that many epochs.
- The continuous auxiliary MSE projection (`logic=False`) from the August
  compatibility correction is retained. LQR is fitted once on the same
  824-row training/validation pool, as in the demo, to initialize DPLQR and
  provide an epoch-independent reference.

To avoid repeating identical training prefixes, each seed/quantile/network
is trained once up to the largest ceiling or early stopping. Best validation
weights available at each ceiling are saved in memory. All evaluations take
place after training; test data never select a checkpoint. `--verify` checks
every ceiling at the median quantile for the first seed against independently
trained ordinary `EarlyStopping()` fits, and checks the reported standard
errors against `dqAux.getSESingle`. During verification, ordinary PyTorch
checkpoint serialization uses RAM buffers to avoid a Windows file-lock error;
the stopping and weight-selection logic is unchanged.

## Run and outputs

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe results/2026-09-03-epochs-vary/epochs_vary.py --verify
```

The script writes only to its own folder by default. Use `--output-dir PATH`
for a separate rerun. `--epochs`, `--seeds`, and `--split-seed` are configurable;
include 500 among the ceilings for the paired baseline comparison.

- `summary.md` / `summary.csv`: table of means and sample SDs across seeds.
- `epochs_comparison.png` / `.pdf`: loss, CI area, coefficient curves, and
  actual training lengths.
- `metrics_by_seed.csv`: integrated metrics for each seed and ceiling.
- `results_by_quantile.csv`: coefficients, standard errors, test check loss,
  completed epochs, and selected best epochs for all 735 combinations.
- `training_history.csv`: training and validation losses by completed epoch.
- `df_train.csv`, `df_val.csv`, `df_test.csv`: the fixed split.
- `lqr_reference.csv`, `lqr_warnings.json`: same-split LQR results and warnings.
- `run_config.json`: settings, source hashes, package versions, and run status.
- `verification.json`: independent-fit and standard-error verification.

ACL and CI area use the demo's Riemann sums: `0.02 * sum(check_loss)` and
`0.02 * sum(upperCI - lowerCI)` over 49 quantiles. ACL is therefore not
renormalized to an arithmetic mean. Percentage changes are computed within
each seed relative to 500, then averaged.

This assesses sensitivity to the training ceiling on one real-data split.
The SD describes training randomness, not sampling uncertainty or empirical
confidence-interval coverage. A smaller interval area alone does not establish
better inference. Varying the auxiliary ceiling also affects CI area. The
experiment retains early stopping; it does not measure what would happen if
training were forced to continue past the stopping point.

## Results

The mean test ACL was 0.076169 at 50 epochs, 0.068076 at 100, and 0.066808
at 250, 500, and 1,000. Relative to the 500 ceiling, this is 14.02% worse at
50 and 1.90% worse at 100. All main fits had stopped by epoch 220 and all
projection fits by epoch 113, so 250, 500, and 1,000 select exactly the same
weights and give identical coefficients, standard errors, and test losses in
this controlled run. The practical conclusion is that the demo's 500 ceiling
is safely above the observed stopping point; 250 was already sufficient here,
while 50 was too restrictive and 100 retained a small loss penalty.

Mean CI area fell from 0.384422 at 50 to 0.370851 at 250 and above, but its
across-seed SD was comparatively large (0.055798 at 250 and above). Because
the auxiliary projection ceiling also changes and coverage is not evaluated,
that pattern should not be interpreted as evidence that narrower intervals
are better calibrated.

The LQR reference emitted one iteration-limit warning at quantile 0.02 under
the demo's default 1,000-iteration limit. This is recorded in
`lqr_warnings.json`; the same fitted coefficient initializes all ceilings at
that quantile. Torchtuples also emits a deprecated PyTorch `add` warning from
its optimizer callback; the library code is retained unchanged.

`run_config.json` records 55,465 seconds of wall time because Codex suspended
the local process while the task was inactive. That field includes idle time
and is not a compute-time benchmark.
