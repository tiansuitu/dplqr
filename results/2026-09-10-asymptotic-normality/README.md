# DPLQR asymptotic-normality diagnostic

This experiment diagnoses inference for **theta1 = 1** in Zhong and Wang
(2024), Theorem 3.3 and homoscedastic Corollary 3.1. It uses tau=0.5, all
three Simulation I cases, and total sample sizes **500, 1000, 2000, 4000**.
Both coefficients are saved. No LQR/PLAQR models are fitted or analyzed.

**Completed:** Q=20 in all 12 cells (240 replications), with no failed fits.
At n=4000 coverage is 90%, 85%, and 75% in Cases 1-3. See
[PILOT_FINDINGS.md](PILOT_FINDINGS.md) for interpretation and
[the full report](pilot/report.md) for all sample sizes. The pilot took
13 minutes; extending its checkpoints to Q=200 is now estimated at roughly
117 additional minutes. Q=200 has not been run.

## Run and extend

From the repository root, use its existing `.venv` (Python, PyTorch,
torchtuples, NumPy, SciPy, pandas, scikit-learn, matplotlib). Base R alone is
needed for `stats::density`; `RSCRIPT` can specify the Rscript executable.
The Windows installation under `C:/Program Files/R` is detected automatically.

```powershell
# Small smoke: 4 replications, full training settings, nonlinear Case 3.
.\.venv\Scripts\python.exe results/2026-09-10-asymptotic-normality/run_normality.py run --repetitions 1 --cases 3

# Project ALL three cases at the four sample sizes before starting a pilot.
.\.venv\Scripts\python.exe results/2026-09-10-asymptotic-normality/run_normality.py estimate-runtime

# Extend the SAME default pilot/ folder; the four smoke fits are reused.
.\.venv\Scripts\python.exe results/2026-09-10-asymptotic-normality/run_normality.py run --repetitions 20

# Optional future extension: 20 -> 200, retaining all completed fits.
# This expensive command has NOT been run automatically.
.\.venv\Scripts\python.exe results/2026-09-10-asymptotic-normality/run_normality.py run --repetitions 200

# Regenerate summaries/figures without fitting anything.
.\.venv\Scripts\python.exe results/2026-09-10-asymptotic-normality/run_normality.py report --repetitions 20
```

`--cases`, `--sample-sizes`, and `--repetitions` can extend an existing grid;
they must retain all completed keys. `--output-dir` selects a fresh subfolder
here for changed seeds or training settings. Defaults are deliberately cheap:
no tuning grid, test sample, prediction-performance analysis, or benchmark
stage. There is no destructive restart option. `--help` lists settings.

## Faithful reuse and sample-size convention

`source_bridge.py` parses the September 4 notebook and loads an explicit
whitelist of its DGP, fitting, and inference definitions. It never executes
notebook imports, settings, benchmark definitions, or run cells. Networks,
masks, and losses come directly from root `dqAux.py`. The new runner only
adds replication scheduling, fit checkpoints, timing, and inferential outputs.
The September 4 notebook, root helper, demo, and previous results are unchanged.

- The same 10-dimensional equicorrelated Gaussian copula (correlation 0.5)
  gives Uniform[0,2] margins; Z is coordinates 1-8, X1 is the ninth exceeding
  1, and X2 is the tenth. Theta=(1,-1), with independent Student-t3 errors
  and the original three nuisance functions, including nonlinear Cases 2/3.
- Generate **n** observations, randomly split 80:20; **n_train = floor(0.8n)**
  estimates theta and the covariance. Standardize only Z with training-sample
  means/SDs; X and Y remain on their original scales.
- Retain the September 4 **fixed first-pass** profile: depth=2 (the constructor
  makes three hidden layers), width=32, 50% random sparsity, ReLU, dropout=1e-5,
  batch=128, maximum 100 epochs, validation early stopping patience=15,
  best-state restoration, learning rate=.005, one CPU thread. The original
  torchtuples optimizer is AdamW with zero weight decay and betas (.9,.99).
  Linear coefficients use PyTorch initialization; nuisance weights/biases
  are clipped to [-1,1] after fitting, as in September 4.
- Retain both squared-error `covNet` projections: sigmoid for X1, linear
  output for X2. The primary coefficient still requires the **full 2x2
  inverse**, not a scalar projection variance. Training residual density uses
  the exact September 4 R function, historical density coordinates, and linear
  interpolation at zero. `density_only.R` contains only that base-R routine.
- Equation (14) estimates asymptotic covariance Sigma. Coefficient covariance
  is `tau*(1-tau)*inv(Omega_hat)/(f_hat(0)^2*n_train)`, where Omega_hat is the
  centered sample covariance (`ddof=1`) of the projected X residuals. CIs use
  `theta_hat +/- qnorm(.975)*SE`; T is `(theta1_hat-1)/SE1`.

Requested `sqrt(n)` bias/SD use **total generated n**. The theorem's estimation
sample scale is also saved as `sqrt_n_train_*`. With an 80% split, total-n
scaled SD has a constant 1/sqrt(.8) factor relative to estimation-n scaling;
this does not change the rate. The retained SE already uses n_train correctly.

## Reproducibility and artifacts

Base seed 20260904 and the unchanged September 4 mapping
`base + case*10000000 + n*1000 + rep` generate data/splits; main fitting adds
500000, and projections add another 500000 plus their column index. Seeds do
not depend on Q, grid order, or prior fits. Different n use independent seeds.
CPU deterministic algorithms are enabled. Source definitions, helper code,
training configuration, and numerical/R versions are fingerprinted; incompatible
resumes fail. Presentation changes can regenerate reports without new fits.

Each replication saves `main.pt`, `projection_1.pt`, `projection_2.pt`, then
`result.json`, with atomic replacement. An interruption during inference reuses
every completed network fit; an interrupted unfinished fit is restarted.
Completed-result JSONs are authoritative, with stage SHA256 checks, and rebuild
the CSV on resume. Failures stop the run and are recorded in `run_config.json`;
no failed fit is silently discarded. A writer lock prevents simultaneous runs.
If a process is killed, remove its `.run.lock` only after confirming it exited.

- `pilot/raw_replications.csv`: estimates, SEs, CIs, coverage, T, both root-n
  error scales, seeds, data hash, selected/actual epochs and timings.
- `pilot/summary.csv` and `pilot/quantiles.csv`: per-case/per-n bias, scaled
  bias/SD, sample SD (`ddof=1`), mean SE, SE/SD, RMSE, T moments, unbiased
  skewness/excess kurtosis, linearly interpolated empirical tail/median
  quantiles and normal references, coverage and Monte Carlo uncertainty.
- `pilot/figures/`: histograms/KDE with N(0,1), normal QQ, studentized shape
  and quantiles, across-n scaled bias/SD, T mean/SD, and coverage, plus root-n
  error QQ plots against fitted normal lines to assess estimator shape.
- `pilot/report.md`: numerical results and their interpretation.
- `smoke-test/`: frozen four-replication CSV/config and runtime projections,
  saved **before** launching Q=20. Its fits live in `pilot/` and were reused.
- `verification.json`: numerical parity and stage-restoration checks. Rerun
  `verify_experiment.py` to intentionally repeat one small DPLQR replication
  for parity and prohibit training while testing checkpoint restoration.
- `pilot/validation.json`: no-fit checks of raw/summary formulas, all network
  checkpoint hashes, and unchanged reuse of smoke rows; `validate_outputs.py`
  reproduces these checks after report generation.

The smoke measured about **7.3 minutes for Q=20** (240 main + 480 auxiliary
fits), or **72.9 minutes for Q=200** (2400 main + 4800 auxiliary fits), excluding
startup/reporting. It observed one Case 3 replication per n and extrapolated
to the other cases, so early-stopping variability makes these rough estimates.
The Q=200 experiment was not launched. See the saved runtime report for detail.

## What the experiment can establish

Shrinking raw bias **and spread/RMSE** support ordinary consistency. Stable
sqrt(n)-SD with vanishing sqrt(n)-bias supports the centered root-n behavior
sought in Theorem 3.3. Shrinking raw bias with persistent nonzero scaled bias
can coexist with consistency and a root-n rate, but is problematic for that
centered limit. Four finite sample sizes cannot establish a limiting claim.

Normal QQ shape, skewness and kurtosis assess shape; T mean near zero, SD near
one, normal tail quantiles and 95% coverage jointly assess centering and SE
calibration. Poor coverage alone cannot identify bias, nonnormality, or faulty
variance estimation. Mean SE versus empirical SD helps separate scale error.
Q=20 is a pilot: coverage changes in 5-point steps and has nominal Monte Carlo
SE about **4.9 percentage points** (1.54 at Q=200). Tail quantiles and higher
moments are especially noisy. Wilson intervals show coverage uncertainty;
scaled-bias error bars are approximate Monte Carlo intervals, not theorem tests.

This fixed architecture and finite early-stopped optimizer do **not** implement
the theorem's growing network-class assumptions or guarantee its empirical
minimizer/optimization accuracy. Findings diagnose this retained computational
procedure under Simulation I; they neither prove nor logically refute the
theorem. A failed approximation may reflect nuisance approximation or
optimization bias as well as inference error.

Sources inspected: repository `Zhong and Wang 2024 JBES.pdf`, pp. 606-608
(Theorem 3.3, Corollary 3.1, Eq. 14 and Simulation I); supplement pp. 4-6
(proof) and Table 13, p. 18 (tuning); root `dqAux.py` and `demo.ipynb`;
`results/2026-09-04-homoscedastic-simulation/` notebook, README and density code.
