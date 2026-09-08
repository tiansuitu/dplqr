# Simulation I: Homoscedastic Errors

Created 04 September 2026 (04092026); checked against the paper and supplement
08 September 2026 (08092026). Targets: Zhong and Wang (2024), pp. 608-609,
Tables 1-3; tuning candidates: supplement Table 13, p. 18.

## Open and run

Open **simulate_homoscedastic.ipynb** in Jupyter or VS Code and select the
repository's `.venv` Python kernel, as for `demo.ipynb`. The notebook contains
the Python source in cells, editable experiment settings, the Python run, the
R PLAQR run, and displays of reproduced and published tables. It is supplied
without saved numerical results. Run cells in order, from either the repository
root or this folder. This notebook is the only simulation Python source; it
imports the original root `dqAux.py` directly.

The active settings are the agreed **first pass**: two repetitions, all three
cases, n=1000, test size=1000, and tau=0.5. Fixed tuning uses depth=2, width=32,
100 epochs, batch size=128, patience=15, learning rate=0.005, and one CPU thread.
Section 8 fits six DPLQR networks, twelve auxiliary networks, and six LQR models.
Its twelve result rows are a preliminary check; two repetitions do not provide
reliable Monte Carlo bias, SD, or coverage estimates.
This exact profile passed end-to-end validation: about 20 seconds for Python
and 4 seconds for R on this machine, with 18 combined result rows.

Conflicting full-replication assignments remain commented immediately above
the active replacements, marked **`# *%%* FULL REPLICATION:`**, in the notebook
and the R defaults. To restore the full profile, uncomment those assignments
and comment their active replacements. The full experiment uses 200 repetitions
and the 384-candidate grid (1,382,400 DPLQR candidate fits). The preserved epoch,
patience, and learning-rate scalars are original fixed-mode defaults; grid mode
searches the supplement's complete candidate sets instead.

Run Sections **1-7** before Section 8. In VS Code, select the Section 8 code
cell and use **Run All Above**, then run Section 8 to produce LQR/DPLQR CSVs.
Section 9 optionally adds PLAQR. **Restart Kernel and Run All** runs every stage
with the current settings, including the first-pass experiment with the current defaults.

If a kernel is restarted, earlier imports and settings must be run again.
Definition cells defer type annotations so loading a function cell by itself
does not evaluate `pd.DataFrame` before pandas is imported. The run cells check
for missing setup and explain which earlier sections to execute.

Both stages default to `first-pass-run/`. R uses the exact Python exports in
`first-pass-run/data_for_r/`, reads Python's LQR/DPLQR results, and adds PLAQR.
Section 9 passes the notebook settings automatically; standalone R defaults
also match this first pass. R performs six PLAQR fits, giving eighteen combined
result rows. The commented full profile uses `full-run/`.
R must be installed before running Python because DPLQR inference calls
`stats::density` in R. If Rscript is not detected, set the environment variable
`RSCRIPT` to its executable. PLAQR uses installed `plaqr` 2.0 and `quantreg`;
the ignored `_r_libs/` folder supplies the local library. To install if needed:

```r
dir.create("results/2026-09-04-homoscedastic-simulation/_r_libs", showWarnings=FALSE)
lib <- "results/2026-09-04-homoscedastic-simulation/_r_libs"
install.packages(c("SparseM", "MatrixModels", "quantreg"), repos="https://cloud.r-project.org", lib=lib)
install.packages("https://cran.r-project.org/src/contrib/Archive/plaqr/plaqr_2.0.tar.gz", repos=NULL, type="source", lib=lib)
```

## What the code implements

- The 10-dimensional Gaussian copula has latent correlation 0.5 and
  Uniform[0,2] margins. `Z` uses coordinates 1-8, `X1 = I(Z9 > 1)`, and
  `X2 = Z10`; theta is `(1,-1)` and errors are independent Student t(3).
- The three nuisance functions follow the equations on p. 608, including
  `sqrt(z6 + 0.5)` in Case 2. The true quantile nuisance function is
  `m_tau(z) = m(z) + qt(tau, 3)`.
- Settings are `n = 1000, 2000`, quantiles `0.25, 0.50, 0.75`, a random 80:20
  estimation/validation split, and 5,000 fresh test observations per repetition.
- Table 1 gives signed bias and Monte Carlo sample SD of theta1. Table 2 gives
  empirical 95% interval coverage. Table 3 is the ratio of mean squared error
  to mean squared true nuisance value, with **no square root** (Eq. 16).
- LQR follows the demo's `statsmodels.formula.api.quantreg`; PLAQR follows the
  original `plaqr.R` formula, fit, prediction, and check-loss calculation.
  Both use the 80% estimation sample specified in Simulation I.
- DPLQR directly reuses `dqNetSparse`, `checkLoss`, `checkErrorMean`, and
  `covNet` from the root `dqAux.py`. Training uses `torchtuples.Model` and the
  original optimizer. The linear branch uses PyTorch initialization, as
  described on p. 607. The nonlinear weights and biases are clipped to [-1,1]
  after training; validation compares the resulting candidate estimators.
- DPLQR inference uses squared-error auxiliary projections for both linear
  covariates, R `stats::density`, the 2-by-2 covariance in Eq. 14, and the
  estimation sample size (800 or 1600). Validation observations tune networks.
- PLAQR explicitly requests alpha=0.05. Its package-default small-sample rank
  bounds are used directly; when the package returns standard errors at the
  larger sample size, the interval is estimate +/- qnorm(.975) times SE.
  Raw results store the actual interval bounds used for coverage.

## Details the released material does not settle

The paper and released demo differ in some conventions. We retain the authors'
network implementation: a `depth` argument d creates d+1 hidden layers and d+2
affine layers. The supplement's depth grid includes 1, while the theoretical
layer definition requires at least 2 affine layers. We therefore use the
released constructor convention and record it explicitly, rather than infer
an undocumented remapping.

Likewise, the released network applies a random approximately 50% mask to
internal hidden matrices. The paper describes 50% zeros per row. The source's
masking is preserved, so this is a known difference from a literal reading of
the paper. The root source files are not modified.

The released `torchtuples.Model` default is also preserved: AdamW with zero
weight decay and beta values (0.9, 0.99). The paper names Adam without reporting
these coefficients; they are another inherited implementation choice.

Random seeds and selected tuning settings are not reported. Projection-network
tuning and exact density settings are also unspecified. We use the selected
DPLQR profile for projections (sigmoid output for binary X1, linear output for
X2, both with squared-error loss), and historical R density defaults
(`old.coords=TRUE` when available) with linear interpolation at zero. We use
training observations for the covariance because only 80% estimate theta.
These choices are explicit; exact agreement with the published numbers cannot
be guaranteed from the released material.

## Outputs and resuming

`published_table1.csv`, `published_table2.csv`, and `published_table3.csv` are
transcriptions of the paper, not local simulation results. The final run folder
contains `raw_results.csv` (Python), `raw_results_r.csv` (PLAQR),
`raw_results_combined.csv`, `simulation_summary_long.csv`, `table1_bias_sd.csv`,
`table1_bias.csv`, `table1_sd.csv`, `table2_coverage.csv`, and `table3_rmse.csv`.

Each stage checkpoints completed fits. Python refuses to mix changed settings
or source code with previous results; repetitions may be extended. R verifies
that all requested Python fits and exports exist and refuses changed inputs
when resuming. Use a fresh output folder for a changed experiment. Match any
custom settings in both stages; the notebook passes them automatically.
Resume checks fingerprint the live notebook functions (including defaults and
nested functions), scientific constants, and the original helper/R files.
Execution counts, cell filenames, and saved outputs do not change that identity.
Changes made and executed in source cells are detected even before saving the
notebook. Use a fresh output directory for a changed implementation.

Verification covers reduced end-to-end runs, both PLAQR interval formats,
Table 1-3 calculations, loading Section 7 first, and two fresh Jupyter kernels
started in different repository folders. Live-code/default/constant changes
were detected and an unchanged cross-kernel resume preserved the CSVs. These are
software checks; the full 200-repetition grid experiment has not been run.
