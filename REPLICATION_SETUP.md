# Replication setup notes

These notes record the local setup used to replicate the paper on 14 August
2026. Run PowerShell commands from the repository root:

```powershell
cd "C:\Users\Tiansui Tu\Documents\GitHub\dplqr"
```

## Python environment

Activate the existing project environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python --version
```

The execution-policy change applies only to the current PowerShell window. It
does not make a permanent system-wide change.

Recreate the environment on another Windows computer:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that installed packages have compatible dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

After intentionally installing, removing, or upgrading a Python package,
refresh the package record:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
Get-Content requirements.txt
```

Do not commit `.venv/`. It is machine-specific and is excluded by `.gitignore`.

## R environment

R 4.6.1 is installed at:

```text
C:\Program Files\R\R-4.6.1
```

In PowerShell, `r` is a built-in alias for `Invoke-History`, and `Rscript.exe`
is not currently on `PATH`. Use the complete executable path when launching R
from PowerShell:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" --version
```

The default R 4.6 user-package location is:

```text
C:\Users\Tiansui Tu\AppData\Local\R\win-library\4.6
```

The completed setup uses these package versions:

```text
plaqr 2.0 (archived CRAN release)
latex2exp 0.9.8
quantreg 6.1
```

An in-memory `plaqr` fit and prediction using the package's `simData` example
completed successfully under R 4.6.1.

They load from the R 4.6 user library. Verify the versions and library
locations with:

```r
R.version.string
.libPaths()
packageVersion("plaqr")
packageVersion("latex2exp")
sessionInfo()
```

If packages exist under a library for an older R release, reinstall them for
R 4.6 instead of copying the old library. R's default user library is
version-specific.

### R script working-directory note

The repository contains its datasets and result files under `./data/`, but the
original `plaqr.R` and `plot.R` scripts read and write `../data/...`. Their
original path statements are retained as comments, followed by active
`data/...` paths. The scripts can therefore be run from the repository root
while preserving the authors' original path choices for reference.

## Before the first replication run

1. Commit the setup files so the pre-run state is recoverable.
2. Open `demo.ipynb` and select
   `.venv\Scripts\python.exe` as the notebook kernel.
3. Run the notebook from the repository root so its `./data/...` paths resolve.
4. Expect a CPU-only run: the installed PyTorch build is `2.13.0+cpu`, and the
   notebook trains across many quantile levels for up to 500 epochs each.
5. Preserve or compare the existing tracked result CSV files. The notebook
   overwrites `data/resDeep.csv`, `data/resLin.csv`, `data/df_train.csv`, and
   `data/df_test.csv`; `plaqr.R` overwrites `data/resAdd.csv` after its paths are
   corrected.
6. Treat the first run as a reproduction of the authors' code. The notebook
   does not set NumPy, PyTorch, or `train_test_split` random seeds, so exact
   results can vary. Add fixed seeds only as a separately documented robustness
   run if exact repeatability is needed.

## Git checks

Inspect pending changes:

```powershell
git status
```

Confirm that Git ignores the local Python environment:

```powershell
git check-ignore -v .venv
git ls-files .venv
```

The second command should print nothing. `.gitignore` prevents new matching
files from being tracked; it does not delete files from the computer.

Files that should be committed for the replication setup include:

```text
.gitignore
requirements.txt
README.md
REPLICATION_SETUP.md
```

The `.venv/` directory should not be committed.

## Compatibility change log

This section records changes needed to run the authors' older demonstration
code with the replication environment. Original code should be retained as a
nearby comment whenever practical.

### 26 August 2026 — pandas parameter indexing

- File: `demo.ipynb`
- Status: applied locally, not yet committed
- Original code: `paramsLin = model_lin.params[1]`
- Replacement: `paramsLin = model_lin.params.loc["inX"]`
- Reason: with pandas 3.0.5, integer indexing on a Series with string parameter
  labels is label-based and raises `KeyError: 1`. Selecting `"inX"` explicitly
  identifies the coefficient intended by the analysis.

### 26 August 2026 — continuous projection loss

- File: `demo.ipynb`
- Status: applied locally, not yet committed
- Original call: `getSESingle(..., logic=True)`
- Replacement: `getSESingle(..., logic=False)`
- Reason: in `dqAux.py`, `logic=True` selects a sigmoid output and binary
  cross-entropy loss. Its target in the concrete application is `inX`, the
  continuous water-cement ratio, whose observed range is approximately 0.267
  to 1.882; 198 observations exceed 1. Modern PyTorch therefore raises
  `RuntimeError: all elements of target should be between 0 and 1`.
  `logic=False` selects an unrestricted output and mean squared error, which is
  the appropriate branch for a continuous target.
- Reproducibility note: this is a methodological compatibility correction and
  may produce numerical results different from the authors' saved output.
- Scope clarification: We retain quantile/check loss for the main DPLQR
  estimation, but replace an inappropriate binary auxiliary projection with
  continuous MSE regression for standard-error estimation.

## Complete concrete-data demonstration: 26 August 2026

### Execution outcome

The complete public repository workflow ran successfully on 26 August 2026:

1. `demo.ipynb` fitted DPLQR and LQR at the 49 quantile levels from 0.02 to
   0.98 and exported the train/test split and Python results.
2. `plaqr.R` fitted PLAQR at the same 49 quantile levels using the exact split
   exported by Python.
3. `plot.R` combined the DPLQR, LQR and PLAQR results into a four-panel figure.

An automated second run produced an executed notebook with zero error outputs.
The complete dated artifacts are under `results/2026-08-26/`. The user's first
successful run is preserved in `initial-user-run/`; the automated confirmation
run is preserved in `codex-rerun/` together with a high-resolution PNG and the
PLAQR console transcript.

Environment used for the automated run:

```text
Python 3.11.9
NumPy 2.4.6
pandas 3.0.5
scikit-learn 1.6.1
PyTorch 2.13.0+cpu
torchtuples 0.2.2
R 4.6.1
plaqr 2.0
quantreg 6.1
latex2exp 0.9.8
```

### What Supplementary Table 9 measures

Supplementary Table 9 is not the target of the repository's concrete-data
demonstration. It reports empirical coverage probabilities for 95% confidence
intervals for two coefficients under simulated heteroscedastic errors. It uses
Cases 4-6, sample sizes 1,000 and 2,000, and quantile levels 0.25, 0.50 and
0.75. Its entries are repeated-simulation coverage proportions, mostly between
0.875 and 0.960.

In contrast, the repository demonstration uses the real Concrete Compressive
Strength dataset with 1,030 observations, one random train/validation/test
split, one linear water-cement-ratio coefficient, and 49 quantile levels. It
reports integrated confidence-interval area and average test-set check loss.
These are different datasets, estimands and evaluation statistics, so its
numbers cannot and should not equal Supplementary Table 9. Table numbering is
continuous: the published article contains Tables 1-4 and the supplement
continues with Tables 5 onward.

The appropriate published comparison for this repository workflow is instead
the supplement's Concrete Compressive Strength application, especially Figures
5 and 6.

### Comparison with Supplementary Figures 5 and 6

| Source/run | Method | 95% CI area | ACL |
|---|---|---:|---:|
| Supplement Figure 5/6 | DPLQR | 0.2891 | 0.0675 |
| Supplement Figure 5/6 | LQR | 0.3408 | 0.1052 |
| Supplement Figure 5/6 | PLAQR | 0.3137 | 0.0786 |
| Supplement Figure 6 | DNQR | not reported | 0.0626 |
| Initial user run | DPLQR | 0.403194 | 0.066841 |
| Initial user run | LQR | 0.315169 | 0.100762 |
| Initial user run | PLAQR | 0.309762 | 0.071060 |
| Automated confirmation run | DPLQR | 0.440810 | 0.061441 |
| Automated confirmation run | LQR | 0.318443 | 0.099818 |
| Automated confirmation run | PLAQR | 0.328683 | 0.075505 |

The qualitative prediction conclusion is reproduced: DPLQR has lower ACL than
the two interpretable comparison methods, PLAQR is second, and LQR is third.
The reproduced ACL values are also close to the supplement, although the
automated run is modestly lower for all three methods. DNQR cannot be rerun or
fairly compared because its implementation is not included in this repository.

The LQR and PLAQR confidence-interval areas remain fairly close to the
supplement. The automated DPLQR interval area, however, is 0.440810 rather than
0.2891, about 52% wider. The continuous-MSE compatibility correction is a
likely major contributor: it changes only the auxiliary projection used for
DPLQR standard errors and therefore directly affects DPLQR intervals, while
leaving the main check-loss estimator and its predictions intact. This branch
also matches the paper's stated squared-error projection of the continuous
linear covariate. The result should therefore be described as a transparent
methodological correction, not a bit-for-bit reconstruction of the authors'
original confidence bands.

Additional sources of numerical variation are the absence of fixed NumPy,
PyTorch and split seeds, random neural-network initialization and sparsity
masks, early stopping, and substantial software-version differences. The two
successful runs themselves demonstrate this stochastic variation.

### R prediction warning

`plaqr.R` completed and produced all 49 rows, but prediction emitted one warning
per quantile that some `flh` test values were beyond the training B-spline
boundary knots and could make the basis ill-conditioned. The warning is caused
by fitting spline boundaries on one random training split and then predicting
outside that range. It did not stop prediction or invalidate file creation, but
it should be retained in the replication record and examined in any robustness
analysis.

## Epoch-ceiling sensitivity: 03 September 2026 (03092026)

Added `results/2026-09-03-epochs-vary/epochs_vary.py`, based on `demo.ipynb`,
to compare ceilings of 50, 100, 250, 500 (the demo default), and 1,000 epochs
for both DPLQR and its auxiliary SE projection. The experiment uses all 49
quantiles, one fixed 742/82/206 train/validation/test split, and three paired
training seeds (11, 22, 33). Architecture, learning rate, batch size, sparsity,
validation early stopping (patience 10), and the continuous-MSE correction
remain as in the current demo. Actual and selected-best epochs are recorded.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe results/2026-09-03-epochs-vary/epochs_vary.py --verify
```

The folder contains the comparison table (`summary.md`/`.csv`), figure
(`epochs_comparison.png`/`.pdf`), per-seed and per-quantile results, training
histories, split files, and configuration. The shared-training-path shortcut
was checked against independent default fits at every ceiling for the median
quantile: weights matched exactly and standard errors matched `getSESingle`.
All outputs are separate from the original demo and August results.

Mean test ACL was 0.076169, 0.068076, and 0.066808 at ceilings 50, 100, and
250 respectively. Results at 250, 500, and 1,000 were identical: across every
seed and quantile, the main fit stopped by epoch 220 and the auxiliary fit by
epoch 113. Thus 50 was clearly restrictive, 100 retained a small penalty, and
the demo's 500 ceiling was comfortably above the observed stopping point in
this controlled split-and-seed experiment.
