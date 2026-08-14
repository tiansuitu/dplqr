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
original `plaqr.R` and `plot.R` scripts read and write `../data/...`. Therefore,
running either script from the repository root will produce a file-not-found
error even after the packages are installed.

Before running the scripts from the repository root, change their path strings
from `../data/` to `data/`. This setup check did not alter the original analysis
scripts.

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
