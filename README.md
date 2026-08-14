# dplqr
This is the public code of the paper "Neural Networks for Partially Linear Quantile Regression" by Qixian Zhong and Jane-Ling Wang (2023).

To get started, you first need to install python package PyTorch (https://pytorch.org), torchtuples (https://pypi.org/project/torchtuples) and R package plaqr (https://cran.r-project.org/web/packages/plaqr). 
- demo.ipynb: Jupyter file to demonstrate the implementation of the proposed DPLQR.
- dqAux.py: auxiliary functions for the DPLQR.
- plaqr.R: R code to implement PLAQR.
- plot.R: R code to plot confidence intervals and prediction errors.
- data: Concrete Compressive Strength Data Set (concrete.csv), available at https://archive.ics.uci.edu.


# Replication, Sean Tiansui Tu, 14 August 2026, CUHK

## Replication environment

The following environment was checked on 14 August 2026. The required Python
and R packages load successfully.

### Python

- Python 3.11.9
- The project-local virtual environment is `.venv/` (excluded from Git).
- Exact installed package versions are recorded in `requirements.txt`.
- Key packages include NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.6.1,
  PyTorch 2.13.0, and torchtuples 0.2.2.
- The notebook metadata records Python 3.9.7 from the original environment;
  this replication environment uses Python 3.11.9.

To recreate the Python environment in PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### R

- R 4.6.1 (2026-06-24 ucrt)
- R executable: `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`
- Default user library: `C:\Users\Tiansui Tu\AppData\Local\R\win-library\4.6`
- `plaqr` 2.0 (archived CRAN release)
- `latex2exp` 0.9.8
- `quantreg` 6.1

The packages were verified with:

```r
library(plaqr)
library(latex2exp)
packageVersion("plaqr")
packageVersion("latex2exp")
```

A small in-memory fit/predict smoke test using `plaqr::simData` also passed
under R 4.6.1.

CRAN reports `plaqr` as unavailable for this R version through the normal
installer, so archived release 2.0 is used to match the older paper code.

> **Path note:** `plaqr.R` and `plot.R` currently refer to `../data/...`, while
> this repository stores the files in `./data/...`. If the scripts are run from
> the repository root, change those references to `data/...` first. The
> original analysis scripts have otherwise been left unchanged.

Confirm the active R setup with:

```r
R.version.string
.libPaths()
packageVersion("plaqr")
packageVersion("latex2exp")
sessionInfo()
```

The original notebook does not set NumPy, PyTorch, or train/test-split random
seeds. Exact numerical results may therefore vary between runs even with the
same package versions.

See `REPLICATION_SETUP.md` for environment checks and maintenance commands.
