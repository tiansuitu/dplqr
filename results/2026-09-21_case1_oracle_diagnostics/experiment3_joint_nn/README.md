# Experiment 3 — Joint Adam (θ, m) DPLQR-style estimator

## How to run (primary)

Open this notebook and **Run All**:

`experiment3_joint_nn.ipynb`

You do **not** need to run the `.py` by hand.  
`run_experiment3.py` is a modular helper that the notebook imports (same pattern as Experiments 0 and 2). Optional CLI:

```powershell
..\..\..\.venv\Scripts\python.exe run_experiment3.py
```

## Purpose

Full **joint** estimation of \((\theta,m)\) by Adam on the empirical median check loss:

\[
L_n(\theta,m)=\frac1n\sum_i\rho_{0.5}\bigl(Y_i-X_i^\top\theta-m(Z_i)\bigr),\qquad \rho_{0.5}(u)=\tfrac12|u|.
\]

Compare later with:

| Experiment | What it isolates |
|------------|------------------|
| 0 | exact \(m_0\) supplied |
| 2 | NN \(m\) with \(\theta\) fixed at \(\theta_0\) |
| **3** | joint feedback \(\theta\leftrightarrow m\) |

Oracle throughout: exact \(f_0\), exact \(\varphi^*(Z)\), known \(m_0\) for diagnostics only.

## Why \(m^*=m_0\)

On \(z_j\in[0,2]\), \(\mathrm{ReLU}(z_j)=z_j\). Thus \(m_0(z)=0.56\sum_j z_j\) lies in a dense ReLU class with identity paths (widths `[32,32,16]`, **no sparsity mask**). Population sieve error can be taken as zero.

## Primary vs secondary estimator

1. **Primary:** `theta_hat_joint_adam` — final joint Adam iterate (this *is* Experiment 3).
2. **Secondary (diagnostic only):** `theta_hat_joint_profiled` — after freezing \(\hat m_{\mathrm{joint}}\), re-solve \(\theta\) by convex `linprog`. Does **not** replace the primary estimator.

## Config (defaults; match Exp 2)

| Key | Value |
|-----|-------|
| `N_VALUES` | `[1000, 2000]` |
| `Q` | `50` |
| `BASE_SEED` | `20260921` |
| `HIDDEN` | `[32, 32, 16]` |
| `EPOCHS` | `1000` (**no early stopping**) |
| `LEARNING_RATE` | `1e-3` (Adam) |
| `BATCH_SIZE` | `128` |
| `f_0` | `2/(π√3)` |
| sandwich / IF / \(C_n\) | residualized \(\tilde X=X-\varphi^*(Z)\) |

## Outputs (created when you run)

```
run/
  config.json
  replication_results.csv
  summary.csv
  figures/
```

Restartable: completed `(n, replication)` rows are skipped.
