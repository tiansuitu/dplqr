# Experiment 2 — NN nuisance with \(\theta\) fixed at \(\theta_0\)

## Purpose

Experiment 2 asks: what happens when the nuisance \(m\) is estimated by a neural net **while \(\theta\) is held at the true** \(\theta_0=(1,-1)^\top\) during that training?

This removes the feedback channel

\[
\text{error in }\hat\theta \;\longrightarrow\; \text{contamination of }m\text{-training}.
\]

After fitting \(\hat m_{\mathrm{fixed}}\), freeze it and re-estimate \(\theta\) by a **global convex** median regression (same LP style as Experiment 0).

Oracle throughout:

- exact \(f_0=2/(\pi\sqrt{3})\) (Student-\(t_3\) density at 0);
- exact \(\varphi^*(Z)=E[X\mid Z]\) from the equicorrelated Gaussian construction;
- known \(m_0\) for diagnostics only.

**No early stopping.** Train for exactly `EPOCHS` epochs on all \(n\) observations.

## Relation to Experiment 0

| | Experiment 0 | Experiment 2 |
|--|--------------|--------------|
| \(m\) in \(\theta\)-step | exact \(m_0\) | frozen \(\hat m_{\mathrm{fixed}}\) |
| \(m\) training | none | NN, \(\theta\equiv\theta_0\) |
| \(\theta\) solver | convex LP | convex LP |
| \(f_0,\varphi^*\) | oracle | oracle |

Any gap vs Experiment 0 is therefore attributable primarily to **finite-sample NN nuisance fitting / optimization** and its empirical interaction with the score — **not** to joint \(\theta\)–\(m\) feedback.

## Why \(m^*=m_0\)

On \(z_j\in[0,2]\), \(\mathrm{ReLU}(z_j)=z_j\). Thus \(m_0(z)=0.56\sum_{j=1}^8 z_j\) lies in a sufficiently flexible ReLU class (identity paths, final weights \(0.56\)). This pack uses a dense MLP with widths `[32,32,16]` and **no random sparsity mask**, so the class can represent \(m_0\). Population sieve error can be taken as zero; observed nuisance error is estimation / optimization error.

## Primary diagnostics

1. **\(C_n(\hat m)\)** — \(\sqrt{n}\,P_n[f_0\,\tilde X\,(\hat m-m_0)]\). Does it shrink with \(n\), or stay \(O_p(1)\)?
2. **IF gap** — \(\Sigma_2\sqrt{n}(\hat\theta^{(2)}-\theta_0)+S_n\) with true \(\varepsilon\) and oracle \(\tilde X\).
3. **`nuisance_L2`** — \(\|\hat m-m_0\|_{n,2}\).

## Configuration (defaults)

| Key | Value |
|-----|-------|
| `N_VALUES` | `[1000, 2000]` |
| `Q` | `50` |
| `BASE_SEED` | `20260921` |
| `tau` | `0.5` |
| `theta_0` | `(1, -1)` |
| `HIDDEN` | `[32, 32, 16]` |
| `EPOCHS` | `1000` (exact; no early stop) |
| `LEARNING_RATE` | `1e-3` (Adam) |
| `BATCH_SIZE` | `128` |
| `dtype` | `float64` |

## How to run

```powershell
cd results\2026-09-21_case1_oracle_diagnostics\experiment2_fixed_theta_nn
..\..\..\.venv\Scripts\python.exe run_experiment2.py
```

Or open `experiment2_fixed_theta_nn.ipynb` and run all cells.

Restartable: completed `(n, replication)` rows in `run/replication_results.csv` are skipped.

## Outputs

```
run/
  config.json
  replication_results.csv
  summary.csv
  figures/
```

## Zhong–Wang alignment

DGP matches JBES 2024 Simulation I, Case 1 (Gaussian copula \(\rho=0.5\) on \([0,2]\), \(\theta_0=(1,-1)\), \(m_0=0.56\sum z_j\), \(t_3\) errors, \(\tau=0.5\)). Intentional differences from the paper’s *estimation* pipeline: oracle \(f_0/\varphi^*\), fixed-\(\theta\) nuisance training, no early stopping, no sparsity mask, \(Q=50\), convex LP for the final \(\theta\) step.
