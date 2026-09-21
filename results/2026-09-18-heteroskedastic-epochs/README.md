# 2026-09-18 — Heteroskedastic DPLQR epoch drift

## Question

Under Zhong & Wang (2024) Simulation II–style DGPs, which **stronger-than-paper** heteroskedasticity / error laws make DPLQR `\(\hat\theta\)` drift away from structural `\(\theta_0=(1,-1)\)` as training epochs increase, even when \(n\) is large?

DPLQR only (no LQR / PLAQR).

## Paper Simulation II (baseline)

Same \(X,Z,\theta,m\) as Simulation I; \(Y=X^\top\theta+m(Z)+\sigma(X,Z)\varepsilon\), \(\varepsilon\sim t_3\).

| Case | \(\sigma\) |
|------|------------|
| 4 (linear) | \((x_1+x_2+\sum z_k)/5\) (paper text has `x1+x1` typo → treat as \(x_1+x_2\)) |
| 6 (deep) | \((x_1+x_2)/3 + 3\Phi(\sum(z_k-1)/5)\) |

Also included: the milder 2026-09-15 formula \((x_1+x_2)/10+\Phi(\mathrm{mean}(z-1))\), plus sharper designs (`steep_x`, `interaction`) and explicit scale multipliers (1, 2, 5).

## Recycled pieces

- Hyperparameters / continuous-trajectory idea: `results/2026-09-15_heteroskedastic_epoch_drift_case6`
- Covariates + deep \(m(\cdot)\): Sept 4 homoscedastic notebook definitions
- Network / check loss: repo-root `dqAux.py` (`dqNetSparse`, `checkLoss`)

## Run

From the repo root, project venv:

```powershell
cd "C:\Users\Tiansui Tu\Documents\GitHub\dplqr"
.\.venv\Scripts\python.exe results\2026-09-18-heteroskedastic-epochs\heteroskedastic_epochs.py --verify
```

Full grid (CPU; long):

```powershell
.\.venv\Scripts\python.exe results\2026-09-18-heteroskedastic-epochs\heteroskedastic_epochs.py --output-name run
```

Useful knobs: `--n-list 1000,2000` `--epochs 200,500,1000,2000` `--reps 20` `--designs paper6_s3_t3,steep_s1_t3`.

Smoke designs: `mild_s1_t3`, `paper6_s1_t3`, `paper6_s3_t3`, `steep_s1_t3`, `interact_s1_t3`, `paper6_s3_normal`.

## Outputs

Under `smoke/` or `run/`:

- `run_config.json` — design grid and HP
- `raw_checkpoints.csv` — per-replication \(\hat\theta\) at each checkpoint epoch
- `summary.csv` — mean bias / RMSE vs \(\theta_0\) (and vs \(\theta_\tau\) when \(\sigma\) is affine in \(X\))
- `epoch_drift.csv` — \(|\mathrm{bias}|\) change from first to last epoch
- `run_meta.json` — wall time

Primary readout: rows in `epoch_drift.csv` with large positive `abs_bias_delta` at large `n_total`.
