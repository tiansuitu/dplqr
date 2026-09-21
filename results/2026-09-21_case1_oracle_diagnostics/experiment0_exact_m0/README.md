# Experiment 0 — Exact \(m_0\) oracle (Case 1, \(\tau=0.5\))

## Purpose

This is the cleanest oracle benchmark for Zhong–Wang Simulation I, Case 1 (homoscedastic errors, \(\tau=0.5\)).

We feed in the **exact** nuisance \(m_0\) and estimate only \(\theta\) by convex median regression. There is:

- no sieve approximation error (\(m^*=m_0\));
- no nuisance estimation;
- no neural-network optimization;
- no joint \(\theta\)–\(m\) feedback;
- no error-density estimation (\(f_0\) is analytic);
- no projection estimation (\(\varphi^*\) is analytic).

If Experiment 0 itself fails the stated influence-function diagnostics, the issue **cannot** be blamed on NN nuisance estimation or joint optimization.

## Why \(m^*=m_0\)

\[
m^*\in\arg\min_{m\in\mathcal{M}_A}\|m-m_0\|_{L^2}.
\]

Here \(m_0(z)=0.56\sum_{j=1}^8 z_j\) with \(z_j\in[0,2]\), so \(\mathrm{ReLU}(z_j)=z_j\). A ReLU network with identity paths and final weights \(0.56\) represents \(m_0\) exactly, hence \(m_0\in\mathcal{M}_A\) and \(\inf\|m-m_0\|_{L^2}=0\). We take \(m^*=m_0\) and supply it directly.

## What is removed vs what remains

| Removed | Remains |
|--------|---------|
| sieve approx error | parametric median QR for \(\theta\) |
| nuisance estimation | Monte Carlo sampling variability |
| NN / joint optimization | finite-\(n\) QR asymptotics |
| density / \(\varphi^*\) estimation | oracle sandwich \(V/n\) |

## How to run

From this folder (or open the notebook in Jupyter with this folder as cwd):

```powershell
cd results\2026-09-21_case1_oracle_diagnostics\experiment0_exact_m0
jupyter notebook experiment0_exact_m0.ipynb
```

Run all cells. The notebook is **restartable**: completed `(n, replication)` pairs in `run/replication_results.csv` are skipped.

**Do not** expect this pack to execute Monte Carlo for you automatically outside the notebook — you run it manually.

## Configuration (defaults)

| Key | Value |
|-----|-------|
| `N_VALUES` | `[1000, 2000]` |
| `Q` | `50` replications per \(n\) |
| `BASE_SEED` | `20260921` |
| `tau` | `0.5` |
| `theta_0` | `(1, -1)` |
| `f_0` | `2/(π√3)` (Student-\(t_3\) density at 0) |

## Outputs (created when you run the notebook)

```
run/
  config.json
  replication_results.csv
  summary.csv
  figures/
```

## Influence-function check

With oracle \(\tilde X=X-\varphi^*(Z)\) and exact \(f_0\),

\[
\Sigma_2\sqrt{n}(\hat\theta-\theta_0)\approx -S_n,
\quad
S_n=\frac1{\sqrt n}\sum_i\tilde X_i\bigl(\mathbf{1}\{\varepsilon_i<0\}-\tau\bigr).
\]

`IF_gap` is the left-hand side plus \(S_n\). In Experiment 0, \(C_n(m_0)=0\) exactly.

## Sandwich note (corrected 2026-09-21)

With known \(m_0\), Experiment 0 is ordinary linear median regression of \(Y-m_0(Z)\) on \(X\). Oracle SEs / \(T\) / coverage / IF diagnostics use **raw \(X\)** (not \(\tilde X=X-\varphi^*(Z)\)). Residualized \(\tilde X\) is reserved for estimated-\(m\) experiments. See `SANDWICH_CORRECTION.md`.
