# Task B: toy PLQR, joint (θ, t) LP with a cross-fitted φ̂

Toy design from `2026-09-23_psi_n_thm33/run_expansion_v2/main_26a7078945fc`: θ₀ = 1, τ = 0.5, n ∈ {500, 1000, 2000, 4000}, reps 1..50 per n.

## Method
* **m̂** is the in-sample Adam `m_net` prediction from the saved `polished` checkpoint. It is held fixed.
* **φ̂(Z) = Ê[X|Z]** is 2-fold cross-fitted on each rep's own sample. Two learners are used:
  `MLPRegressor(32,32)` with early stopping (`phihat_lp_mlp`, primary) and `GradientBoostingRegressor` (`phihat_lp_gb`, robustness check).
* **Joint LP**: min over (θ, t) of Σ ρ_τ(Y − m̂ − θX − tφ̂), solved with `lp_fit` from `2026-09-27_phi_augmented_lp/run_phi_lp.py`.
* **Comparators**: `adam` (the Adam θ̂) and `phistar_lp` (the joint LP with the oracle φ*). Both are recomputed and checked against `2026-09-27_phi_augmented_lp/run/replication_results.csv`. θ̂ matches to 1e-8, which is asserted.
* **SEs**: var = τ(1−τ) / (f(0)² · Var(X̃) · n).
  *feasible*: X̃ = X − φ̂ (the MLP φ̂ for the adam and phistar rows), with f̂(0) from R `stats::density` of the row's residuals.
  *oracle*: X̃ = V = X − φ*, with f(0) = dnorm(0). *pop_oracle*: the fixed population SD (2.5066).

## How it was run
```
python run_xfit_phihat.py --n <n> --cache-only                       # fills _cache/n_{n}_rep_{rep:04d}.json (resumable)
python run_xfit_phihat.py --n 500 1000 2000 4000 --reps 50           # loads the cache and writes the CSVs
```
The first run (`run_log_interrupted_attempt1.txt`) was interrupted. The cache was then refilled by per-n runs (`log_n*.txt`, started at 14:23 HKT on 2026-09-28), which were also interrupted, at 49/50 reps for n=2000 and 28/50 for n=4000.
Those were resumed from the cache in parallel (`log_n2000_resume.txt`, `log_n4000_resume.txt`; the `err_*` files are empty). Aggregation output is in `aggregate_log.txt`.
Final count: 50 reps × 4 methods × 4 n = 800 rows.

## Files
* `replication_results.csv` has one row per (n, rep, method).
* `summary.csv` has √n bias, MCSE, SD, RMSE, coverage (feasible, oracle, pop_oracle), mean |√n φ*-score|, mean |S|, mean ‖φ̂−φ*‖, mean ‖m̂−m₀‖, and the mean of √n‖m̂−m₀‖‖φ̂−φ*‖.
* `paired.csv` has paired per-rep shifts, φ̂-LP minus {adam, phistar_lp}, for √n(θ̂−θ₀), the √n φ*-score, and S.
* `smoke/` holds an earlier smoke test. `_cache/` holds the per-rep JSON files.

## Results (50 reps per cell)
√n bias ± MCSE / SD / RMSE / coverage (feasible SE, oracle SE)

| n | Adam | φ*-LP | φ̂-LP (MLP) | φ̂-LP (GB) |
|---|---|---|---|---|
| 500 | −5.69±0.16 / 1.16 / 5.80 / 0.16, 0.24 | −0.42±0.38 / 2.71 / 2.72 / 0.98, 0.98 | +0.72±0.36 / 2.56 / 2.63 / 0.92, 0.94 | −0.19±0.38 / 2.68 / 2.66 / 0.94, 0.96 |
| 1000 | +0.02±0.31 / 2.22 / 2.20 / 1.00, 1.00 | −0.09±0.35 / 2.46 / 2.44 / 0.94, 0.98 | +0.90±0.35 / 2.47 / 2.61 / 0.92, 0.96 | +0.21±0.35 / 2.46 / 2.45 / 0.96, 1.00 |
| 2000 | +0.82±0.36 / 2.56 / 2.66 / 0.96, 0.96 | +0.07±0.35 / 2.47 / 2.45 / 0.94, 0.96 | +0.69±0.36 / 2.52 / 2.59 / 0.98, 0.98 | +0.25±0.35 / 2.49 / 2.48 / 0.96, 0.96 |
| 4000 | +0.61±0.32 / 2.23 / 2.29 / 0.94, 0.94 | +0.14±0.31 / 2.21 / 2.20 / 0.94, 0.94 | +0.40±0.31 / 2.21 / 2.22 / 0.94, 0.94 | +0.23±0.32 / 2.28 / 2.27 / 0.92, 0.92 |

Paired shift in √n(θ̂−θ₀), φ̂-LP minus φ*-LP (mean ± SE): MLP +1.14±0.14, +0.99±0.11, +0.62±0.09, +0.26±0.04; GB +0.24±0.15, +0.30±0.07, +0.19±0.06, +0.09±0.05.
‖φ̂−φ*‖ (MLP / GB): 0.167/0.141, 0.149/0.114, 0.108/0.089, 0.073/0.068. ‖m̂−m₀‖: 0.273, 0.185, 0.149, 0.110.
√n‖m̂−m₀‖‖φ̂−φ*‖ (MLP / GB): 1.01/0.86, 0.88/0.67, 0.72/0.60, 0.51/0.47.

The cross-fitted φ̂ removes the n=500 Adam bias about as well as φ*. However, φ̂-LP carries a positive paired bias relative to φ*-LP that is highly significant for the MLP. It shrinks with n roughly as the product term does. The GB φ̂ is closer to φ* at every n and has a smaller bias. Coverage is 0.92 to 0.98 for every LP variant.

Sanity check: the recomputed S and φ*-score match the old CSV in 399/400 rows. The exception is n=2000 rep 38 (phistar_lp), with |ΔS| = 0.004 and |Δφ-score| = 0.010. That is one sign flip at an LP residual that is numerically zero. θ̂ is identical there.
