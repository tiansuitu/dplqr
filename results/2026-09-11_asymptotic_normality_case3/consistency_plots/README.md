# Case 3: coefficient stability across training epochs

These plots use only the saved DPLQR point estimates in `run/raw_replications.csv`. True values come from `run/run_config.json` and are checked against the saved truth columns. No models were fitted. The sample size n is the total generated size; 80% trains the model.

## How to read the plots

- **Mean versus epochs:** distance from the dashed true-value line measures signed mean bias.
- **Absolute bias:** absolute distance of the Monte Carlo mean from the truth; lower is better.
- **RMSE:** square root of the average squared coefficient error across replications; includes both bias and spread.
- **Boxplots:** center lines are medians, boxes span the middle 50%, whiskers extend to the most extreme observations within 1.5 interquartile ranges, and dots show more extreme observations. The dashed line is the true value.

Monte Carlo SD uses divisor Q-1; RMSE averages squared errors with divisor Q. Plots use all saved replications, with actual Q shown. Epoch settings reuse the same seeded datasets.

## What changes between the first and last saved epochs?

### theta_1: true value 1

- n=500, 200 to 1000 epochs: mean 0.76710 to 0.84868; absolute bias decreases (0.23290 to 0.15132); SD 0.18111 to 0.23765; RMSE 0.29475 to 0.28123.
- n=1000, 200 to 1000 epochs: mean 0.90750 to 0.95073; absolute bias decreases (0.09250 to 0.04927); SD 0.13056 to 0.17281; RMSE 0.15974 to 0.17928.
- n=2000, 200 to 1000 epochs: mean 0.98359 to 0.98749; absolute bias decreases (0.01641 to 0.01251); SD 0.10203 to 0.12137; RMSE 0.10309 to 0.12171.
- n=4000, 200 to 1000 epochs: mean 0.98720 to 0.98481; absolute bias increases (0.01280 to 0.01519); SD 0.06416 to 0.07268; RMSE 0.06527 to 0.07407.

| n | epochs | Q | mean_theta_hat | bias | absolute_bias | MC_SD | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 200 | 200 | 0.76710 | -0.23290 | 0.23290 | 0.18111 | 0.29475 |
| 500 | 500 | 200 | 0.82786 | -0.17214 | 0.17214 | 0.20202 | 0.26503 |
| 500 | 1000 | 200 | 0.84868 | -0.15132 | 0.15132 | 0.23765 | 0.28123 |
| 1000 | 200 | 200 | 0.90750 | -0.09250 | 0.09250 | 0.13056 | 0.15974 |
| 1000 | 500 | 200 | 0.93383 | -0.06617 | 0.06617 | 0.14875 | 0.16247 |
| 1000 | 1000 | 200 | 0.95073 | -0.04927 | 0.04927 | 0.17281 | 0.17928 |
| 2000 | 200 | 200 | 0.98359 | -0.01641 | 0.01641 | 0.10203 | 0.10309 |
| 2000 | 500 | 200 | 0.98721 | -0.01279 | 0.01279 | 0.11268 | 0.11312 |
| 2000 | 1000 | 200 | 0.98749 | -0.01251 | 0.01251 | 0.12137 | 0.12171 |
| 4000 | 200 | 200 | 0.98720 | -0.01280 | 0.01280 | 0.06416 | 0.06527 |
| 4000 | 500 | 200 | 0.98546 | -0.01454 | 0.01454 | 0.06896 | 0.07031 |
| 4000 | 1000 | 200 | 0.98481 | -0.01519 | 0.01519 | 0.07268 | 0.07407 |

### theta_2: true value -1

- n=500, 200 to 1000 epochs: mean -0.67535 to -0.81709; absolute bias decreases (0.32465 to 0.18291); SD 0.18545 to 0.21776; RMSE 0.37365 to 0.28397.
- n=1000, 200 to 1000 epochs: mean -0.84360 to -0.91600; absolute bias decreases (0.15640 to 0.08400); SD 0.13080 to 0.15793; RMSE 0.20368 to 0.17853.
- n=2000, 200 to 1000 epochs: mean -0.95028 to -0.98097; absolute bias decreases (0.04972 to 0.01903); SD 0.08487 to 0.09930; RMSE 0.09818 to 0.10086.
- n=4000, 200 to 1000 epochs: mean -0.98739 to -1.00425; absolute bias decreases (0.01261 to 0.00425); SD 0.05691 to 0.06412; RMSE 0.05816 to 0.06410.

| n | epochs | Q | mean_theta_hat | bias | absolute_bias | MC_SD | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 200 | 200 | -0.67535 | 0.32465 | 0.32465 | 0.18545 | 0.37365 |
| 500 | 500 | 200 | -0.78448 | 0.21552 | 0.21552 | 0.19757 | 0.29204 |
| 500 | 1000 | 200 | -0.81709 | 0.18291 | 0.18291 | 0.21776 | 0.28397 |
| 1000 | 200 | 200 | -0.84360 | 0.15640 | 0.15640 | 0.13080 | 0.20368 |
| 1000 | 500 | 200 | -0.89342 | 0.10658 | 0.10658 | 0.14041 | 0.17600 |
| 1000 | 1000 | 200 | -0.91600 | 0.08400 | 0.08400 | 0.15793 | 0.17853 |
| 2000 | 200 | 200 | -0.95028 | 0.04972 | 0.04972 | 0.08487 | 0.09818 |
| 2000 | 500 | 200 | -0.96793 | 0.03207 | 0.03207 | 0.09190 | 0.09712 |
| 2000 | 1000 | 200 | -0.98097 | 0.01903 | 0.01903 | 0.09930 | 0.10086 |
| 4000 | 200 | 200 | -0.98739 | 0.01261 | 0.01261 | 0.05691 | 0.05816 |
| 4000 | 500 | 200 | -0.99919 | 0.00081 | 0.00081 | 0.06016 | 0.06001 |
| 4000 | 1000 | 200 | -1.00425 | -0.00425 | 0.00425 | 0.06412 | 0.06410 |

## Interpreting sample size and consistency

- theta_1, 200 epochs: absolute bias decreases at every saved increase in n; RMSE decreases at every saved increase in n.
- theta_1, 500 epochs: absolute bias does not decrease at every saved increase in n; RMSE decreases at every saved increase in n.
- theta_1, 1000 epochs: absolute bias does not decrease at every saved increase in n; RMSE decreases at every saved increase in n.
- theta_2, 200 epochs: absolute bias decreases at every saved increase in n; RMSE decreases at every saved increase in n.
- theta_2, 500 epochs: absolute bias decreases at every saved increase in n; RMSE decreases at every saved increase in n.
- theta_2, 1000 epochs: absolute bias decreases at every saved increase in n; RMSE decreases at every saved increase in n.

Closeness to the true coefficient **as epochs increase at fixed n** is a training-stability diagnostic. Formal asymptotic consistency concerns convergence to the true coefficient **as n tends to infinity**, under stated assumptions and a specified estimator/training sequence. These finite-sample plots do not establish that property. Small changes in empirical bias also have Monte Carlo uncertainty; they need not reflect a systematic population drift.

## Reproduce

From the repository root, using its Python environment:

```powershell
& ./.venv/Scripts/python.exe results/2026-09-11_asymptotic_normality_case3/consistency_plots/plot_coefficient_stability.py
```

Requires NumPy, pandas and Matplotlib only. Existing outputs are preserved; repeat runs write to a fresh `rerun_*` subfolder.
