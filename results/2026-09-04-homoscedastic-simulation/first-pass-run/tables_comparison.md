# First-pass results compared with Zhong and Wang (2024)

Executed 2026-09-08; total notebook execution time: 28.8 seconds.

**Scope: n = 1000, quantile = 0.50, Cases 1-3.** The tables below show the corresponding slice of published Tables 1-3 (paper p. 609). The first pass completed successfully, but its estimates are preliminary.

| Setting | This first pass | Published simulation |
|:--|:--|:--|
| Repetitions per setting | 2 | 200 |
| Total sample size | 1000 | 1000 and 2000 |
| Training / validation split | 800 / 200 | 80% / 20% |
| Test sample size | 1000 | 5000 |
| Quantiles | 0.50 | 0.25, 0.50, 0.75 |
| DPLQR tuning | Fixed profile, at most 100 epochs | Validation grid, including 200/500 epochs |

The fixed profile uses the original depth argument 2, width 32, batch size 128, patience 15, learning rate 0.005, and seed 20260904. All three methods use the same exported samples. Published values are comparison targets, never substituted for simulated estimates.

## Table 1. Bias and standard deviation of theta1

| Case | Results | LQR | PLAQR | DPLQR |
|:--|:--|--:|--:|--:|
| Case 1 (linear) | First pass (2 repetitions) | -0.0286 (0.0227) | -0.0043 (0.0413) | -0.3378 (0.1832) |
| Case 1 (linear) | Paper (200 repetitions) | 0.0219 (0.1116) | 0.0413 (0.1199) | 0.0636 (0.1235) |
| Case 2 (additive) | First pass (2 repetitions) | 0.0091 (0.0786) | 0.0064 (0.0677) | -0.3390 (0.1134) |
| Case 2 (additive) | Paper (200 repetitions) | -0.0611 (0.1158) | -0.0124 (0.1008) | 0.0338 (0.1116) |
| Case 3 (deep) | First pass (2 repetitions) | 0.1980 (0.0493) | 0.1836 (0.1126) | 0.0645 (0.1611) |
| Case 3 (deep) | Paper (200 repetitions) | 0.1068 (0.1444) | -0.0902 (0.1337) | 0.0403 (0.0998) |

Parentheses contain the sample SD across repetitions. Bias is mean(theta1 estimate) minus 1.

[Download the first-pass Table 1 CSV](table1_bias_sd.csv).

## Table 2. Empirical coverage of the 95% interval for theta1

| Case | Results | LQR | PLAQR | DPLQR |
|:--|:--|--:|--:|--:|
| Case 1 (linear) | First pass (2 repetitions) | 1.000 | 1.000 | 0.500 |
| Case 1 (linear) | Paper (200 repetitions) | 0.975 | 0.915 | 0.905 |
| Case 2 (additive) | First pass (2 repetitions) | 1.000 | 1.000 | 0.000 |
| Case 2 (additive) | Paper (200 repetitions) | 0.960 | 0.955 | 0.945 |
| Case 3 (deep) | First pass (2 repetitions) | 1.000 | 1.000 | 0.500 |
| Case 3 (deep) | Paper (200 repetitions) | 0.885 | 0.875 | 0.930 |

With two repetitions, first-pass coverage can only be 0.000, 0.500, or 1.000. It cannot estimate coverage near 95% reliably.

[Download the first-pass Table 2 CSV](table2_coverage.csv).

## Table 3. Relative mean squared error of the nuisance estimate

| Case | Results | LQR | PLAQR | DPLQR |
|:--|:--|--:|--:|--:|
| Case 1 (linear) | First pass (2 repetitions) | 0.0008 | 0.0025 | 0.0245 |
| Case 1 (linear) | Paper (200 repetitions) | 0.0064 | 0.0070 | 0.0081 |
| Case 2 (additive) | First pass (2 repetitions) | 0.0457 | 0.0044 | 0.0138 |
| Case 2 (additive) | Paper (200 repetitions) | 0.0062 | 0.0053 | 0.0059 |
| Case 3 (deep) | First pass (2 repetitions) | 0.0502 | 0.0538 | 0.0805 |
| Case 3 (deep) | Paper (200 repetitions) | 0.0679 | 0.0539 | 0.0209 |

Relative MSE follows Eq. (16), without a square root. The true nuisance function includes the Student-t quantile shift.

[Download the first-pass Table 3 CSV](table3_rmse.csv).

## Interpretation

These results are not yet a close numerical reproduction of the paper. In particular, DPLQR theta1 bias is -0.3378 and -0.3390 in Cases 1 and 2, compared with 0.0636 and 0.0338 in the paper. In Case 3, DPLQR relative MSE is 0.0805 versus 0.0209. The small number of repetitions and reduced fixed training profile do not support firm conclusions about method rankings.

DPLQR main fits ran for 70, 24, 77, 91, 77, and 100 epochs across the six datasets. Five fits stopped before the ceiling; the final fit reached it. A full comparison still requires the larger Monte Carlo design and the reported tuning search, while accounting for the source-code ambiguities documented in [the setup README](../README.md).

## Files and validation

- [Executed notebook](executed_first_pass.ipynb)
- [Combined replicate-level results](raw_results_combined.csv)
- [Summary for all nine case/method combinations](simulation_summary_long.csv)
- [Python settings and completion status](run_config.json)
- [R settings](run_config.R)

Validation confirmed 12 Python result rows and 6 PLAQR result rows, no missing or duplicate combinations, finite summary values, coverage consistent with saved interval bounds, and independently recomputed bias, SD, coverage, and relative MSE. All notebook cells completed without errors.
