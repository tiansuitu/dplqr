# Continued-training diagnostic results

These results follow one initialized DPLQR model continuously on one training dataset per replicate. Training uses the fixed epoch ceiling, with no early stopping, no best-validation weight restoration, and no checkpoint selection. Validation, test and evaluation data do not affect optimization.

Completed data: **Case 1: 2 repetitions; Case 2: 2 repetitions; Case 3: 2 repetitions**; n = 1000; tau = 0.5; training/validation sizes = 800/200; independent test size = 10000; independent evaluation size = 10000.

Recorded epochs: 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000. There are 90 checkpoint rows and 6 distinct case–replicate trajectories.

**This is a reduced run. The intended design is 100 repetitions per case; the current run does not complete that design.** Small-run differences and Monte Carlo uncertainty intervals are exploratory.

## Paired endpoint comparisons

The following comparisons use epoch **100 → 1000** on the same replicate. Changes are end minus start. The coefficient diagnostic is the Euclidean error `||theta_hat - theta_true||_2`; it is not the absolute value of average bias.

### Coefficient L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.3847 | 0.1476 | -0.2372 | 0.1191 | [-0.4705, -0.0038] |
| Case 2 | 0.3816 | 0.2026 | -0.1790 | 0.0731 | [-0.3224, -0.0357] |
| Case 3 | 0.2645 | 0.2149 | -0.0496 | 0.0885 | [-0.2231, 0.1238] |

### Training check loss

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.5057 | 0.2716 | -0.2341 | 0.0223 | [-0.2778, -0.1904] |
| Case 2 | 0.4669 | 0.2509 | -0.2161 | 0.0129 | [-0.2413, -0.1908] |
| Case 3 | 0.4406 | 0.1789 | -0.2617 | 0.0120 | [-0.2853, -0.2381] |

### Independent test check loss

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.6054 | 0.8087 | 0.2033 | 0.0058 | [0.1919, 0.2146] |
| Case 2 | 0.6155 | 0.7879 | 0.1725 | 0.0112 | [0.1505, 0.1945] |
| Case 3 | 0.6306 | 0.8380 | 0.2074 | 0.0077 | [0.1924, 0.2225] |

### Nuisance L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.4781 | 1.4407 | 0.9626 | 0.0405 | [0.8832, 1.0419] |
| Case 2 | 0.6215 | 1.4045 | 0.7830 | 0.0116 | [0.7602, 0.8058] |
| Case 3 | 0.7422 | 1.5814 | 0.8393 | 0.0406 | [0.7597, 0.9188] |

### Full conditional-quantile L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.5433 | 1.4442 | 0.9009 | 0.0150 | [0.8715, 0.9303] |
| Case 2 | 0.6302 | 1.4095 | 0.7793 | 0.0402 | [0.7005, 0.8582] |
| Case 3 | 0.7304 | 1.5624 | 0.8320 | 0.0392 | [0.7552, 0.9088] |

The intervals are mean paired change ± 1.96 × its Monte Carlo SE, treating a replicate as the independent unit. They are pointwise normal approximations, not simultaneous intervals or tests adjusted for looking across cases, coefficients and epochs. In small runs they can be unreliable. Evaluation-sample approximation error is also part of each replicate's measurement.

### Counts of paired directional changes

| Case | Q | Train loss ↓ | Theta L2 ↑ | q L2 ↓ | Test loss ↑ | Train ↓, theta ↑, q ↓ | Train ↓, theta ↑, test ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| Case 2 | 2 | 2 | 0 | 0 | 2 | 0 | 0 |
| Case 3 | 2 | 2 | 1 | 0 | 2 | 0 | 1 |

These are literal strict directional comparisons of finite-sample estimates, without a post hoc threshold for 'roughly stable'. Counts alone do not establish a population effect. A lower q error with a worse coefficient estimate is compatible with compensation by the nuisance network; a negative cross-error moment directly describes cancellation on the evaluation sample. Neither observation alone establishes a causal or asymptotic mechanism. Worsening test loss alongside lower training loss is evidence compatible with ordinary prediction overfitting, which can coexist with compensation.

## Coefficient bias and RMSE

| Case | Component | Epoch | Mean estimate | Truth | Bias | SD | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | theta_1 | 100 | 0.9082 | 1.0000 | -0.0918 | 0.1586 | 0.1449 |
| 1 | theta_2 | 100 | -0.6357 | -1.0000 | 0.3643 | 0.1943 | 0.3894 |
| 1 | theta_1 | 1000 | 0.9349 | 1.0000 | -0.0651 | 0.1253 | 0.1100 |
| 1 | theta_2 | 1000 | -0.8946 | -1.0000 | 0.1054 | 0.0026 | 0.1054 |
| 2 | theta_1 | 100 | 0.7750 | 1.0000 | -0.2250 | 0.0184 | 0.2254 |
| 2 | theta_2 | 100 | -0.6919 | -1.0000 | 0.3081 | 0.0490 | 0.3100 |
| 2 | theta_1 | 1000 | 0.8417 | 1.0000 | -0.1583 | 0.0315 | 0.1598 |
| 2 | theta_2 | 1000 | -0.8741 | -1.0000 | 0.1259 | 0.0457 | 0.1300 |
| 3 | theta_1 | 100 | 1.0430 | 1.0000 | 0.0430 | 0.1586 | 0.1201 |
| 3 | theta_2 | 100 | -0.7688 | -1.0000 | 0.2312 | 0.0767 | 0.2375 |
| 3 | theta_1 | 1000 | 1.0168 | 1.0000 | 0.0168 | 0.2429 | 0.1726 |
| 3 | theta_2 | 1000 | -0.8938 | -1.0000 | 0.1062 | 0.1314 | 0.1411 |

Bias is the mean signed coefficient error. SD uses the sample standard deviation across replicates (ddof = 1). RMSE is sqrt(mean squared coefficient error). Small bias can coexist with large SD or RMSE because positive and negative errors cancel across replicates.

## Figures

A. Mean signed coefficient error; zero denotes no bias.

![A. Mean signed coefficient error; zero denotes no bias.](figures/A_theta_bias.png)

[PDF version](figures/A_theta_bias.pdf)

B. Component RMSE and mean vector coefficient L2 error.

![B. Component RMSE and mean vector coefficient L2 error.](figures/B_theta_error.png)

[PDF version](figures/B_theta_error.pdf)

C. Training and genuinely independent test check loss.

![C. Training and genuinely independent test check loss.](figures/C_check_loss.png)

[PDF version](figures/C_check_loss.pdf)

D. Nuisance and full quantile L2 error on fixed evaluation samples.

![D. Nuisance and full quantile L2 error on fixed evaluation samples.](figures/D_nuisance_and_quantile_error.png)

[PDF version](figures/D_nuisance_and_quantile_error.pdf)

E. Squared-error decomposition and cancellation between linear and nonlinear errors.

![E. Squared-error decomposition and cancellation between linear and nonlinear errors.](figures/E_error_decomposition.png)

[PDF version](figures/E_error_decomposition.pdf)

F. Raw coefficient paths for pre-specified replicates 1–3, where available.

![F. Raw coefficient paths for pre-specified replicates 1–3, where available.](figures/F_fixed_replicate_paths.png)

[PDF version](figures/F_fixed_replicate_paths.pdf)

## Definitions and reproducibility

The nuisance and full-quantile L2 diagnostics are square roots of empirical mean squared errors against the known true functions on a fixed independent evaluation sample within each replicate. They are **not** the relative mean squared error used in the older paper-table replication output. The nuisance truth includes the error-distribution quantile shift when tau differs from 0.5. All reported network values are evaluated directly without post hoc coefficient clipping or nuisance recalibration.

The decomposition is `q_mse = linear_error_mse + nonlinear_error_mse + 2 * cross_error_moment`. The raw `cancellation_fraction` is `-2 * cross_error_moment / (linear_error_mse + nonlinear_error_mse)` (zero if the denominator is zero); a positive value indicates error cancellation. The maximum absolute recorded decomposition residual is 3.55e-15.

Per-replicate data, initialization, shuffle/training, test and evaluation seeds are retained in the raw CSV. Optimizer-step counts, model-instance identifiers and fit-call counts are also retained to audit the continuous paths. Consult the run configuration and execution validation record for the architecture, optimizer, ceiling and checks actually used.

- [Raw checkpoint trajectories](raw_epoch_trajectory.csv)
- [Tidy coefficient summaries](monte_carlo_summary.csv)
- [Tidy diagnostic summaries](diagnostic_summary.csv)
- [Paired endpoint changes and Monte Carlo uncertainty](paired_endpoint_changes.csv)
- [Paired directional counts](endpoint_pattern_counts.csv)
- [Run configuration](run_config.json)
