# Continued-training diagnostic results

These results follow one initialized DPLQR model continuously on one training dataset per replicate. Training uses the fixed epoch ceiling, with no early stopping, no best-validation weight restoration, and no checkpoint selection. Validation, test and evaluation data do not affect optimization.

Completed data: **Case 1: 10 repetitions; Case 2: 10 repetitions; Case 3: 10 repetitions**; n = 1000; tau = 0.5; training/validation sizes = 800/200; independent test size = 10000; independent evaluation size = 10000.

Recorded epochs: 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000. There are 450 checkpoint rows and 30 distinct case–replicate trajectories.

**This is a reduced run. The intended design is 100 repetitions per case; the current run does not complete that design.** Small-run differences and Monte Carlo uncertainty intervals are exploratory.

## Paired endpoint comparisons

The following comparisons use epoch **100 → 1000** on the same replicate. Changes are end minus start. The coefficient diagnostic is the Euclidean error `||theta_hat - theta_true||_2`; it is not the absolute value of average bias.

### Coefficient L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.3111 | 0.1669 | -0.1441 | 0.0685 | [-0.2785, -0.0098] |
| Case 2 | 0.3319 | 0.2005 | -0.1314 | 0.0230 | [-0.1765, -0.0863] |
| Case 3 | 0.2943 | 0.1864 | -0.1079 | 0.0550 | [-0.2158, 0.0000] |

### Training check loss

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.4991 | 0.2523 | -0.2469 | 0.0103 | [-0.2670, -0.2268] |
| Case 2 | 0.4824 | 0.2537 | -0.2287 | 0.0088 | [-0.2460, -0.2113] |
| Case 3 | 0.4423 | 0.1822 | -0.2602 | 0.0046 | [-0.2693, -0.2511] |

### Independent test check loss

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.6049 | 0.7955 | 0.1906 | 0.0083 | [0.1744, 0.2068] |
| Case 2 | 0.6160 | 0.7836 | 0.1676 | 0.0105 | [0.1471, 0.1881] |
| Case 3 | 0.6325 | 0.8458 | 0.2133 | 0.0067 | [0.2001, 0.2264] |

### Nuisance L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.6444 | 1.4009 | 0.7564 | 0.0624 | [0.6341, 0.8788] |
| Case 2 | 0.6776 | 1.3729 | 0.6953 | 0.0452 | [0.6068, 0.7838] |
| Case 3 | 0.7360 | 1.5822 | 0.8462 | 0.0244 | [0.7983, 0.8941] |

### Full conditional-quantile L2 error

| Case | Mean at 100 | Mean at 1000 | Mean paired change | MCSE of change | Approx. MC interval |
| --- | --- | --- | --- | --- | --- |
| Case 1 | 0.5430 | 1.3939 | 0.8509 | 0.0367 | [0.7791, 0.9228] |
| Case 2 | 0.6374 | 1.3626 | 0.7252 | 0.0423 | [0.6422, 0.8082] |
| Case 3 | 0.7254 | 1.5737 | 0.8483 | 0.0232 | [0.8029, 0.8937] |

The intervals are mean paired change ± 1.96 × its Monte Carlo SE, treating a replicate as the independent unit. They are pointwise normal approximations, not simultaneous intervals or tests adjusted for looking across cases, coefficients and epochs. In small runs they can be unreliable. Evaluation-sample approximation error is also part of each replicate's measurement.

### Counts of paired directional changes

| Case | Q | Train loss ↓ | Theta L2 ↑ | q L2 ↓ | Test loss ↑ | Train ↓, theta ↑, q ↓ | Train ↓, theta ↑, test ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 10 | 10 | 2 | 0 | 10 | 0 | 2 |
| Case 2 | 10 | 10 | 0 | 0 | 10 | 0 | 0 |
| Case 3 | 10 | 10 | 3 | 0 | 10 | 0 | 3 |

These are literal strict directional comparisons of finite-sample estimates, without a post hoc threshold for 'roughly stable'. Counts alone do not establish a population effect. A lower q error with a worse coefficient estimate is compatible with compensation by the nuisance network; a negative cross-error moment directly describes cancellation on the evaluation sample. Neither observation alone establishes a causal or asymptotic mechanism. Worsening test loss alongside lower training loss is evidence compatible with ordinary prediction overfitting, which can coexist with compensation.

## Coefficient bias and RMSE

| Case | Component | Epoch | Mean estimate | Truth | Bias | SD | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | theta_1 | 100 | 0.9611 | 1.0000 | -0.0389 | 0.1003 | 0.1028 |
| 1 | theta_2 | 100 | -0.7059 | -1.0000 | 0.2941 | 0.1548 | 0.3287 |
| 1 | theta_1 | 1000 | 1.0464 | 1.0000 | 0.0464 | 0.0884 | 0.0959 |
| 1 | theta_2 | 1000 | -0.9677 | -1.0000 | 0.0323 | 0.1755 | 0.1696 |
| 2 | theta_1 | 100 | 0.8676 | 1.0000 | -0.1324 | 0.1144 | 0.1712 |
| 2 | theta_2 | 100 | -0.7365 | -1.0000 | 0.2635 | 0.1686 | 0.3082 |
| 2 | theta_1 | 1000 | 0.9170 | 1.0000 | -0.0830 | 0.0891 | 0.1185 |
| 2 | theta_2 | 1000 | -0.8781 | -1.0000 | 0.1219 | 0.1629 | 0.1969 |
| 3 | theta_1 | 100 | 0.9278 | 1.0000 | -0.0722 | 0.1456 | 0.1558 |
| 3 | theta_2 | 100 | -0.7507 | -1.0000 | 0.2493 | 0.1368 | 0.2811 |
| 3 | theta_1 | 1000 | 0.9939 | 1.0000 | -0.0061 | 0.1696 | 0.1610 |
| 3 | theta_2 | 1000 | -0.9509 | -1.0000 | 0.0491 | 0.1288 | 0.1317 |

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

The decomposition is `q_mse = linear_error_mse + nonlinear_error_mse + 2 * cross_error_moment`. The raw `cancellation_fraction` is `-2 * cross_error_moment / (linear_error_mse + nonlinear_error_mse)` (zero if the denominator is zero); a positive value indicates error cancellation. The maximum absolute recorded decomposition residual is 7.11e-15.

Per-replicate data, initialization, shuffle/training, test and evaluation seeds are retained in the raw CSV. Optimizer-step counts, model-instance identifiers and fit-call counts are also retained to audit the continuous paths. Consult the run configuration and execution validation record for the architecture, optimizer, ceiling and checks actually used.

- [Raw checkpoint trajectories](raw_epoch_trajectory.csv)
- [Tidy coefficient summaries](monte_carlo_summary.csv)
- [Tidy diagnostic summaries](diagnostic_summary.csv)
- [Paired endpoint changes and Monte Carlo uncertainty](paired_endpoint_changes.csv)
- [Paired directional counts](endpoint_pattern_counts.csv)
- [Run configuration](run_config.json)
