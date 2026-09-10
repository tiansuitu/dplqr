# DPLQR theta1 asymptotic-normality diagnostics

Primary target: theta1=1 at tau=0.5. Requested repetitions per case and n: **20**. Tables and figures use only completed replications; valid counts are recorded separately for estimates, SEs, studentization, and coverage.

| Case | n | Q | Bias | sqrt(n) bias | sqrt(n) SD | T mean | T SD | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 500 | 20 | -0.710 | -15.867 | 7.424 | -4.115 | 1.255 | 0.050 |
| 1 | 1000 | 20 | -0.301 | -9.505 | 5.670 | -2.913 | 1.786 | 0.300 |
| 1 | 2000 | 20 | -0.088 | -3.929 | 5.999 | -1.181 | 1.765 | 0.800 |
| 1 | 4000 | 20 | -0.033 | -2.092 | 3.589 | -0.611 | 1.030 | 0.900 |
| 2 | 500 | 20 | -0.608 | -13.588 | 6.439 | -3.577 | 1.809 | 0.200 |
| 2 | 1000 | 20 | -0.249 | -7.869 | 4.130 | -2.470 | 1.291 | 0.400 |
| 2 | 2000 | 20 | -0.111 | -4.984 | 3.150 | -1.489 | 0.946 | 0.600 |
| 2 | 4000 | 20 | -0.027 | -1.715 | 4.363 | -0.490 | 1.240 | 0.850 |
| 3 | 500 | 20 | -0.709 | -15.853 | 7.235 | -4.235 | 1.930 | 0.200 |
| 3 | 1000 | 20 | -0.243 | -7.680 | 5.771 | -2.434 | 1.907 | 0.300 |
| 3 | 2000 | 20 | -0.123 | -5.518 | 6.225 | -1.664 | 1.829 | 0.600 |
| 3 | 4000 | 20 | -0.062 | -3.934 | 4.641 | -1.165 | 1.352 | 0.750 |

| Case | n | Raw SD | Mean SE | RMSE | SE / SD | Coverage Wilson lower | Coverage Wilson upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 500 | 0.332 | 0.168 | 0.780 | 0.506 | 0.009 | 0.236 |
| 1 | 1000 | 0.179 | 0.104 | 0.348 | 0.581 | 0.145 | 0.519 |
| 1 | 2000 | 0.134 | 0.073 | 0.158 | 0.544 | 0.584 | 0.919 |
| 1 | 4000 | 0.057 | 0.055 | 0.064 | 0.963 | 0.699 | 0.972 |
| 2 | 500 | 0.288 | 0.175 | 0.669 | 0.609 | 0.081 | 0.416 |
| 2 | 1000 | 0.131 | 0.101 | 0.280 | 0.772 | 0.219 | 0.613 |
| 2 | 2000 | 0.070 | 0.075 | 0.131 | 1.071 | 0.387 | 0.781 |
| 2 | 4000 | 0.069 | 0.055 | 0.072 | 0.802 | 0.640 | 0.948 |
| 3 | 500 | 0.324 | 0.171 | 0.776 | 0.529 | 0.081 | 0.416 |
| 3 | 1000 | 0.182 | 0.097 | 0.301 | 0.529 | 0.145 | 0.519 |
| 3 | 2000 | 0.139 | 0.071 | 0.183 | 0.513 | 0.387 | 0.781 |
| 3 | 4000 | 0.073 | 0.054 | 0.095 | 0.729 | 0.531 | 0.888 |

## How to read these diagnostics

- **Consistency:** raw bias, empirical SD, and RMSE shrinking with n are relevant; a small raw bias alone is insufficient.
- **Root-n behavior:** sqrt(n) SD should approach a finite positive constant and sqrt(n) bias should approach zero for the centered limit being assessed. Shrinking raw bias with persistent nonzero sqrt(n) bias is compatible with consistency but problematic for the centered claim in Theorem 3.3 if it persists beyond Monte Carlo and optimization noise.
- **Asymptotic normality:** compare the whole distribution using QQ shape, skewness, excess kurtosis, and tails, in addition to location and scale. A nearly straight QQ curve with the wrong slope or intercept does not establish the desired N(0,1) studentized limit.
- **Variance and CI calibration:** mean estimated SE / empirical SD should approach one; T mean/SD should approach 0/1, its quantiles should approach (-1.959964, 0, 1.959964), and coverage should approach 0.95. Studentization jointly checks the estimator and its estimated SE: a nonstandard T distribution can result from estimator bias, nonnormality, or a miscalibrated variance estimate. Coverage alone cannot distinguish these causes.

The requested n is the total sample before the retained 80:20 training/validation split. The covariance uses n_train. summary.csv reports both sqrt(n) and sqrt(n_train) scalings; their fixed ratio changes the scale constant but not the studentized statistic or coverage.

Separate root-n-error QQ plots compare sqrt(n)(theta1_hat-1) with a normal reference fitted to its empirical mean and SD. These plots and the saved root-n-error skewness/kurtosis help separate estimator shape from estimated-SE/studentization effects. Their fitted lines intentionally absorb location and scale: agreement with those lines does not validate zero root-n bias or the paper's estimated variance. The studentized QQ plots instead use the fixed N(0,1) reference y=x.

## Observed pilot patterns

- Case 1, n=500 to 4000: raw bias -0.710 to -0.033; sqrt(n) bias -15.867 to -2.092; sqrt(n) SD 7.424 to 3.589; T mean/SD -4.115/1.255 to -0.611/1.030; coverage 0.050 to 0.900. These endpoint comparisons are descriptive; intermediate n values and Monte Carlo uncertainty must also be considered.
- Case 2, n=500 to 4000: raw bias -0.608 to -0.027; sqrt(n) bias -13.588 to -1.715; sqrt(n) SD 6.439 to 4.363; T mean/SD -3.577/1.809 to -0.490/1.240; coverage 0.200 to 0.850. These endpoint comparisons are descriptive; intermediate n values and Monte Carlo uncertainty must also be considered.
- Case 3, n=500 to 4000: raw bias -0.709 to -0.062; sqrt(n) bias -15.853 to -3.934; sqrt(n) SD 7.235 to 4.641; T mean/SD -4.235/1.930 to -1.165/1.352; coverage 0.200 to 0.750. These endpoint comparisons are descriptive; intermediate n values and Monte Carlo uncertainty must also be considered.

Valid studentized repetitions range from 20 to 20 per group. 
This is a small pilot. At Q=20, coverage moves in steps of 0.05 and its Monte Carlo SE is about 0.049 when true coverage is 0.95. Tail quantiles, skewness, and kurtosis are particularly noisy; extending to Q=200 is needed before placing weight on apparent trends.

Coverage uncertainty uses a 95% Wilson interval; the plug-in coverage MCSE is also saved, but can misleadingly be zero when observed coverage is 0 or 1. Bias MCSE is empirical SD/sqrt(Q). Across-n bias bars are approximate 95% Monte Carlo intervals, not confidence intervals for individual fitted coefficients. Empirical SD uses ddof=1. Skewness and excess kurtosis use SciPy's bias corrections and are unavailable below Q=3 and Q=4 respectively (or for a constant sample). Empirical quantiles use NumPy's linear interpolation; QQ plotting positions are (rank-0.5)/Q.

This finite-sample experiment neither verifies nor disproves Theorem 3.3. A fixed network/training schedule can introduce approximation or optimization error that does not vanish at the rate required by the theorem. Persistent anomalies call for checking those errors and the variance estimator as well as adding repetitions and larger n.

All detailed metrics are in summary.csv; quantiles.csv compares empirical and standard-normal quantiles. PNG figures show the distributions, QQ plots, and across-n patterns. No benchmark estimators or formal normality tests are used.
