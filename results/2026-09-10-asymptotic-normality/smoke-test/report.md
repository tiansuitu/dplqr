# DPLQR theta1 asymptotic-normality diagnostics

Primary target: theta1=1 at tau=0.5. Requested repetitions per case and n: **1**. Tables and figures use only completed replications; valid counts are recorded separately for estimates, SEs, studentization, and coverage.

| Case | n | Q | Bias | sqrt(n) bias | sqrt(n) SD | T mean | T SD | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 500 | 1 | -1.229 | -27.483 | NA | -6.545 | NA | 0.000 |
| 3 | 1000 | 1 | -0.049 | -1.562 | NA | -0.557 | NA | 1.000 |
| 3 | 2000 | 1 | -0.046 | -2.079 | NA | -0.629 | NA | 1.000 |
| 3 | 4000 | 1 | 0.067 | 4.250 | NA | 1.147 | NA | 1.000 |

| Case | n | Raw SD | Mean SE | RMSE | SE / SD | Coverage Wilson lower | Coverage Wilson upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 500 | NA | 0.188 | 1.229 | NA | 0.000 | 0.793 |
| 3 | 1000 | NA | 0.089 | 0.049 | NA | 0.207 | 1.000 |
| 3 | 2000 | NA | 0.074 | 0.046 | NA | 0.207 | 1.000 |
| 3 | 4000 | NA | 0.059 | 0.067 | NA | 0.207 | 1.000 |

## How to read these diagnostics

- **Consistency:** raw bias, empirical SD, and RMSE shrinking with n are relevant; a small raw bias alone is insufficient.
- **Root-n behavior:** sqrt(n) SD should approach a finite positive constant and sqrt(n) bias should approach zero for the centered limit being assessed. Shrinking raw bias with persistent nonzero sqrt(n) bias is compatible with consistency but problematic for the centered claim in Theorem 3.3 if it persists beyond Monte Carlo and optimization noise.
- **Asymptotic normality:** compare the whole distribution using QQ shape, skewness, excess kurtosis, and tails, in addition to location and scale. A nearly straight QQ curve with the wrong slope or intercept does not establish the desired N(0,1) studentized limit.
- **Variance and CI calibration:** mean estimated SE / empirical SD should approach one; T mean/SD should approach 0/1, its quantiles should approach (-1.959964, 0, 1.959964), and coverage should approach 0.95. Studentization jointly checks the estimator and its estimated SE: a nonstandard T distribution can result from estimator bias, nonnormality, or a miscalibrated variance estimate. Coverage alone cannot distinguish these causes.

The requested n is the total sample before the retained 80:20 training/validation split. The covariance uses n_train. summary.csv reports both sqrt(n) and sqrt(n_train) scalings; their fixed ratio changes the scale constant but not the studentized statistic or coverage.

Separate root-n-error QQ plots compare sqrt(n)(theta1_hat-1) with a normal reference fitted to its empirical mean and SD. These plots and the saved root-n-error skewness/kurtosis help separate estimator shape from estimated-SE/studentization effects. Their fitted lines intentionally absorb location and scale: agreement with those lines does not validate zero root-n bias or the paper's estimated variance. The studentized QQ plots instead use the fixed N(0,1) reference y=x.

## Observed pilot patterns

- Case 3, n=500 to 4000: raw bias -1.229 to 0.067; sqrt(n) bias -27.483 to 4.250; sqrt(n) SD NA to NA; T mean/SD -6.545/NA to 1.147/NA; coverage 0.000 to 1.000. These endpoint comparisons are descriptive; intermediate n values and Monte Carlo uncertainty must also be considered.

Valid studentized repetitions range from 1 to 1 per group. 
At least one group has fewer than two valid replications: this output is a smoke test only for those groups. Empirical SD and inferential Monte Carlo precision are unavailable there; single-point quantiles and observed coverage are not distributional evidence.
This is a small pilot. At Q=20, coverage moves in steps of 0.05 and its Monte Carlo SE is about 0.049 when true coverage is 0.95. Tail quantiles, skewness, and kurtosis are particularly noisy; extending to Q=200 is needed before placing weight on apparent trends.

Coverage uncertainty uses a 95% Wilson interval; the plug-in coverage MCSE is also saved, but can misleadingly be zero when observed coverage is 0 or 1. Bias MCSE is empirical SD/sqrt(Q). Across-n bias bars are approximate 95% Monte Carlo intervals, not confidence intervals for individual fitted coefficients. Empirical SD uses ddof=1. Skewness and excess kurtosis use SciPy's bias corrections and are unavailable below Q=3 and Q=4 respectively (or for a constant sample). Empirical quantiles use NumPy's linear interpolation; QQ plotting positions are (rank-0.5)/Q.

This finite-sample experiment neither verifies nor disproves Theorem 3.3. A fixed network/training schedule can introduce approximation or optimization error that does not vanish at the rate required by the theorem. Persistent anomalies call for checking those errors and the variance estimator as well as adding repetitions and larger n.

All detailed metrics are in summary.csv; quantiles.csv compares empirical and standard-normal quantiles. PNG figures show the distributions, QQ plots, and across-n patterns. No benchmark estimators or formal normality tests are used.
