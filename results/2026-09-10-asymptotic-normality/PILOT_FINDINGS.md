# Q=20 pilot findings

Completed all **240 DPLQR replications**: three cases, four total sample sizes
(500, 1000, 2000, 4000), 20 independent replications per cell, tau=0.5.
There were no failed replications. Both auxiliary projections were retained
for the full covariance, and the four initial smoke replications were reused.
No benchmark models were fitted. All figures and statistics use theta1=1 as
the primary target; theta2 is saved in the raw output.

**Errors shrink substantially, but this pilot does not establish the centered
normal approximation or well-calibrated 95% inference.** Case 3 has the clearest
remaining centering and SE-calibration problems.

| At n=4000 (n_train=3200) | Case 1 | Case 2 | Case 3 |
| --- | ---: | ---: | ---: |
| Raw bias | -0.033 | -0.027 | -0.062 |
| sqrt(n) bias | -2.092 | -1.715 | -3.934 |
| sqrt(n) empirical SD | 3.589 | 4.363 | 4.641 |
| Mean estimated SE / empirical SD | 0.963 | 0.802 | 0.729 |
| Studentized mean | -0.611 | -0.490 | -1.165 |
| Studentized SD | 1.030 | 1.240 | 1.352 |
| Empirical 95% CI coverage | 90% | 85% | 75% |
| 95% Wilson interval for coverage | 69.9%-97.2% | 64.0%-94.8% | 53.1%-88.8% |

## Consistency and root-n behavior

Raw bias, SD and RMSE decrease from n=500 to 4000 in every case. RMSE falls
from 0.780 to 0.064 in Case 1, 0.669 to 0.072 in Case 2, and 0.776 to 0.095
in Case 3. These patterns are compatible with ordinary consistency, without
proving the limiting claim.

Scaled bias also moves toward zero, so this pilot should **not** be described
as showing an established nonzero asymptotic bias. Nevertheless, it remains
negative at n=4000, especially in Case 3. Approximate 95% Monte Carlo intervals
for sqrt(n) bias at n=4000 are [-3.665, -0.519], [-3.627, 0.197], and
[-5.967, -1.900] for Cases 1-3. These are descriptive Monte Carlo intervals,
not simultaneous tests of the theorem.

The scaled SD sequences are (7.424, 5.670, 5.999, 3.589),
(6.439, 4.130, 3.150, 4.363), and (7.235, 5.771, 6.225, 4.641).
They do not yet exhibit an unambiguous stable plateau. Persistent nonzero
scaled bias, if it remains at larger n, would be compatible with consistency
but problematic for the centered limit in Theorem 3.3. More replications can
clarify these finite-n patterns; they cannot alone establish an asymptotic rate.

## Normal shape versus valid inference

The separate root-n-error QQ plots use normal lines fitted to each empirical
mean and SD. At n=4000 the estimator shape is closer to Gaussian than the
studentized N(0,1) comparison might suggest, particularly in Case 1. For
example, Case 1's studentized skewness/excess kurtosis are 0.271/-0.344 and
its SD is 1.030, but its mean is still -0.611. A Gaussian-looking distribution
with a negative mean does not deliver the centered inference sought here.

Case 3's studentized 2.5%, 50%, and 97.5% quantiles at n=4000 are
(-3.101, -1.329, 1.390), versus (-1.960, 0, 1.960) for N(0,1).
Its mean SE is only about 73% of the empirical coefficient SD. Its 75%
coverage therefore reflects evidence of both remaining negative centering
and insufficient estimated spread; coverage alone would not separate them.
These observations concern the combined fitted-estimator/inference procedure,
and do not identify a single cause or prove that the covariance formula itself
is wrong.

Q=20 makes the coverage grid coarse (5 percentage points per replication),
and its Monte Carlo SE near nominal coverage is about 4.9 percentage points.
Skewness, kurtosis, and tail quantiles are especially noisy. The exact retained
fixed architecture, finite optimization and validation early stopping do not
guarantee the growing-class and optimization assumptions used in the theorem.
The results diagnose this implementation under Simulation I, not a proof or
refutation of Theorem 3.3.

## Artifacts, validation and runtime

- [Full 12-cell report](pilot/report.md), [summary CSV](pilot/summary.csv),
  [raw replications](pilot/raw_replications.csv), and [quantile comparisons](pilot/quantiles.csv).
- Fifteen diagnostic PNGs are in `pilot/figures/`, including five figure types
  for each case. [Case 3 across-n figure](pilot/figures/case_3_across_n.png)
  gives a compact view of the remaining inference issues.
- [Parity verification](verification.json) matched original DPLQR estimates,
  both SEs, density and epochs bit for bit. Restored-stage testing prohibited
  all fitting calls and reproduced the scientific outputs exactly.
- [Output validation](pilot/validation.json) checked all 240 rows, 12 groups,
  36 quantile comparisons, all 720 network-checkpoint hashes, and unchanged
  reuse of all four smoke rows. Previous simulation files remain unchanged.

The four-fit smoke projected 7.3 minutes for Q=20 and 72.9 minutes for Q=200;
its Case 3 timing was optimistic. The completed pilot took **13.0 minutes**.
The [updated measured projection](pilot/runtime_projection.md) is about
**129.5 minutes total for Q=200**, or **116.6 additional minutes** when
extending these Q=20 checkpoints. Q=200 has **not** been run. Commands and
the retained settings are in [README.md](README.md).
