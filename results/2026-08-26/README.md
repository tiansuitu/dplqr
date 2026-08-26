# Concrete-data replication results - 26 August 2026

This folder preserves two successful executions of the public concrete-data
demonstration from Zhong and Wang (2024).

## Folder contents

- `initial-user-run/`: CSV outputs from the first successful interactive run
  and a screenshot confirming completion in RStudio.
- `codex-rerun/`: an independently executed notebook, its exported split and
  result CSVs, the PLAQR transcript, and `concrete-comparison.png`.
- `metrics-comparison.csv`: exact run metrics alongside the published values
  from Supplementary Figures 5 and 6.

The automated notebook contains zero error outputs. The automated PLAQR run
completed all 49 quantile levels. Its repeated warning concerned test-set fly
ash values outside the B-spline boundary knots learned from the random training
split; predictions and result-file creation still completed.

## Interpretation

Both runs reproduce the supplement's prediction ranking among the available
interpretable methods: DPLQR has the lowest average check loss, followed by
PLAQR and then LQR. Exact values vary because the repository does not set
random seeds.

Supplementary Table 9 is a different simulation experiment about confidence-
interval coverage and is not directly reproducible from this concrete-data
demo. The direct empirical comparators are Supplementary Figures 5 and 6.

See the 26 August 2026 section of `../../REPLICATION_SETUP.md` for the detailed
comparison, methodological compatibility note, and limitations.
