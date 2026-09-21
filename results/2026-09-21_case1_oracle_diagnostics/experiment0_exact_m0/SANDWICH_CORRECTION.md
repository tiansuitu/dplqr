# Sandwich correction (2026-09-21)

Theorist review: when \(m_0\) is known, Experiment 0 is ordinary linear quantile regression of \(Y-m_0(Z)\) on \(X\). The oracle sandwich and influence-function score must use **raw \(X\)**, not \(	ilde X=X-arphi^*(Z)\).

Residualizing is the efficient-score construction for the **semiparametric** case where \(m\) is estimated. Using \(	ilde X\) with known \(m_0\) inflates \(	heta_2\) SEs (under-dispersed \(T_2\), coverage near 100%).

This folder's `replication_results.csv` / `summary.csv` / figures were recomputed with the linear-QR sandwich on raw \(X\). Previous residualized diagnostics were backed up as `run/replication_results_residualized_Xt_BACKUP.csv`.

Experiment 2 (estimated \(m\)) should continue to use \(	ilde X\).
