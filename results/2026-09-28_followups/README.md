# 2026-09-28 follow-ups (Zhong & Wang 2024, Thm 3.3 checks)

All files here are new. Existing scripts and results are only read (imported via `sys.path`), never modified.

| Subfolder | What it contains |
|---|---|
| `class_check/` | Task A. How far the saved fits sit outside ZW's sparse, bounded (Schmidt-Hieber) network class. Covers toy `run_expansion_v2/main_26a7078945fc` (baseline and polished, 4 n × 50 reps) and Case 3 `run/` (epochs = 1000, 4 n × 50 reps). Reports weight magnitudes overall and per layer, parameter counts, the m̂ range vs m₀ on training Z and on 100k fresh Z, and the architecture per n. Files: `class_check.py`, `by_fit.csv`, `summary.csv` (long format: median, max and min across reps), `class_check_log.txt`. |
| `toy_xfit_phihat/` | Task B. Toy design: joint (θ, t) check-loss LP with a 2-fold cross-fitted φ̂ = Ê[X\|Z] (MLP primary, gradient boosting as a robustness check) in place of the oracle φ*. m̂ is the Adam polished fit, held fixed. Includes sandwich-SE coverage. See its `README.md`. |
| `case3_xfit_m/` | Case 3 cross-fitting run for m̂. It is run separately and not produced by the scripts here. |
| `_scratch/` | Throwaway inspection script (`peek_case3.py`) used to read the Case 3 source definitions (m₀ = `nonlinear_truth(z, 3)`, clipping, SE code). |
