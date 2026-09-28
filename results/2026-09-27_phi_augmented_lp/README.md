# 2026-09-27 — oracle φ*-augmented exact-ERM diagnostic (toy PLQR, Thm 3.3 step E1)

**This is a diagnostic only.** It uses the oracle φ*(Z), which is unknown in practice.

## Question
In `2026-09-23_psi_n_thm33/run_expansion_v2/main_26a7078945fc`, the leftover √n bias came from
S = √n Ψ_n(ζ̂, ĥ) ≠ 0. That was driven by the score along the φ* direction, while the θ score on raw X was ≈ 0.
The supplement's (E1) step treats ζ̂ as the exact minimizer with ĥ fixed. That requires m̂ + tφ* to stay inside the model class.
Here we enforce that condition directly and check whether S and the bias collapse.

## What the script does (`run_phi_lp.py`)
For every saved fit (n ∈ {500,1000,2000,4000}, 50 reps, stages baseline and polished), using the same seeds and data:
1. Reload the checkpoint and re-run `diagnostic_helpers.diagnose`. The saved θ̂, S, E, T and φ*-score must match to 1e-7 (asserted).
2. `theta_lp` holds m̂ fixed and solves the check-loss LP in θ only (a contrast).
3. `phi_lp` holds m̂ fixed and solves the check-loss LP jointly in (θ, t) with regressors (X, φ*(Z)) and offset m̂.
   The new fit is (θ̃, m̂ + t̃ φ*). This is exact ERM along both directions, so S should be O(1/n) (ties bound).
   The remaining √n bias should then be ≈ E/J.
Each variant is re-diagnosed with the unchanged `diagnose` (S, E, T, W, R_score, √n G, √n φ*-score, population integration).
The script asserts that each LP lowers the training check loss relative to the Adam fit.

Source run and helpers are only read, never modified.

## Outputs (`run/`)
- `replication_results.csv`: one row per (n, rep, stage, variant ∈ {adam, theta_lp, phi_lp}).
- `summary.csv`: mean, MC SE and mean |·| per (n, stage, variant, metric), including `E_over_J`.
- `paired_vs_adam.csv`: same-seed paired changes relative to Adam (signed change and change in |·|).
- `smoke/`: 2-rep smoke test (n=500 and 4000).

Run: `..\..\.venv\Scripts\python.exe run_phi_lp.py --out run` (about 10–20 min on CPU).
