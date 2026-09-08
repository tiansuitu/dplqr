# DPLQR epoch diagnostic

Created 08 September 2026 (08092026). Start with `RESULTS.md` for the completed findings and plots. Open `epoch_robustness.ipynb` with the repository `.venv` and Run All to reproduce them.

The notebook compares budgets 10, 25, 50, 100, 200, 500 and 1000 using terminal epochs, validation-best epochs, and patience-15 early stopping. It keeps the six first-pass datasets and network seeds. Parent first-pass settings and commented full-replication assignments remain unchanged.

Only scientific definitions are imported from `../../simulate_homoscedastic.ipynb`, together with original root `dqAux.py`. No parent settings or run cells execute. The deleted simulation `.py` is not required; no R step is needed for coefficient/prediction diagnostics without coverage.

Completed trajectories in `trajectories/` are reused on reruns. Changed settings or source require a fresh `OUTPUT_DIR`. Save edits and restart the kernel before Run All, so the recorded source identity matches the executed code. An interrupted trajectory restarts from its original seed. A larger Monte Carlo design should use a separate experiment and baseline.

`epoch_trajectories.csv` includes initialization (epoch 0); summaries exclude it. Budget results distinguish the epoch cap, actual/virtual training duration, and selected epoch. Summary columns include signed bias, sample SD, mean absolute/squared coefficient error, nuisance relative MSE and losses. Paired drift records each dataset's change from terminal epoch 100. PNG/PDF plots and the executed notebook are saved here.

Two repetitions provide preliminary paired diagnostics, not reliable population-bias estimates or the paper's full simulation tables.
