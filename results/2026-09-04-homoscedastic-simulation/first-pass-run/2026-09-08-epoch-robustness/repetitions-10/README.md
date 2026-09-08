# Epoch robustness: 10 repetitions per case

This extends the two-repetition experiment to **10 repetitions in each of Cases 1–3** (30 paired datasets), with the same n=1000, tau=0.5, architecture, seeds, optimizer settings and epoch budgets through 1000. Six previously verified paths are reused; 24 new paths extend the seed sequence.

Open `epoch_robustness.ipynb` with the repository `.venv` and Run All. The source is notebook-only, with no R step or dependency on the deleted simulation Python script. Original two-repetition results and main simulation settings are unchanged. Read `RESULTS.md` for results and plots; `executed_epoch_robustness.ipynb` preserves the executed cells.

`paired_datasets/` stores the train/validation/test arrays. `trajectories/` checkpoints each full path; unchanged reruns reuse them. Source/settings changes require a fresh OUTPUT_DIR. Save notebook edits and restart the kernel before running. Cache reuse checks scientific functions, original helper, training definitions, software versions, original inputs and settings. Only the six existing first-pass baselines have independent saved-fit comparisons; new baseline estimates are selected from their continuous training paths.

CSV outputs include all paths, three selection rules at seven budgets, bias/SD/MAE, bias Monte Carlo SE, paired drift, and provenance. Monte Carlo SE describes uncertainty across these repetitions, not a coefficient confidence interval for an individual dataset. No coverage, auxiliary-network or R fits are included.

Ten repetitions provide a larger preliminary diagnostic, not the paper's full 200-repetition study. One initialization accompanies each newly generated dataset; sampling and initialization variability are combined.
