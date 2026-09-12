# Case 3 fixed-epoch normality

Open `asymptotic_normality_case3.ipynb` with the repository's Python kernel and run cells in order. Defaults: Q=20, n=500/1000/2000/4000, epochs=200/500/1000; no early stopping in the main or SE projection fits. Requires existing Python dependencies and base R for the unchanged residual-density estimator.

Increase Q to resume and extend `run/`; changed scientific settings require a new output subfolder. The notebook saves keyed fit/result checkpoints, raw rows, summaries for both coefficients, and theta1 plots. Skip the simulation cell to regenerate reports from saved results.

Setup only: no simulations or smoke tests were run, and no results are included.
