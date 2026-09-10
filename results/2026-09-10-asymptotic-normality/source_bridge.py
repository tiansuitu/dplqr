"""Load only the September 4 DGP and DPLQR definitions; never run its notebook.

The whitelist excludes benchmark definitions, run cells, exports and settings.
The original dqAux.py supplies all network and loss implementations unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torchtuples import Model
import torchtuples as tt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOTEBOOK = ROOT / "results/2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb"
sys.path.insert(0, str(ROOT))
from dqAux import checkErrorMean, checkLoss, covNet, dqNetSparse

DEFINITIONS = (
    "seed_all", "nonlinear_truth", "generate_covariates", "generate_dataset", "as_numpy",
    "EarlyStopInMemory", "clip_neural_weights", "tensor_pair", "train_dplqr_once",
    "select_dplqr", "fit_projection", "dplqr_standard_errors", "fit_dplqr",
)


def find_rscript():
    executable = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if executable:
        return str(executable)
    candidates = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("R/R-*/bin/Rscript.exe"))
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError("Set RSCRIPT to Rscript; only base R stats::density is required.")


def residual_density_zero(residual):
    result = subprocess.run(
        [find_rscript(), "--vanilla", str(HERE / "density_only.R")],
        input="\n".join(format(v, ".17g") for v in residual),
        capture_output=True, text=True, check=True, timeout=120,
    )
    density = float(result.stdout.strip())
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Residual density at zero must be finite and positive")
    return density


def selected_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    selected = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS:
                if node.name in selected:
                    raise ValueError(f"Duplicate original definition: {node.name}")
                selected[node.name] = node
    if set(selected) != set(DEFINITIONS):
        raise ValueError(f"Missing original definitions: {set(DEFINITIONS) - set(selected)}")
    return selected


def source_digest():
    selected = selected_source()
    canonical = "\n".join(ast.dump(selected[name], include_attributes=False) for name in DEFINITIONS)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_original():
    # Explicit namespace avoids executing any notebook imports or executable cells.
    namespace = {name: globals()[name] for name in (
        "np", "norm", "random", "torch", "nn", "tt", "Model", "StandardScaler",
        "math", "itertools", "checkErrorMean", "checkLoss", "covNet", "dqNetSparse",
        "residual_density_zero",
    )}
    namespace.update(THETA=np.array([1.0, -1.0]), __name__="september4_dplqr")
    selected = selected_source()
    module = ast.Module(body=[selected[name] for name in DEFINITIONS], type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in DEFINITIONS}, namespace=namespace)
