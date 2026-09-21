#!/usr/bin/env python3
"""Experiment 2: NN nuisance with theta fixed at theta_0; then convex theta step.

No early stopping. Train exactly EPOCHS epochs. Oracle f0 and varphi*.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import linprog
from scipy.stats import norm

np.set_printoptions(precision=6, suppress=True)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_VALUES = [1000, 2000]
Q = 50
BASE_SEED = 20260921
TAU = 0.5
THETA0 = np.array([1.0, -1.0], dtype=np.float64)
M0_COEF = 0.56
DF_T = 3
Z_DIM = 8
WALD_Z = 1.959963984540054

# Oracle t_3 density at 0: 2 / (pi * sqrt(3))
F0 = 2.0 / (math.pi * math.sqrt(3.0))
assert np.isclose(F0, stats.t.pdf(0.0, df=DF_T), rtol=1e-12, atol=1e-12)

HIDDEN = [32, 32, 16]
EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
LOSS_LOG_EPOCHS = (1, 100, 250, 500, 1000)

ROOT = Path(".").resolve()
RUN_DIR = ROOT / "run"
FIG_DIR = RUN_DIR / "figures"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RUN_DIR / "replication_results.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"
CONFIG_JSON = RUN_DIR / "config.json"

DEVICE = torch.device("cpu")


def replication_seed(n: int, rep: int) -> int:
    return int(BASE_SEED + 1_000_003 * n + 1_009 * rep)


def equicorr_R(d: int = 10, rho: float = 0.5) -> np.ndarray:
    R = np.full((d, d), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    return R


def generate_sample(n: int, rng: np.random.Generator) -> dict:
    """Zhong–Wang Simulation I Case 1 DGP + analytic varphi*."""
    R = equicorr_R(10, 0.5)
    L = np.linalg.cholesky(R)
    G = rng.standard_normal(size=(n, 10)) @ L.T

    Ztilde = 2.0 * norm.cdf(G)
    Z = Ztilde[:, :8].astype(np.float64)
    X1 = (G[:, 8] > 0.0).astype(np.float64)
    X2 = Ztilde[:, 9].astype(np.float64)
    X = np.column_stack([X1, X2])

    m0 = M0_COEF * Z.sum(axis=1)
    eps = rng.standard_t(df=DF_T, size=n)
    Y = X @ THETA0 + m0 + eps

    # G9 | G1:8 ~ N(S/9, 5/9). Equiv. G_j = Phi^{-1}(Z_j/2).
    S = G[:, :8].sum(axis=1)
    phi1 = norm.cdf(S / (3.0 * math.sqrt(5.0)))
    phi2 = 2.0 * norm.cdf(S / (3.0 * math.sqrt(14.0)))
    varphi = np.column_stack([phi1, phi2])
    X_tilde = X - varphi

    return {
        "Y": Y.astype(np.float64),
        "X": X,
        "Z": Z,
        "m0": m0.astype(np.float64),
        "eps": eps.astype(np.float64),
        "X_tilde": X_tilde.astype(np.float64),
        "varphi": varphi.astype(np.float64),
    }


def check_loss(resid: np.ndarray, tau: float = TAU) -> float:
    resid = np.asarray(resid, dtype=np.float64).ravel()
    return float(np.mean(resid * (tau - (resid < 0.0).astype(np.float64))))


def median_regression_lp(Y: np.ndarray, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Global L1 / median regression via linprog (HiGHS)."""
    Y = np.asarray(Y, dtype=np.float64).ravel()
    offset = np.asarray(offset, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    c = np.concatenate([np.zeros(p), np.ones(n), np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = Y - offset
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return np.asarray(res.x[:p], dtype=np.float64)


def oracle_sandwich(X_tilde: np.ndarray, f0: float = F0, tau: float = TAU):
    n = X_tilde.shape[0]
    Xt = np.asarray(X_tilde, dtype=np.float64)
    Sxx = (Xt.T @ Xt) / n
    Sigma1 = tau * (1.0 - tau) * Sxx
    Sigma2 = f0 * Sxx
    A = np.linalg.solve(Sigma2, np.eye(Sigma2.shape[0]))
    V = A @ Sigma1 @ A
    return Sigma1, Sigma2, V


class MLP(nn.Module):
    """Dense ReLU MLP for m(Z); no sparsity mask."""

    def __init__(self, in_dim: int = Z_DIM, hidden: list[int] | None = None, out_dim: int = 1):
        super().__init__()
        if hidden is None:
            hidden = list(HIDDEN)
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def train_m_fixed_theta(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    seed: int,
) -> tuple[MLP, float, dict[int, float]]:
    """Adam-train m with theta glued at THETA0. Exactly EPOCHS epochs; no early stop."""
    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    target = torch.as_tensor(Y - X @ THETA0, dtype=torch.float64, device=DEVICE)
    Z_t = torch.as_tensor(Z, dtype=torch.float64, device=DEVICE)

    n = Y.shape[0]
    loss_log: dict[int, float] = {}

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = np_rng.permutation(n)
        running = 0.0
        seen = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
            u = target[idx_t] - model(Z_t[idx_t])
            loss = 0.5 * torch.mean(torch.abs(u))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch in LOSS_LOG_EPOCHS:
            loss_log[epoch] = running / max(seen, 1)

    model.eval()
    with torch.no_grad():
        m_hat = model(Z_t).detach().cpu().numpy()
    final_loss = check_loss(Y - X @ THETA0 - m_hat, TAU)
    return model, final_loss, loss_log


def predict_m(model: MLP, Z: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z_t = torch.as_tensor(Z, dtype=torch.float64, device=DEVICE)
        return model(z_t).detach().cpu().numpy().astype(np.float64)


RESULT_COLUMNS = [
    "n",
    "replication",
    "seed",
    "experiment",
    "theta1_hat",
    "theta2_hat",
    "theta1_error",
    "theta2_error",
    "sqrt_n_theta1_error",
    "sqrt_n_theta2_error",
    "oracle_se1",
    "oracle_se2",
    "T1",
    "T2",
    "coverage1",
    "coverage2",
    "nuisance_L2",
    "nuisance_mean_error",
    "nuisance_max_abs_error",
    "final_training_loss",
    "Cn1",
    "Cn2",
    "Cn_norm",
    "lhs1",
    "lhs2",
    "minus_Sn1",
    "minus_Sn2",
    "IF_gap1",
    "IF_gap2",
    "IF_gap_norm",
    "convex_theta_check_loss",
    "loss_epoch_1",
    "loss_epoch_100",
    "loss_epoch_250",
    "loss_epoch_500",
    "loss_epoch_1000",
]


def load_completed() -> set[tuple[int, int]]:
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        return set()
    return set(zip(df["n"].astype(int), df["replication"].astype(int)))


def append_result(row: dict) -> None:
    df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def run_one(n: int, rep: int) -> dict:
    seed = replication_seed(n, rep)
    rng = np.random.default_rng(seed)
    data = generate_sample(n, rng)
    Y, X, Z = data["Y"], data["X"], data["Z"]
    m0, X_tilde, eps = data["m0"], data["X_tilde"], data["eps"]

    model, final_train_loss, loss_log = train_m_fixed_theta(Y, X, Z, seed=seed)
    m_hat = predict_m(model, Z)

    theta_hat = median_regression_lp(Y, X, m_hat)
    err = theta_hat - THETA0
    sqrt_n = math.sqrt(n)

    _Sigma1, Sigma2, V = oracle_sandwich(X_tilde, F0, TAU)
    se = np.sqrt(np.diag(V) / n)
    T = err / se
    cover = (np.abs(T) <= WALD_Z).astype(np.float64)

    convex_check_loss = check_loss(Y - X @ theta_hat - m_hat, TAU)

    dm = m_hat - m0
    nuisance_L2 = float(np.sqrt(np.mean(dm**2)))
    nuisance_mean_error = float(np.mean(dm))
    nuisance_max_abs = float(np.max(np.abs(dm)))

    Cn = sqrt_n * F0 * (X_tilde * dm[:, None]).mean(axis=0)

    score = (eps < 0.0).astype(np.float64) - TAU
    S_n = (X_tilde * score[:, None]).sum(axis=0) / sqrt_n
    lhs = Sigma2 @ (sqrt_n * err)
    IF_gap = lhs + S_n

    return {
        "n": n,
        "replication": rep,
        "seed": seed,
        "experiment": "experiment2_fixed_theta_nn",
        "theta1_hat": float(theta_hat[0]),
        "theta2_hat": float(theta_hat[1]),
        "theta1_error": float(err[0]),
        "theta2_error": float(err[1]),
        "sqrt_n_theta1_error": float(sqrt_n * err[0]),
        "sqrt_n_theta2_error": float(sqrt_n * err[1]),
        "oracle_se1": float(se[0]),
        "oracle_se2": float(se[1]),
        "T1": float(T[0]),
        "T2": float(T[1]),
        "coverage1": float(cover[0]),
        "coverage2": float(cover[1]),
        "nuisance_L2": nuisance_L2,
        "nuisance_mean_error": nuisance_mean_error,
        "nuisance_max_abs_error": nuisance_max_abs,
        "final_training_loss": float(final_train_loss),
        "Cn1": float(Cn[0]),
        "Cn2": float(Cn[1]),
        "Cn_norm": float(np.linalg.norm(Cn)),
        "lhs1": float(lhs[0]),
        "lhs2": float(lhs[1]),
        "minus_Sn1": float(-S_n[0]),
        "minus_Sn2": float(-S_n[1]),
        "IF_gap1": float(IF_gap[0]),
        "IF_gap2": float(IF_gap[1]),
        "IF_gap_norm": float(np.linalg.norm(IF_gap)),
        "convex_theta_check_loss": float(convex_check_loss),
        "loss_epoch_1": float(loss_log.get(1, np.nan)),
        "loss_epoch_100": float(loss_log.get(100, np.nan)),
        "loss_epoch_250": float(loss_log.get(250, np.nan)),
        "loss_epoch_500": float(loss_log.get(500, np.nan)),
        "loss_epoch_1000": float(loss_log.get(1000, np.nan)),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in df.groupby("n"):
        n = int(n)
        rec: dict = {"n": n, "Q": int(len(g))}
        for j in (1, 2):
            err = g[f"theta{j}_error"].to_numpy(dtype=np.float64)
            T = g[f"T{j}"].to_numpy(dtype=np.float64)
            se = g[f"oracle_se{j}"].to_numpy(dtype=np.float64)
            cov = g[f"coverage{j}"].to_numpy(dtype=np.float64)
            hat = g[f"theta{j}_hat"].to_numpy(dtype=np.float64)
            rec.update(
                {
                    f"theta{j}_mean": float(hat.mean()),
                    f"theta{j}_bias": float(err.mean()),
                    f"theta{j}_sqrtn_bias": float(math.sqrt(n) * err.mean()),
                    f"theta{j}_mc_sd": float(err.std(ddof=1)),
                    f"theta{j}_mean_oracle_se": float(se.mean()),
                    f"T{j}_mean": float(T.mean()),
                    f"T{j}_sd": float(T.std(ddof=1)),
                    f"coverage{j}": float(cov.mean()),
                }
            )
        rec.update(
            {
                "nuisance_L2_mean": float(g["nuisance_L2"].mean()),
                "nuisance_L2_sd": float(g["nuisance_L2"].std(ddof=1)),
                "Cn1_mean": float(g["Cn1"].mean()),
                "Cn1_sd": float(g["Cn1"].std(ddof=1)),
                "Cn2_mean": float(g["Cn2"].mean()),
                "Cn2_sd": float(g["Cn2"].std(ddof=1)),
                "Cn_norm_mean": float(g["Cn_norm"].mean()),
                "IF_gap1_mean": float(g["IF_gap1"].mean()),
                "IF_gap2_mean": float(g["IF_gap2"].mean()),
                "IF_gap_norm_mean": float(g["IF_gap_norm"].mean()),
                "final_training_loss_mean": float(g["final_training_loss"].mean()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def make_figures(df: pd.DataFrame) -> None:
    for n in N_VALUES:
        g = df[df["n"] == n]

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(g["T1"], bins=15, density=True, alpha=0.75, color="#4C72B0", edgecolor="white")
        xs = np.linspace(-3.5, 3.5, 200)
        ax.plot(xs, norm.pdf(xs), "k-", lw=1.5, label="N(0,1)")
        ax.set_title(f"T1 density (n={n})")
        ax.set_xlabel("T1")
        ax.legend()
        _savefig(FIG_DIR / f"T1_hist_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T1"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T1 QQ vs N(0,1) (n={n})")
        _savefig(FIG_DIR / f"T1_qq_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T2"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T2 QQ vs N(0,1) (n={n})")
        _savefig(FIG_DIR / f"T2_qq_n{n}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, j in zip(axes, (1, 2)):
            x = g[f"minus_Sn{j}"].to_numpy(dtype=np.float64)
            y = g[f"lhs{j}"].to_numpy(dtype=np.float64)
            lim = float(np.max(np.abs(np.concatenate([x, y]))) * 1.1 + 1e-8)
            ax.scatter(x, y, s=18, alpha=0.8)
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax.set_xlabel(rf"$[-S_n]_{j}$")
            ax.set_ylabel(rf"$[\Sigma_2\sqrt{{n}}(\hat\theta^{{(2)}}-\theta_0)]_{j}$")
            ax.set_title(f"IF check component {j} (n={n})")
            ax.set_aspect("equal", adjustable="box")
        _savefig(FIG_DIR / f"IF_scatter_n{n}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(g["Cn1"], bins=15, color="#55A868", edgecolor="white")
        axes[0].set_title(f"Cn1 (n={n})")
        axes[1].hist(g["Cn2"], bins=15, color="#C44E52", edgecolor="white")
        axes[1].set_title(f"Cn2 (n={n})")
        _savefig(FIG_DIR / f"Cn_hist_n{n}.png")

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(g["Cn_norm"], bins=15, color="#8172B3", edgecolor="white")
        ax.set_title(rf"$\|C_n\|_2$ (n={n})")
        ax.set_xlabel("Cn_norm")
        _savefig(FIG_DIR / f"Cn_norm_hist_n{n}.png")

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(g["nuisance_L2"], bins=15, color="#CCB974", edgecolor="white")
        ax.set_title(f"nuisance L2 (n={n})")
        ax.set_xlabel(r"$\|\hat m - m_0\|_{n,2}$")
        _savefig(FIG_DIR / f"nuisance_L2_hist_n{n}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df.loc[df["n"] == n, "Cn_norm"].to_numpy() for n in N_VALUES]
    ax.boxplot(data, tick_labels=[str(n) for n in N_VALUES])
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|C_n\|_2$")
    ax.set_title("Empirical orthogonality norm")
    _savefig(FIG_DIR / "Cn_norm_boxplot.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df.loc[df["n"] == n, "nuisance_L2"].to_numpy() for n in N_VALUES]
    ax.boxplot(data, tick_labels=[str(n) for n in N_VALUES])
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|\hat m-m_0\|_{n,2}$")
    ax.set_title("Nuisance L2 error")
    _savefig(FIG_DIR / "nuisance_L2_boxplot.png")


def write_config() -> dict:
    config = {
        "experiment": "experiment2_fixed_theta_nn",
        "N_VALUES": N_VALUES,
        "Q": Q,
        "BASE_SEED": BASE_SEED,
        "TAU": TAU,
        "THETA0": THETA0.tolist(),
        "M0_COEF": M0_COEF,
        "DF_T": DF_T,
        "F0": F0,
        "WALD_Z": WALD_Z,
        "HIDDEN": HIDDEN,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "early_stopping": False,
        "sparsity_mask": False,
        "theta_during_m_train": "fixed_at_theta0",
        "theta_solver": "scipy.optimize.linprog(method='highs') L1 / median LP",
        "device": str(DEVICE),
        "note": "NN m with theta glued at truth; then convex theta; oracle f0 and varphi*.",
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    config = write_config()
    print("Wrote", CONFIG_JSON)
    print(json.dumps(config, indent=2))

    completed = load_completed()
    print(f"Already completed: {len(completed)} rows")

    for n in N_VALUES:
        for rep in range(1, Q + 1):
            key = (n, rep)
            if key in completed:
                continue
            row = run_one(n, rep)
            append_result(row)
            completed.add(key)
            if rep % 10 == 0 or rep == 1:
                print(
                    f"n={n} rep={rep}/{Q}  "
                    f"theta_hat={row['theta1_hat']:.4f},{row['theta2_hat']:.4f}  "
                    f"nuisance_L2={row['nuisance_L2']:.4f}  "
                    f"Cn_norm={row['Cn_norm']:.3f}  "
                    f"IF_gap_norm={row['IF_gap_norm']:.3f}"
                )

    if not RESULTS_CSV.exists():
        raise FileNotFoundError("No replication_results.csv — nothing to summarize")

    df = pd.read_csv(RESULTS_CSV)
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print("Wrote", SUMMARY_CSV)

    make_figures(df)
    print("Figures written to", FIG_DIR)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
