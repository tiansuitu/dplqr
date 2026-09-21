#!/usr/bin/env python3
"""Experiment 3: joint Adam (theta, m); optional profiled-theta diagnostic.

PRIMARY estimator = final joint-Adam theta.
SECONDARY = linprog theta with frozen m_hat (diagnostic only).
No early stopping. Oracle f0 and varphi*. Sandwich uses residualized X_tilde.
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
# Configuration (aligned with Experiment 2)
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

F0 = 2.0 / (math.pi * math.sqrt(3.0))
assert np.isclose(F0, stats.t.pdf(0.0, df=DF_T), rtol=1e-12, atol=1e-12)

HIDDEN = [32, 32, 16]
EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
CHECKPOINT_EPOCHS = (1, 100, 250, 500, 1000)

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
    """Convex L1 / median regression (secondary profiled diagnostic only)."""
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
    """Semiparametric sandwich on residualized X_tilde (estimated-m setting)."""
    n = X_tilde.shape[0]
    Xt = np.asarray(X_tilde, dtype=np.float64)
    Sxx = (Xt.T @ Xt) / n
    Sigma1 = tau * (1.0 - tau) * Sxx
    Sigma2 = f0 * Sxx
    A = np.linalg.solve(Sigma2, np.eye(Sigma2.shape[0]))
    V = A @ Sigma1 @ A
    return Sigma1, Sigma2, V


class MLP(nn.Module):
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


class JointModel(nn.Module):
    """q = X @ theta + m_NN(Z); theta and NN jointly trainable."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.m_net = MLP()

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return X @ self.theta + self.m_net(Z)


def train_joint(Y: np.ndarray, X: np.ndarray, Z: np.ndarray, seed: int):
    """Joint Adam for exactly EPOCHS epochs. No early stopping / no val split."""
    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))

    model = JointModel().to(DEVICE)
    # mild random init for theta away from zero but not at truth
    with torch.no_grad():
        model.theta.copy_(torch.tensor([0.0, 0.0], dtype=torch.float64, device=DEVICE))
    theta_init = model.theta.detach().cpu().numpy().copy()

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    Y_t = torch.as_tensor(Y, dtype=torch.float64, device=DEVICE)
    X_t = torch.as_tensor(X, dtype=torch.float64, device=DEVICE)
    Z_t = torch.as_tensor(Z, dtype=torch.float64, device=DEVICE)

    n = Y.shape[0]
    loss_log: dict[int, float] = {}
    theta_log: dict[int, np.ndarray] = {}

    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = np_rng.permutation(n)
        running = 0.0
        seen = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
            pred = model(X_t[idx_t], Z_t[idx_t])
            u = Y_t[idx_t] - pred
            loss = 0.5 * torch.mean(torch.abs(u))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch in CHECKPOINT_EPOCHS:
            loss_log[epoch] = running / max(seen, 1)
            theta_log[epoch] = model.theta.detach().cpu().numpy().copy()

    model.eval()
    with torch.no_grad():
        m_hat = model.m_net(Z_t).detach().cpu().numpy().astype(np.float64)
        theta_adam = model.theta.detach().cpu().numpy().astype(np.float64)
        final_loss = check_loss(Y - X @ theta_adam - m_hat, TAU)

    return model, theta_init, theta_adam, m_hat, final_loss, loss_log, theta_log


RESULT_COLUMNS = [
    "n",
    "replication",
    "seed",
    "experiment",
    # primary joint Adam
    "theta1_hat_joint_adam",
    "theta2_hat_joint_adam",
    "theta1_error_adam",
    "theta2_error_adam",
    "sqrt_n_theta1_error_adam",
    "sqrt_n_theta2_error_adam",
    "oracle_se1",
    "oracle_se2",
    "T1_adam",
    "T2_adam",
    "coverage1_adam",
    "coverage2_adam",
    "nuisance_L2",
    "nuisance_mean_error",
    "final_training_loss",
    "Cn1",
    "Cn2",
    "Cn_norm",
    "lhs1_adam",
    "lhs2_adam",
    "minus_Sn1",
    "minus_Sn2",
    "IF_gap1_adam",
    "IF_gap2_adam",
    "IF_gap_norm_adam",
    # secondary profiled
    "theta1_hat_profiled",
    "theta2_hat_profiled",
    "theta1_error_profiled",
    "theta2_error_profiled",
    "T1_profiled",
    "T2_profiled",
    "coverage1_profiled",
    "coverage2_profiled",
    "IF_gap1_profiled",
    "IF_gap2_profiled",
    "IF_gap_norm_profiled",
    "profiled_check_loss",
    # checkpoints
    "theta1_init",
    "theta2_init",
    "loss_epoch_1",
    "loss_epoch_100",
    "loss_epoch_250",
    "loss_epoch_500",
    "loss_epoch_1000",
    "theta1_epoch_1000",
    "theta2_epoch_1000",
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


def _studentize(err: np.ndarray, se: np.ndarray):
    T = err / se
    cover = (np.abs(T) <= WALD_Z).astype(np.float64)
    return T, cover


def run_one(n: int, rep: int) -> dict:
    seed = replication_seed(n, rep)
    rng = np.random.default_rng(seed)
    data = generate_sample(n, rng)
    Y, X, Z = data["Y"], data["X"], data["Z"]
    m0, X_tilde, eps = data["m0"], data["X_tilde"], data["eps"]

    _model, theta_init, theta_adam, m_hat, final_loss, loss_log, theta_log = train_joint(
        Y, X, Z, seed=seed
    )

    # Secondary diagnostic: freeze m_hat, convex median regression for theta
    theta_prof = median_regression_lp(Y, X, m_hat)
    profiled_loss = check_loss(Y - X @ theta_prof - m_hat, TAU)

    err_a = theta_adam - THETA0
    err_p = theta_prof - THETA0
    sqrt_n = math.sqrt(n)

    _S1, Sigma2, V = oracle_sandwich(X_tilde, F0, TAU)
    se = np.sqrt(np.diag(V) / n)
    T_a, cov_a = _studentize(err_a, se)
    T_p, cov_p = _studentize(err_p, se)

    dm = m_hat - m0
    nuisance_L2 = float(np.sqrt(np.mean(dm**2)))
    nuisance_mean_error = float(np.mean(dm))
    Cn = sqrt_n * F0 * (X_tilde * dm[:, None]).mean(axis=0)

    score = (eps < 0.0).astype(np.float64) - TAU
    S_n = (X_tilde * score[:, None]).sum(axis=0) / sqrt_n
    lhs_a = Sigma2 @ (sqrt_n * err_a)
    lhs_p = Sigma2 @ (sqrt_n * err_p)
    IF_a = lhs_a + S_n
    IF_p = lhs_p + S_n

    th1000 = theta_log.get(1000, theta_adam)

    return {
        "n": n,
        "replication": rep,
        "seed": seed,
        "experiment": "experiment3_joint_nn",
        "theta1_hat_joint_adam": float(theta_adam[0]),
        "theta2_hat_joint_adam": float(theta_adam[1]),
        "theta1_error_adam": float(err_a[0]),
        "theta2_error_adam": float(err_a[1]),
        "sqrt_n_theta1_error_adam": float(sqrt_n * err_a[0]),
        "sqrt_n_theta2_error_adam": float(sqrt_n * err_a[1]),
        "oracle_se1": float(se[0]),
        "oracle_se2": float(se[1]),
        "T1_adam": float(T_a[0]),
        "T2_adam": float(T_a[1]),
        "coverage1_adam": float(cov_a[0]),
        "coverage2_adam": float(cov_a[1]),
        "nuisance_L2": nuisance_L2,
        "nuisance_mean_error": nuisance_mean_error,
        "final_training_loss": float(final_loss),
        "Cn1": float(Cn[0]),
        "Cn2": float(Cn[1]),
        "Cn_norm": float(np.linalg.norm(Cn)),
        "lhs1_adam": float(lhs_a[0]),
        "lhs2_adam": float(lhs_a[1]),
        "minus_Sn1": float(-S_n[0]),
        "minus_Sn2": float(-S_n[1]),
        "IF_gap1_adam": float(IF_a[0]),
        "IF_gap2_adam": float(IF_a[1]),
        "IF_gap_norm_adam": float(np.linalg.norm(IF_a)),
        "theta1_hat_profiled": float(theta_prof[0]),
        "theta2_hat_profiled": float(theta_prof[1]),
        "theta1_error_profiled": float(err_p[0]),
        "theta2_error_profiled": float(err_p[1]),
        "T1_profiled": float(T_p[0]),
        "T2_profiled": float(T_p[1]),
        "coverage1_profiled": float(cov_p[0]),
        "coverage2_profiled": float(cov_p[1]),
        "IF_gap1_profiled": float(IF_p[0]),
        "IF_gap2_profiled": float(IF_p[1]),
        "IF_gap_norm_profiled": float(np.linalg.norm(IF_p)),
        "profiled_check_loss": float(profiled_loss),
        "theta1_init": float(theta_init[0]),
        "theta2_init": float(theta_init[1]),
        "loss_epoch_1": float(loss_log.get(1, np.nan)),
        "loss_epoch_100": float(loss_log.get(100, np.nan)),
        "loss_epoch_250": float(loss_log.get(250, np.nan)),
        "loss_epoch_500": float(loss_log.get(500, np.nan)),
        "loss_epoch_1000": float(loss_log.get(1000, np.nan)),
        "theta1_epoch_1000": float(th1000[0]),
        "theta2_epoch_1000": float(th1000[1]),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in df.groupby("n"):
        n = int(n)
        rec: dict = {"n": n, "Q": int(len(g))}
        # primary adam
        for j in (1, 2):
            err = g[f"theta{j}_error_adam"].to_numpy(dtype=np.float64)
            T = g[f"T{j}_adam"].to_numpy(dtype=np.float64)
            se = g[f"oracle_se{j}"].to_numpy(dtype=np.float64)
            cov = g[f"coverage{j}_adam"].to_numpy(dtype=np.float64)
            hat = g[f"theta{j}_hat_joint_adam"].to_numpy(dtype=np.float64)
            rec.update(
                {
                    f"adam_theta{j}_mean": float(hat.mean()),
                    f"adam_theta{j}_bias": float(err.mean()),
                    f"adam_theta{j}_sqrtn_bias": float(math.sqrt(n) * err.mean()),
                    f"adam_theta{j}_mc_sd": float(err.std(ddof=1)),
                    f"adam_theta{j}_mean_oracle_se": float(se.mean()),
                    f"adam_T{j}_mean": float(T.mean()),
                    f"adam_T{j}_sd": float(T.std(ddof=1)),
                    f"adam_coverage{j}": float(cov.mean()),
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
                "IF_gap_norm_adam_mean": float(g["IF_gap_norm_adam"].mean()),
                "final_training_loss_mean": float(g["final_training_loss"].mean()),
            }
        )
        # profiled separately
        for j in (1, 2):
            err = g[f"theta{j}_error_profiled"].to_numpy(dtype=np.float64)
            T = g[f"T{j}_profiled"].to_numpy(dtype=np.float64)
            cov = g[f"coverage{j}_profiled"].to_numpy(dtype=np.float64)
            hat = g[f"theta{j}_hat_profiled"].to_numpy(dtype=np.float64)
            rec.update(
                {
                    f"profiled_theta{j}_mean": float(hat.mean()),
                    f"profiled_theta{j}_bias": float(err.mean()),
                    f"profiled_theta{j}_mc_sd": float(err.std(ddof=1)),
                    f"profiled_T{j}_sd": float(T.std(ddof=1)),
                    f"profiled_coverage{j}": float(cov.mean()),
                }
            )
        rec["IF_gap_norm_profiled_mean"] = float(g["IF_gap_norm_profiled"].mean())
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
        ax.hist(g["T1_adam"], bins=15, density=True, alpha=0.75, color="#4C72B0", edgecolor="white")
        xs = np.linspace(-3.5, 3.5, 200)
        ax.plot(xs, norm.pdf(xs), "k-", lw=1.5, label="N(0,1)")
        ax.set_title(f"Primary T1 (joint Adam), n={n}")
        ax.legend()
        _savefig(FIG_DIR / f"T1_adam_hist_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T1_adam"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T1 Adam QQ, n={n}")
        _savefig(FIG_DIR / f"T1_adam_qq_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T2_adam"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T2 Adam QQ, n={n}")
        _savefig(FIG_DIR / f"T2_adam_qq_n{n}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, j in zip(axes, (1, 2)):
            x = g[f"minus_Sn{j}"].to_numpy(dtype=np.float64)
            y = g[f"lhs{j}_adam"].to_numpy(dtype=np.float64)
            lim = float(np.max(np.abs(np.concatenate([x, y]))) * 1.1 + 1e-8)
            ax.scatter(x, y, s=18, alpha=0.8)
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax.set_xlabel(rf"$[-S_n]_{j}$")
            ax.set_ylabel(rf"$[\Sigma_2\sqrt{{n}}(\hat\theta_{{\mathrm{{adam}}}}-\theta_0)]_{j}$")
            ax.set_title(f"IF check {j} (Adam), n={n}")
            ax.set_aspect("equal", adjustable="box")
        _savefig(FIG_DIR / f"IF_scatter_adam_n{n}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(g["Cn1"], bins=15, color="#55A868", edgecolor="white")
        axes[0].set_title(f"Cn1 n={n}")
        axes[1].hist(g["Cn2"], bins=15, color="#C44E52", edgecolor="white")
        axes[1].set_title(f"Cn2 n={n}")
        _savefig(FIG_DIR / f"Cn_hist_n{n}.png")

        for col, name in [
            ("Cn_norm", "Cn_norm"),
            ("nuisance_L2", "nuisance_L2"),
            ("IF_gap_norm_adam", "IF_gap_norm_adam"),
        ]:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.hist(g[col], bins=15, color="#8172B3", edgecolor="white")
            ax.set_title(f"{name} (n={n})")
            _savefig(FIG_DIR / f"{name}_hist_n{n}.png")

        # one concise adam vs profiled comparison
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, j in zip(axes, (1, 2)):
            xa = g[f"theta{j}_hat_joint_adam"]
            xp = g[f"theta{j}_hat_profiled"]
            ax.scatter(xa, xp, s=18, alpha=0.8)
            lo = float(min(xa.min(), xp.min()))
            hi = float(max(xa.max(), xp.max()))
            pad = 0.05 * (hi - lo + 1e-8)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            ax.axhline(THETA0[j - 1], color="gray", lw=0.8, alpha=0.7)
            ax.axvline(THETA0[j - 1], color="gray", lw=0.8, alpha=0.7)
            ax.set_xlabel(f"theta{j} joint Adam (primary)")
            ax.set_ylabel(f"theta{j} profiled (secondary)")
            ax.set_title(f"Adam vs profiled theta{j} (n={n})")
            ax.set_aspect("equal", adjustable="box")
        _savefig(FIG_DIR / f"adam_vs_profiled_n{n}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [df.loc[df["n"] == n, "Cn_norm"].to_numpy() for n in N_VALUES],
        tick_labels=[str(n) for n in N_VALUES],
    )
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|C_n\|_2$")
    ax.set_title("Empirical orthogonality norm (joint)")
    _savefig(FIG_DIR / "Cn_norm_boxplot.png")


def write_config() -> dict:
    config = {
        "experiment": "experiment3_joint_nn",
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
        "primary_estimator": "theta_hat_joint_adam",
        "secondary_diagnostic": "theta_hat_joint_profiled (linprog after freeze m)",
        "sandwich": "semiparametric_on_X_tilde",
        "device": str(DEVICE),
        "note": "Joint Adam (theta,m); oracle f0/varphi*; profiled theta is diagnostic only.",
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
                    f"adam={row['theta1_hat_joint_adam']:.4f},{row['theta2_hat_joint_adam']:.4f}  "
                    f"prof={row['theta1_hat_profiled']:.4f},{row['theta2_hat_profiled']:.4f}  "
                    f"nuisance_L2={row['nuisance_L2']:.4f}  Cn_norm={row['Cn_norm']:.3f}"
                )

    if not RESULTS_CSV.exists():
        raise FileNotFoundError("No replication_results.csv")

    df = pd.read_csv(RESULTS_CSV)
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print("Wrote", SUMMARY_CSV)

    make_figures(df)
    print("Figures in", FIG_DIR)
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
