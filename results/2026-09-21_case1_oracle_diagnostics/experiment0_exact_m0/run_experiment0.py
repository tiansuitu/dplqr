#!/usr/bin/env python3
"""Non-interactive runner for Experiment 0 (exact m0 oracle)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from scipy.stats import norm

np.set_printoptions(precision=6, suppress=True)

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
X_DIM = 2
WALD_Z = 1.959963984540054  # approx 1.96

# Analytic t_3 density at 0: f_eps(0) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1+0)^...
# For nu=3: 2 / (pi * sqrt(3))
F0 = 2.0 / (math.pi * math.sqrt(3.0))
assert np.isclose(F0, stats.t.pdf(0.0, df=DF_T), rtol=1e-12, atol=1e-12)

ROOT = Path(".").resolve()
RUN_DIR = ROOT / "run"
FIG_DIR = RUN_DIR / "figures"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RUN_DIR / "replication_results.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"
CONFIG_JSON = RUN_DIR / "config.json"

print("F0 =", F0)
print("cwd =", ROOT)


# --- next cell ---

def replication_seed(n: int, rep: int) -> int:
    """Deterministic seed from BASE_SEED, n, and replication index (1-based)."""
    return int(BASE_SEED + 1_000_003 * n + 1_009 * rep)


def equicorr_R(d: int = 10, rho: float = 0.5) -> np.ndarray:
    R = np.full((d, d), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    return R


def generate_sample(n: int, rng: np.random.Generator):
    """Zhong–Wang Simulation I Case 1 DGP; returns arrays and latent G[:, :8]."""
    R = equicorr_R(10, 0.5)
    L = np.linalg.cholesky(R)
    G = rng.standard_normal(size=(n, 10)) @ L.T  # N(0,R)

    Ztilde = 2.0 * norm.cdf(G)  # each Uniform[0,2] marginally
    Z = Ztilde[:, :8].astype(np.float64)
    X1 = (G[:, 8] > 0.0).astype(np.float64)  # == 1{Ztilde_9 > 1}
    X2 = Ztilde[:, 9].astype(np.float64)
    X = np.column_stack([X1, X2])

    m0 = M0_COEF * Z.sum(axis=1)
    eps = rng.standard_t(df=DF_T, size=n)
    Y = X @ THETA0 + m0 + eps

    # Analytic varphi^*(Z) using latent G_1..G_8 (equiv. G_j = Phi^{-1}(Z_j/2))
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
        "G8": G[:, :8].astype(np.float64),
    }


# --- next cell ---

def median_regression_lp(Y: np.ndarray, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Global L1 / median regression: min sum_i |Y_i - offset_i - X_i @ theta|.

    LP form: min 1^T (u+v) s.t. X theta + u - v = Y - offset, u,v >= 0.
    Variables: [theta (p), u (n), v (n)].
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    offset = np.asarray(offset, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    assert Y.shape == (n,) and offset.shape == (n,)

    # objective: 0*theta + 1*u + 1*v
    c = np.concatenate([np.zeros(p), np.ones(n), np.ones(n)])

    # X theta + u - v = Y - offset
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = Y - offset

    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return np.asarray(res.x[:p], dtype=np.float64)


def oracle_sandwich(X: np.ndarray, f0: float, tau: float):
    """Linear QR sandwich on raw X when m0 is known (not residualized Xt).

    With known m0, Y-m0(Z)=X'theta+eps is ordinary linear median regression.
    Use E[XX']; reserve Xt=X-phi*(Z) for estimated-m experiments.
    """
    n = X.shape[0]
    Xt = X.astype(np.float64)
    Sxx = (Xt.T @ Xt) / n
    Sigma1 = tau * (1.0 - tau) * Sxx
    Sigma2 = f0 * Sxx
    A = np.linalg.solve(Sigma2, np.eye(Sigma2.shape[0]))
    V = A @ Sigma1 @ A
    return Sigma1, Sigma2, V


# --- next cell ---

RESULT_COLUMNS = [
    "n", "replication", "seed", "experiment",
    "theta1_hat", "theta2_hat",
    "theta1_error", "theta2_error",
    "sqrt_n_theta1_error", "sqrt_n_theta2_error",
    "oracle_se1", "oracle_se2",
    "T1", "T2",
    "coverage1", "coverage2",
    "empirical_check_loss",
    "Cn1", "Cn2",
    "lhs1", "lhs2",
    "minus_Sn1", "minus_Sn2",
    "IF_gap1", "IF_gap2", "IF_gap_norm",
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

    Y, X, m0 = data["Y"], data["X"], data["m0"]
    X_tilde, eps = data["X_tilde"], data["eps"]

    theta_hat = median_regression_lp(Y, X, m0)
    err = theta_hat - THETA0
    sqrt_n = math.sqrt(n)

    Sigma1, Sigma2, V = oracle_sandwich(X, F0, TAU)
    se = np.sqrt(np.diag(V) / n)
    T = err / se
    cover = (np.abs(T) <= WALD_Z).astype(np.float64)

    resid = Y - X @ theta_hat - m0
    check_loss = float(np.mean(0.5 * np.abs(resid)))

    # C_n(m0) = 0 exactly
    Cn = np.zeros(2, dtype=np.float64)

    # True score / IF diagnostic using true eps = Y - X theta0 - m0
    # S_n = n^{-1/2} sum X * (1{eps<0} - tau)  [raw X: known-m0 linear QR]
    score = (eps < 0.0).astype(np.float64) - TAU
    S_n = (X * score[:, None]).sum(axis=0) / sqrt_n
    lhs = Sigma2 @ (sqrt_n * err)
    IF_gap = lhs + S_n

    return {
        "n": n,
        "replication": rep,
        "seed": seed,
        "experiment": "experiment0_exact_m0",
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
        "empirical_check_loss": check_loss,
        "Cn1": float(Cn[0]),
        "Cn2": float(Cn[1]),
        "lhs1": float(lhs[0]),
        "lhs2": float(lhs[1]),
        "minus_Sn1": float(-S_n[0]),
        "minus_Sn2": float(-S_n[1]),
        "IF_gap1": float(IF_gap[0]),
        "IF_gap2": float(IF_gap[1]),
        "IF_gap_norm": float(np.linalg.norm(IF_gap)),
    }


# --- next cell ---

# Save config (no Monte Carlo executed until the next cell is run by you)
config = {
    "experiment": "experiment0_exact_m0",
    "N_VALUES": N_VALUES,
    "Q": Q,
    "BASE_SEED": BASE_SEED,
    "TAU": TAU,
    "THETA0": THETA0.tolist(),
    "M0_COEF": M0_COEF,
    "DF_T": DF_T,
    "F0": F0,
    "WALD_Z": WALD_Z,
    "solver": "scipy.optimize.linprog(method='highs') L1 / median regression LP",
    "note": "Oracle m0, f0, varphi*; estimate theta only.",
}
CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print("Wrote", CONFIG_JSON)
print(json.dumps(config, indent=2))


# --- next cell ---

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
            print(f"n={n} rep={rep}/{Q}  theta_hat={row['theta1_hat']:.4f},{row['theta2_hat']:.4f}  "
                  f"T=({row['T1']:.2f},{row['T2']:.2f})  IF_gap_norm={row['IF_gap_norm']:.3f}")

print("Done. Results at", RESULTS_CSV)


# --- next cell ---

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in df.groupby("n"):
        n = int(n)
        rec = {"n": n, "Q": int(len(g))}
        for j in (1, 2):
            err = g[f"theta{j}_error"].to_numpy(dtype=np.float64)
            T = g[f"T{j}"].to_numpy(dtype=np.float64)
            se = g[f"oracle_se{j}"].to_numpy(dtype=np.float64)
            cov = g[f"coverage{j}"].to_numpy(dtype=np.float64)
            hat = g[f"theta{j}_hat"].to_numpy(dtype=np.float64)
            rec.update({
                f"theta{j}_mean": float(hat.mean()),
                f"theta{j}_bias": float(err.mean()),
                f"theta{j}_sqrtn_bias": float(math.sqrt(n) * err.mean()),
                f"theta{j}_mc_sd": float(err.std(ddof=1)),
                f"theta{j}_mean_oracle_se": float(se.mean()),
                f"T{j}_mean": float(T.mean()),
                f"T{j}_sd": float(T.std(ddof=1)),
                f"coverage{j}": float(cov.mean()),
            })
        gap = g["IF_gap_norm"].to_numpy(dtype=np.float64)
        rec.update({
            "IF_gap_norm_mean": float(gap.mean()),
            "IF_gap_norm_sd": float(gap.std(ddof=1)),
            "IF_gap_norm_median": float(np.median(gap)),
            "check_loss_mean": float(g["empirical_check_loss"].mean()),
            "Cn1_mean": float(g["Cn1"].mean()),
            "Cn2_mean": float(g["Cn2"].mean()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


if not RESULTS_CSV.exists():
    raise FileNotFoundError("Run the Monte Carlo cell first to create replication_results.csv")

df = pd.read_csv(RESULTS_CSV)
summary = summarize(df)
summary.to_csv(SUMMARY_CSV, index=False)
print(summary.to_string(index=False))
print("Wrote", SUMMARY_CSV)


# --- next cell ---

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


df = pd.read_csv(RESULTS_CSV)

for n in N_VALUES:
    g = df[df["n"] == n]
    # 1-2: T1 histogram + QQ
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(g["T1"], bins=15, density=True, alpha=0.75, color="#4C72B0", edgecolor="white")
    xs = np.linspace(-3.5, 3.5, 200)
    ax.plot(xs, norm.pdf(xs), "k-", lw=1.5, label="N(0,1)")
    ax.set_title(f"T1 density (n={n})")
    ax.set_xlabel("T1")
    ax.legend()
    savefig(FIG_DIR / f"T1_hist_n{n}.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(g["T1"], dist="norm", plot=ax)
    ax.set_title(f"T1 QQ vs N(0,1) (n={n})")
    savefig(FIG_DIR / f"T1_qq_n{n}.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(g["T2"], dist="norm", plot=ax)
    ax.set_title(f"T2 QQ vs N(0,1) (n={n})")
    savefig(FIG_DIR / f"T2_qq_n{n}.png")

    # IF scatter: lhs vs -S_n
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, j in zip(axes, (1, 2)):
        x = g[f"minus_Sn{j}"]
        y = g[f"lhs{j}"]
        lim = float(np.max(np.abs(np.concatenate([x, y]))) * 1.1 + 1e-8)
        ax.scatter(x, y, s=18, alpha=0.8)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlabel(rf"$[-S_n]_{j}$")
        ax.set_ylabel(rf"$[\Sigma_2\sqrt{{n}}(\hat\theta-\theta_0)]_{j}$")
        ax.set_title(f"IF check component {j} (n={n})")
        ax.set_aspect("equal", adjustable="box")
    savefig(FIG_DIR / f"IF_scatter_n{n}.png")

# IF gap norm across n
fig, ax = plt.subplots(figsize=(6, 4))
data = [df.loc[df["n"] == n, "IF_gap_norm"].to_numpy() for n in N_VALUES]
ax.boxplot(data, tick_labels=[str(n) for n in N_VALUES])
ax.set_xlabel("n")
ax.set_ylabel(r"$\|IF\_gap\|_2$")
ax.set_title("Influence-function gap norm")
savefig(FIG_DIR / "IF_gap_norm_boxplot.png")

print("Figures written to", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p.name)
