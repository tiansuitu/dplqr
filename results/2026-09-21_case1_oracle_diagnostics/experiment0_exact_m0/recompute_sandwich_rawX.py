"""Recompute Exp0 oracle SE/T/coverage/IF using raw X (linear QR sandwich).

When m0 is known, Y - m0(Z) = X'θ + ε is ordinary linear median regression.
Asymptotic variance uses E[XX'], not E[X̃X̃'] with X̃=X-φ*(Z).
θ̂ is kept from the saved LP fits; only diagnostics are recomputed.
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
from scipy import stats
from scipy.stats import norm

BASE_SEED = 20260921
TAU = 0.5
THETA0 = np.array([1.0, -1.0], dtype=np.float64)
M0_COEF = 0.56
DF_T = 3
F0 = 2.0 / (math.pi * math.sqrt(3.0))
WALD_Z = 1.959963984540054
N_VALUES = [1000, 2000]

ROOT = Path(".").resolve()
RUN_DIR = ROOT / "run"
FIG_DIR = RUN_DIR / "figures"
RESULTS_CSV = RUN_DIR / "replication_results.csv"
SUMMARY_CSV = RUN_DIR / "summary.csv"
CONFIG_JSON = RUN_DIR / "config.json"
NOTE_MD = ROOT / "SANDWICH_CORRECTION.md"


def replication_seed(n: int, rep: int) -> int:
    return int(BASE_SEED + 1_000_003 * n + 1_009 * rep)


def equicorr_R(d: int = 10, rho: float = 0.5) -> np.ndarray:
    R = np.full((d, d), rho, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    return R


def generate_sample(n: int, rng: np.random.Generator):
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
    return {"Y": Y, "X": X, "Z": Z, "m0": m0, "eps": eps.astype(np.float64)}


def linear_qr_sandwich(X: np.ndarray, f0: float, tau: float):
    """Ordinary linear QR sandwich on raw X (known-m0 case)."""
    n = X.shape[0]
    Sxx = (X.T @ X) / n
    Sigma1 = tau * (1.0 - tau) * Sxx
    Sigma2 = f0 * Sxx
    A = np.linalg.solve(Sigma2, np.eye(Sigma2.shape[0]))
    V = A @ Sigma1 @ A
    return Sigma1, Sigma2, V


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
            "sandwich": "linear_QR_raw_X",
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def make_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for n in N_VALUES:
        g = df[df["n"] == n]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(g["T1"], bins=15, density=True, alpha=0.75, color="#4C72B0", edgecolor="white")
        xs = np.linspace(-3.5, 3.5, 200)
        ax.plot(xs, norm.pdf(xs), "k-", lw=1.5, label="N(0,1)")
        ax.set_title(f"T1 density (raw-X SE), n={n}")
        ax.set_xlabel("T1")
        ax.legend()
        savefig(FIG_DIR / f"T1_hist_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T1"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T1 QQ (raw-X SE), n={n}")
        savefig(FIG_DIR / f"T1_qq_n{n}.png")

        fig, ax = plt.subplots(figsize=(5, 5))
        stats.probplot(g["T2"].to_numpy(dtype=np.float64), dist="norm", plot=ax)
        ax.set_title(f"T2 QQ (raw-X SE), n={n}")
        savefig(FIG_DIR / f"T2_qq_n{n}.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, j in zip(axes, (1, 2)):
            x = g[f"minus_Sn{j}"].to_numpy(dtype=np.float64)
            y = g[f"lhs{j}"].to_numpy(dtype=np.float64)
            lim = float(np.max(np.abs(np.concatenate([x, y]))) * 1.1 + 1e-8)
            ax.scatter(x, y, s=18, alpha=0.8)
            ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax.set_xlabel(rf"$[-S_n]_{j}$ (raw X)")
            ax.set_ylabel(rf"$[\Sigma_2\sqrt{{n}}(\hat\theta-\theta_0)]_{j}$")
            ax.set_title(f"IF check {j} (n={n})")
            ax.set_aspect("equal", adjustable="box")
        savefig(FIG_DIR / f"IF_scatter_n{n}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df.loc[df["n"] == n, "IF_gap_norm"].to_numpy() for n in N_VALUES]
    ax.boxplot(data, tick_labels=[str(n) for n in N_VALUES])
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\|IF\_gap\|_2$")
    ax.set_title("IF gap norm (raw-X score)")
    savefig(FIG_DIR / "IF_gap_norm_boxplot.png")


def main():
    df = pd.read_csv(RESULTS_CSV)
    # backup previous (residualized) results once
    bak = RUN_DIR / "replication_results_residualized_Xt_BACKUP.csv"
    if not bak.exists():
        df.to_csv(bak, index=False)
        print("Backed up previous residualized diagnostics to", bak)

    rows = []
    for _, r in df.iterrows():
        n = int(r["n"])
        rep = int(r["replication"])
        seed = int(r["seed"])
        expected = replication_seed(n, rep)
        if seed != expected:
            raise RuntimeError(f"seed mismatch n={n} rep={rep}: csv={seed} expected={expected}")

        rng = np.random.default_rng(seed)
        data = generate_sample(n, rng)
        X, eps, m0, Y = data["X"], data["eps"], data["m0"], data["Y"]

        theta_hat = np.array([r["theta1_hat"], r["theta2_hat"]], dtype=np.float64)
        # sanity: errors should match saved
        err = theta_hat - THETA0
        if not np.allclose(err, [r["theta1_error"], r["theta2_error"]], atol=1e-10):
            raise RuntimeError(f"theta error mismatch at n={n} rep={rep}")

        sqrt_n = math.sqrt(n)
        Sigma1, Sigma2, V = linear_qr_sandwich(X, F0, TAU)
        se = np.sqrt(np.diag(V) / n)
        T = err / se
        cover = (np.abs(T) <= WALD_Z).astype(np.float64)

        resid = Y - X @ theta_hat - m0
        check_loss = float(np.mean(0.5 * np.abs(resid)))

        score = (eps < 0.0).astype(np.float64) - TAU
        S_n = (X * score[:, None]).sum(axis=0) / sqrt_n
        lhs = Sigma2 @ (sqrt_n * err)
        IF_gap = lhs + S_n

        row = {
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
            "Cn1": 0.0,
            "Cn2": 0.0,
            "lhs1": float(lhs[0]),
            "lhs2": float(lhs[1]),
            "minus_Sn1": float(-S_n[0]),
            "minus_Sn2": float(-S_n[1]),
            "IF_gap1": float(IF_gap[0]),
            "IF_gap2": float(IF_gap[1]),
            "IF_gap_norm": float(np.linalg.norm(IF_gap)),
            "sandwich": "linear_QR_raw_X",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_CSV, index=False)
    summary = summarize(out)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print("Wrote", RESULTS_CSV)
    print("Wrote", SUMMARY_CSV)

    # update config note
    if CONFIG_JSON.exists():
        cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    else:
        cfg = {}
    cfg["sandwich"] = "linear_QR_raw_X"
    cfg["sandwich_note"] = (
        "With known m0, Exp0 uses ordinary linear median QR of Y-m0(Z) on X; "
        "oracle sandwich/IF use raw X. Residualized X-phi*(Z) is for estimated-m experiments only."
    )
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    NOTE_MD.write_text(
        """# Sandwich correction (2026-09-21)

Theorist review: when \(m_0\) is known, Experiment 0 is ordinary linear quantile regression of \(Y-m_0(Z)\) on \(X\). The oracle sandwich and influence-function score must use **raw \(X\)**, not \(\tilde X=X-\varphi^*(Z)\).

Residualizing is the efficient-score construction for the **semiparametric** case where \(m\) is estimated. Using \(\tilde X\) with known \(m_0\) inflates \(\theta_2\) SEs (under-dispersed \(T_2\), coverage near 100%).

This folder's `replication_results.csv` / `summary.csv` / figures were recomputed with the linear-QR sandwich on raw \(X\). Previous residualized diagnostics were backed up as `run/replication_results_residualized_Xt_BACKUP.csv`.

Experiment 2 (estimated \(m\)) should continue to use \(\tilde X\).
""",
        encoding="utf-8",
    )

    make_figures(out)
    print("Figures refreshed under", FIG_DIR)

    # quick calibration print
    for n, g in out.groupby("n"):
        print(
            f"n={n}: mean_se2={g['oracle_se2'].mean():.4f} mc_sd2={g['theta2_error'].std(ddof=1):.4f} "
            f"T2_sd={g['T2'].std(ddof=1):.3f} cov2={g['coverage2'].mean():.3f} | "
            f"mean_se1={g['oracle_se1'].mean():.4f} mc_sd1={g['theta1_error'].std(ddof=1):.4f} "
            f"T1_sd={g['T1'].std(ddof=1):.3f} cov1={g['coverage1'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
