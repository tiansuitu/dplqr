"""Case-3 Psi_n / IF-expansion diagnostics (Zhong-Wang Thm 3.3 proof pieces).

Read-only w.r.t. existing Case-3 artifacts under run/.
New outputs go only under run_psi_n_if/.

Theorist formula list (homo DPLQR; oracle phi*, f0 = t3 density at 0):
  Xtilde = X - phi*(Z)
  xi = theta_hat - theta0
  h = (m_hat - m0) + xi' phi*
  psi_tau(xi,h) = -{tau - 1{eps - xi'Xtilde - h(Z) < 0}} Xtilde
  Psi_n = P_n psi_tau
  Sigma2 = f0 E[Xtilde Xtilde']
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import norm, t as student_t
from sklearn.preprocessing import StandardScaler
import torch

CASE = 3
TAU = 0.5
THETA_0 = np.array([1.0, -1.0], dtype=float)
BASE_SEED = 20260904
TRAIN_FRACTION = 0.8
NET_NODES = [2, 32]
ORACLE_F0 = float(student_t.pdf(0.0, df=3))
REUSED = (
    "seed_all",
    "nonlinear_truth",
    "generate_covariates",
    "generate_dataset",
    "as_numpy",
    "clip_neural_weights",
    "tensor_pair",
    "dplqr_standard_errors",
)

PRIMARY_MEAN_COLS = [
    "adam_sn_xi_1",
    "adam_C_n_1",
    "adam_Gap_Taylor_1",
    "adam_Gap_EP_1",
    "adam_IF_resid_1",
    "adam_R2_n",
    "prof_sn_xi_1",
    "prof_C_n_1",
    "prof_Gap_Taylor_1",
    "prof_Gap_EP_1",
    "prof_IF_resid_1",
    "prof_R2_n",
]


def locate_experiment(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        direct = parent / "results" / "2026-09-11_asymptotic_normality_case3"
        if (direct / "run" / "raw_replications.csv").is_file():
            return direct
        if parent.name == "2026-09-11_asymptotic_normality_case3" and (
            parent / "run" / "raw_replications.csv"
        ).is_file():
            return parent
    raise FileNotFoundError("Cannot locate 2026-09-11_asymptotic_normality_case3/run")


def load_stack(experiment: Path):
    root = next(p for p in [experiment, *experiment.parents] if (p / "dqAux.py").is_file())
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import dqAux  # noqa: F401

    bridge_path = root / "results" / "2026-09-10-asymptotic-normality" / "source_bridge.py"
    spec = importlib.util.spec_from_file_location("case3_psi_n_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(bridge)
    finally:
        sys.dont_write_bytecode = previous

    nodes = bridge.selected_source()
    science = dict(
        np=np,
        norm=norm,
        random=random,
        torch=torch,
        THETA=THETA_0,
        residual_density_zero=bridge.residual_density_zero,
    )
    module = ast.Module(body=[nodes[name] for name in REUSED], type_ignores=[])
    exec(compile(module, str(bridge.NOTEBOOK), "exec"), science)
    original = SimpleNamespace(**{name: science[name] for name in REUSED})
    return root, bridge, original


def true_phi_star_case3(z_raw, eps=np.finfo(float).eps):
    z_raw = np.asarray(z_raw, dtype=float)
    if z_raw.ndim != 2 or z_raw.shape[1] != 8:
        raise ValueError("Expected raw (n, 8) Z on [0, 2]")
    latent_sum = norm.ppf(np.clip(z_raw / 2.0, eps, 1.0 - eps)).sum(axis=1)
    return np.column_stack(
        (
            norm.cdf(latent_sum / np.sqrt(45.0)),
            2.0 * norm.cdf(latent_sum / np.sqrt(126.0)),
        )
    )


def regenerate(original, n: int, rep: int):
    data_seed = BASE_SEED + CASE * 10_000_000 + n * 1000 + rep
    fit_seed = data_seed + int(round(TAU * 1_000_000))
    rng = np.random.default_rng(data_seed)
    x, z, y, _ = original.generate_dataset(n, 3, rng)
    order = rng.permutation(n)
    n_train = int(TRAIN_FRACTION * n)
    tr, va = order[:n_train], order[n_train:]
    digest = hashlib.sha256(b"".join(a.tobytes() for a in (x, z, y, order))).hexdigest()
    scaler = StandardScaler().fit(z[tr])
    return SimpleNamespace(
        n=n,
        rep=rep,
        data_seed=data_seed,
        fit_seed=fit_seed,
        data_sha256=digest,
        x=x,
        z=z,
        y=y,
        order=order,
        tr=tr,
        va=va,
        scaler=scaler,
        x_train=x[tr],
        z_train=z[tr],
        y_train=y[tr],
        x_val=x[va],
        z_val=z[va],
        y_val=y[va],
    )


def replication_dir(experiment: Path, n: int, epochs: int, rep: int) -> Path:
    return experiment / "run" / "replications" / f"n_{n}_epochs_{epochs}_rep_{rep:06d}"


def load_main_net(state_dict):
    import dqAux

    net = dqAux.dqNetSparse(
        2,
        8,
        torch.zeros((1, 2), dtype=torch.float32),
        list(NET_NODES),
        sparseRatio=0.5,
    )
    net.load_state_dict(state_dict)
    net.eval()
    return net


def predict_train(net, data):
    xt = torch.tensor(data.x_train, dtype=torch.float32)
    zt = torch.tensor(data.scaler.transform(data.z_train), dtype=torch.float32)
    with torch.no_grad():
        pred = net(xt, zt).cpu().numpy().reshape(-1)
    theta = net.linLinear.weight.detach().cpu().numpy().reshape(-1).astype(float)
    m_hat = pred - data.x_train @ theta
    return theta, m_hat, pred


def predict_m_on_z(net, data, z_raw):
    """m_hat(Z) via X=0 forward pass (net = X'theta + m(Z))."""
    z_raw = np.asarray(z_raw, dtype=float)
    n = len(z_raw)
    xt = torch.zeros((n, 2), dtype=torch.float32)
    zt = torch.tensor(data.scaler.transform(z_raw), dtype=torch.float32)
    with torch.no_grad():
        m_hat = net(xt, zt).cpu().numpy().reshape(-1)
    return m_hat


def mean_psi(residuals, x_tilde, tau=TAU):
    marks = (residuals < 0).astype(float)
    return (-(tau - marks)[:, None] * x_tilde).mean(axis=0)


def check_linprog(offset_residuals, design, tau=TAU):
    """argmin_b P_n rho_tau(offset - design b)."""
    n, p = design.shape
    c = np.concatenate([np.full(n, tau), np.full(n, 1.0 - tau), np.zeros(p)])
    eye = sparse.eye(n, format="csc")
    A = sparse.hstack([eye, -eye, sparse.csc_matrix(design)], format="csc")
    bounds = [(0, None)] * (2 * n) + [(None, None)] * p
    result = linprog(c, A_eq=A, b_eq=offset_residuals, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"linprog failed: {result.message}")
    return result.x[-p:].astype(float)


def profile_theta(y, x, m_hat, tau=TAU):
    return check_linprog(y - m_hat, x, tau=tau)


def frozen_h_adjustment(residuals, x_tilde, tau=TAU):
    return check_linprog(residuals, x_tilde, tau=tau)


def independent_eval_sample(original, n_eval: int, seed: int):
    rng = np.random.default_rng(seed)
    x, z, y, _ = original.generate_dataset(n_eval, 3, rng)
    return x, z, y


def fill_terms(prefix, xi, m_hat_train, data, original, net, n_eval, eval_seed):
    """Items 1-9 for one (xi, m_hat); includes Gap_EP / Gap_Taylor via eval sample."""
    phi_tr = true_phi_star_case3(data.z_train)
    x_tilde = data.x_train - phi_tr
    m0 = original.nonlinear_truth(data.z_train, 3)
    eps = data.y_train - data.x_train @ THETA_0 - m0
    h = (m_hat_train - m0) + phi_tr @ xi

    n_train = len(data.y_train)
    root_n = math.sqrt(n_train)

    sn_psi_00 = root_n * mean_psi(eps, x_tilde)
    r_fit = eps - x_tilde @ xi - h
    sn_psi_fit = root_n * mean_psi(r_fit, x_tilde)

    sigma2 = ORACLE_F0 * (x_tilde.T @ x_tilde) / n_train
    sn_xi = root_n * xi
    sigma2_sn_xi = sigma2 @ sn_xi

    # independent eval for Psi0_N(xi, h)
    x_e, z_e, y_e = independent_eval_sample(original, n_eval, eval_seed)
    m_hat_e = predict_m_on_z(net, data, z_e)
    phi_e = true_phi_star_case3(z_e)
    x_tilde_e = x_e - phi_e
    m0_e = original.nonlinear_truth(z_e, 3)
    eps_e = y_e - x_e @ THETA_0 - m0_e
    h_e = (m_hat_e - m0_e) + phi_e @ xi
    r_e = eps_e - x_tilde_e @ xi - h_e
    sn_psi0_N = root_n * mean_psi(r_e, x_tilde_e)

    c_n = root_n * ORACLE_F0 * (x_tilde * (m_hat_train - m0)[:, None]).mean(axis=0)
    r2_n = root_n * float(np.mean((m_hat_train - m0) ** 2))
    if_resid = sigma2_sn_xi + sn_psi_00
    if_resid_2s = 2.0 * sigma2_sn_xi + sn_psi_00
    a_hat = frozen_h_adjustment(r_fit, x_tilde)

    gap_ep = sn_psi0_N + sn_psi_00
    gap_taylor = sn_psi0_N - sigma2_sn_xi

    p = f"{prefix}_"
    return {
        f"{p}sn_xi_1": float(sn_xi[0]),
        f"{p}sn_xi_2": float(sn_xi[1]),
        f"{p}sn_psi_00_1": float(sn_psi_00[0]),
        f"{p}sn_psi_00_2": float(sn_psi_00[1]),
        f"{p}sn_psi_fit_1": float(sn_psi_fit[0]),
        f"{p}sn_psi_fit_2": float(sn_psi_fit[1]),
        f"{p}sn_psi0_N_1": float(sn_psi0_N[0]),
        f"{p}sn_psi0_N_2": float(sn_psi0_N[1]),
        f"{p}sigma2_sn_xi_1": float(sigma2_sn_xi[0]),
        f"{p}sigma2_sn_xi_2": float(sigma2_sn_xi[1]),
        f"{p}Gap_EP_1": float(gap_ep[0]),
        f"{p}Gap_EP_2": float(gap_ep[1]),
        f"{p}Gap_Taylor_1": float(gap_taylor[0]),
        f"{p}Gap_Taylor_2": float(gap_taylor[1]),
        f"{p}C_n_1": float(c_n[0]),
        f"{p}C_n_2": float(c_n[1]),
        f"{p}R2_n": float(r2_n),
        f"{p}IF_resid_1": float(if_resid[0]),
        f"{p}IF_resid_2": float(if_resid[1]),
        f"{p}IF_resid_2Sigma_1": float(if_resid_2s[0]),
        f"{p}IF_resid_2Sigma_2": float(if_resid_2s[1]),
        f"{p}a_hat_1": float(a_hat[0]),
        f"{p}a_hat_2": float(a_hat[1]),
        f"{p}sn_a_1": float(root_n * a_hat[0]),
        f"{p}sn_a_2": float(root_n * a_hat[1]),
        f"{p}m_l2": float(np.sqrt(np.mean((m_hat_train - m0) ** 2))),
        f"{p}h_l2": float(np.sqrt(np.mean(h ** 2))),
        f"{p}Sigma2_11": float(sigma2[0, 0]),
        f"{p}Sigma2_12": float(sigma2[0, 1]),
        f"{p}Sigma2_22": float(sigma2[1, 1]),
        f"{p}theta_1": float((THETA_0 + xi)[0]),
        f"{p}theta_2": float((THETA_0 + xi)[1]),
    }


def diagnose_one(original, experiment: Path, n: int, epochs: int, rep: int, n_eval: int = 50_000) -> dict:
    path = replication_dir(experiment, n, epochs, rep)
    fits = torch.load(path / "fits.pt", map_location="cpu", weights_only=False)
    row = json.loads((path / "result.json").read_text(encoding="utf-8"))["row"]
    data = regenerate(original, n, rep)

    if data.data_sha256 != fits["data_sha256"] or data.data_sha256 != row["data_sha256"]:
        raise ValueError(f"data hash mismatch at {path}")
    if fits["setting"] != {"n": n, "epochs": epochs, "rep": rep}:
        raise ValueError(f"setting mismatch at {path}")

    net = load_main_net(fits["stages"]["main"]["state"])
    theta_adam, m_hat, _ = predict_train(net, data)
    if not np.allclose(theta_adam, [row["theta_hat_1"], row["theta_hat_2"]], atol=1e-5, rtol=0):
        raise ValueError("theta from fits.pt does not match result.json")

    eval_seed = BASE_SEED + 17 * n + 1009 * epochs + 13 * rep
    out = dict(n=n, epochs=epochs, rep=rep, n_train=len(data.y_train), n_eval=n_eval)
    out["oracle_f0"] = ORACLE_F0
    out["data_sha256"] = data.data_sha256

    xi_adam = theta_adam - THETA_0
    out.update(fill_terms("adam", xi_adam, m_hat, data, original, net, n_eval, eval_seed))

    theta_prof = profile_theta(data.y_train, data.x_train, m_hat)
    xi_prof = theta_prof - THETA_0
    out.update(fill_terms("prof", xi_prof, m_hat, data, original, net, n_eval, eval_seed + 1))
    return out


def list_available(experiment: Path):
    rows = []
    root = experiment / "run" / "replications"
    for path in sorted(root.glob("n_*_epochs_*_rep_*")):
        parts = path.name.split("_")
        n, epochs, rep = int(parts[1]), int(parts[3]), int(parts[5])
        if (path / "fits.pt").is_file() and (path / "result.json").is_file():
            rows.append((n, epochs, rep))
    return rows


def analyze_grid(
    experiment: Path,
    n_values=None,
    epoch_values=None,
    max_reps=None,
    reps=None,
    n_eval: int = 50_000,
):
    _root, _bridge, original = load_stack(experiment)
    selected = []
    for n, epochs, rep in list_available(experiment):
        if n_values is not None and n not in n_values:
            continue
        if epoch_values is not None and epochs not in epoch_values:
            continue
        if reps is not None and rep not in reps:
            continue
        selected.append((n, epochs, rep))
    selected.sort()
    if max_reps is not None:
        kept, counts = [], {}
        for key in selected:
            nk = (key[0], key[1])
            counts[nk] = counts.get(nk, 0) + 1
            if counts[nk] <= max_reps:
                kept.append(key)
        selected = kept
    records = [
        diagnose_one(original, experiment, n, epochs, rep, n_eval=n_eval)
        for n, epochs, rep in selected
    ]
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (n, epochs), group in df.groupby(["n", "epochs"]):
        q = len(group)
        item = dict(n=int(n), epochs=int(epochs), Q=q, n_train=int(group.n_train.iloc[0]))
        for col in PRIMARY_MEAN_COLS:
            if col not in group.columns:
                continue
            mean = float(group[col].mean())
            se = float(group[col].std(ddof=1) / math.sqrt(q)) if q > 1 else float("nan")
            item[f"mean_{col}"] = mean
            item[f"mcse_{col}"] = se
            item[f"mean_abs_{col}"] = float(group[col].abs().mean())
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["epochs", "n"]).reset_index(drop=True)


def plot_primary(summary: pd.DataFrame, out_dir: Path, epochs: int = 1000):
    import matplotlib.pyplot as plt

    sub = summary[summary.epochs == epochs].sort_values("n")
    if sub.empty:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = [
        ("mean_adam_sn_xi_1", "mcse_adam_sn_xi_1", r"Adam $\sqrt{n}\xi_1$", "adam_sn_xi1.png"),
        ("mean_adam_C_n_1", "mcse_adam_C_n_1", r"Adam $C_{n,1}$", "adam_Cn1.png"),
        ("mean_adam_Gap_Taylor_1", "mcse_adam_Gap_Taylor_1", r"Adam Gap_Taylor$_1$", "adam_GapTaylor1.png"),
        ("mean_adam_Gap_EP_1", "mcse_adam_Gap_EP_1", r"Adam Gap_EP$_1$", "adam_GapEP1.png"),
        ("mean_adam_IF_resid_1", "mcse_adam_IF_resid_1", r"Adam IF_resid$_1$", "adam_IFresid1.png"),
        ("mean_adam_R2_n", "mcse_adam_R2_n", r"Adam $R2_n$", "adam_R2n.png"),
        ("mean_prof_sn_xi_1", "mcse_prof_sn_xi_1", r"Prof $\sqrt{n}\xi_1$", "prof_sn_xi1.png"),
        ("mean_prof_C_n_1", "mcse_prof_C_n_1", r"Prof $C_{n,1}$", "prof_Cn1.png"),
        ("mean_prof_Gap_Taylor_1", "mcse_prof_Gap_Taylor_1", r"Prof Gap_Taylor$_1$", "prof_GapTaylor1.png"),
        ("mean_prof_IF_resid_1", "mcse_prof_IF_resid_1", r"Prof IF_resid$_1$", "prof_IFresid1.png"),
    ]
    for y, yerr, ylabel, fname in specs:
        if y not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.errorbar(sub.n, sub[y], yerr=sub[yerr], marker="o", capsize=3)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Case 3, epochs={epochs}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths

# --- notebook API aliases ---
# list_available already defined
# analyze_grid already defined
# summarize already defined
# plot_primary already defined
# ORACLE_F0 already defined
