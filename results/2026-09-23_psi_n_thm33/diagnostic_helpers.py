"""Reproducible score, expansion, optimization, and Monte Carlo diagnostics.

Only the stated homoskedastic Gaussian median-regression DGP is supported.
Population integration analytically eliminates epsilon and V; randomized Sobol
integration is used only for the remaining two-dimensional Z expectation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import norm, qmc, t
import torch
from torch import nn

SCHEMA = 2
THETA0 = 1.0
TAU = 0.5
SIGMA_V = 0.5
J = SIGMA_V**2 * norm.pdf(0.0)
ASYMPTOTIC_SD = math.sqrt(TAU * (1 - TAU) * SIGMA_V**2) / J


def make_config(profile="pilot", width_mode="fixed"):
    profiles = {
        "smoke": dict(n_values=[32, 48], reps=2, baseline_epochs=2,
                      polish_epochs=3, integration_power=5, integration_scrambles=3),
        "pilot": dict(n_values=[500, 1000, 2000], reps=10, baseline_epochs=500,
                      polish_epochs=500, integration_power=12, integration_scrambles=8),
        "main": dict(n_values=[500, 1000, 2000, 4000], reps=100, baseline_epochs=500,
                     polish_epochs=1000, integration_power=13, integration_scrambles=8),
    }
    if profile not in profiles or width_mode not in {"fixed", "growing"}:
        raise ValueError("Unknown profile or width_mode")
    return dict(schema=SCHEMA, profile=profile, **profiles[profile],
                width_mode=width_mode, hidden=[16, 16], base_seed=20260923,
                integration_seed=817331, batch_size=128, learning_rate=1e-3,
                polish_lr_start=1e-4, polish_lr_end=1e-6, torch_threads=1,
                integration_score_se_tolerance=0.002,
                score_thresholds=[0.02, 0.05, 0.1],
                theta_thresholds=[0.1, 0.25, 0.5], bias_tolerance=0.25,
                dgp="Z=Uniform([0,1]^2); V=N(0,0.25); epsilon=N(0,1); independent",
                theta0=THETA0, tau=TAU, sigma_v=SIGMA_V)


def validate_config(c):
    if (c["schema"], c["theta0"], c["tau"], c["sigma_v"]) != (SCHEMA, THETA0, TAU, SIGMA_V):
        raise ValueError("Population formulas require the documented Gaussian median DGP")
    if c["reps"] < 2 or c["integration_scrambles"] < 2:
        raise ValueError("At least two replications and two independent scrambles are required")
    if not c["n_values"] or len(set(c["n_values"])) != len(c["n_values"]) or min(c["n_values"]) < 2:
        raise ValueError("n_values must be distinct integers >= 2")
    if min(c["baseline_epochs"], c["polish_epochs"], c["batch_size"], c["torch_threads"]) < 1:
        raise ValueError("Epochs, batch size, and thread count must be positive")
    if c["integration_power"] < 2 or c["width_mode"] not in {"fixed", "growing"}:
        raise ValueError("Invalid integration power or width mode")
    if not 0 < c["polish_lr_end"] <= c["polish_lr_start"] or c["learning_rate"] <= 0:
        raise ValueError("Invalid learning rates")


def seed_for(c, n, rep):
    return int(c["base_seed"] + 1_000_003 * n + 1_009 * rep)


def phi_star(z):
    z1, z2 = np.asarray(z).T
    return 0.5 + 0.3 * np.sin(2 * np.pi * z1) + 0.3 * z1 * z2


def m0_fn(z):
    z1, z2 = np.asarray(z).T
    return (0.5 * np.sin(2 * np.pi * z1) + 0.5 * (z2 - 0.5)**2
            + 0.3 * np.cos(2 * np.pi * z1 * z2))


def generate_sample(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1, (n, 2))
    phi = phi_star(z)
    v = rng.normal(0, SIGMA_V, n)
    m0 = m0_fn(z)
    eps = rng.normal(0, 1, n)
    x = phi + v
    return dict(Z=z, phi=phi, V=v, X=x, m0=m0, epsilon=eps,
                Y=THETA0 * x + m0 + eps)


def check_loss(r):
    return float(0.5 * np.mean(np.abs(r)))


def frozen_h_adjustment(r, v):
    n = len(r)
    matrix = sparse.hstack([sparse.csr_matrix(v.reshape(-1, 1)),
                            sparse.eye(n), -sparse.eye(n)], format="csr")
    fit = linprog(np.r_[0.0, np.full(2 * n, 0.5)], A_eq=matrix, b_eq=r,
                  bounds=[(None, None)] + [(0, None)] * (2 * n), method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    return float(fit.x[0])


def hidden_for(c, n):
    # Exploratory size sensitivity, not a claim of satisfying every sieve assumption.
    factor = max(1.0, (n / 500)**0.25) if c["width_mode"] == "growing" else 1.0
    return [int(math.ceil(h * factor)) for h in c["hidden"]]


class JointModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        layers = []
        previous = 2
        for width in hidden:
            layers.extend([nn.Linear(previous, width, dtype=torch.float64), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1, dtype=torch.float64))
        self.m_net = nn.Sequential(*layers)
        self.double()

    def forward(self, x, z):
        return self.theta * x + self.m_net(z).squeeze(-1)


def predict_m(model, z):
    with torch.no_grad():
        return np.concatenate([model.m_net(torch.as_tensor(chunk, dtype=torch.float64))
                               .squeeze(-1).numpy()
                               for chunk in np.array_split(z, max(1, math.ceil(len(z) / 16384)))])


def fit_stages(data, c, seed):
    """Baseline final iterate, then best empirical-loss full-batch polishing iterate."""
    torch.set_num_threads(c["torch_threads"])
    torch.manual_seed(seed)
    n = len(data["Y"])
    model = JointModel(hidden_for(c, n))
    x, z, y = [torch.as_tensor(data[k], dtype=torch.float64) for k in ["X", "Z", "Y"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    rng = np.random.default_rng(seed)
    for _ in range(c["baseline_epochs"]):
        perm = rng.permutation(n)
        for start in range(0, n, c["batch_size"]):
            idx = perm[start:start+c["batch_size"]]
            objective = 0.5 * torch.mean(torch.abs(y[idx] - model(x[idx], z[idx])))
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
    model.eval()
    baseline = copy.deepcopy(model)
    with torch.no_grad():
        best_loss = float(0.5 * torch.mean(torch.abs(y - model(x, z))))
    best_state = copy.deepcopy(model.state_dict())
    selected_epoch = 0  # Baseline is included as a candidate.
    # Fresh optimizer; the refinement protocol is explicitly distinct from the baseline.
    optimizer = torch.optim.Adam(model.parameters(), lr=c["polish_lr_start"])
    for epoch in range(1, c["polish_epochs"] + 1):
        fraction = (epoch - 1) / max(c["polish_epochs"] - 1, 1)
        lr = c["polish_lr_start"] * (c["polish_lr_end"] / c["polish_lr_start"])**fraction
        for group in optimizer.param_groups:
            group["lr"] = lr
        objective = 0.5 * torch.mean(torch.abs(y - model(x, z)))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            actual_loss = float(0.5 * torch.mean(torch.abs(y - model(x, z))))
        if actual_loss < best_loss:
            best_loss, selected_epoch = actual_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return [("baseline", baseline, 0), ("polished", model, selected_epoch)]


def population_integrand(delta, h):
    """E[V {Phi(delta V+h)-1/2} | Z], with epsilon,V integrated exactly."""
    scale = math.sqrt(1 + SIGMA_V**2 * delta**2)
    return SIGMA_V**2 * delta / scale * norm.pdf(np.asarray(h) / scale)


def population_metrics(model, delta, c, integration_seed):
    values = []
    coarse_scores = []
    for scramble in range(c["integration_scrambles"]):
        seed = np.random.SeedSequence([integration_seed, scramble])
        z = qmc.Sobol(2, scramble=True, seed=np.random.default_rng(seed)).random_base2(c["integration_power"])
        error_m = predict_m(model, z) - m0_fn(z)
        h = error_m + delta * phi_star(z)
        integrand = population_integrand(delta, h)
        values.append([integrand.mean(), np.mean(h*h), np.mean(error_m*error_m)])
        coarse_scores.append(integrand[:len(z)//2].mean())
    values = np.asarray(values)
    means = values.mean(axis=0)
    ses = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    return dict(population_score=float(means[0]), population_score_se=float(ses[0]),
                population_score_refinement_gap=float(means[0] - np.mean(coarse_scores)),
                h_l2_squared=float(means[1]), h_l2_squared_se=float(ses[1]),
                m_l2_squared=float(means[2]), m_l2_squared_se=float(ses[2]))


def diagnose(model, data, c, n, rep, stage, selected_epoch):
    theta = float(model.theta.detach())
    delta = theta - THETA0
    root_n = math.sqrt(n)
    fitted_m = predict_m(model, data["Z"])
    r = data["Y"] - theta * data["X"] - fitted_m
    v = data["V"]
    h_train = fitted_m - data["m0"] + delta * data["phi"]
    assert np.allclose(r, data["epsilon"] - delta*v - h_train)
    signs = TAU - (r < 0)
    psi = -np.mean(signs * v)
    g = -np.mean(signs * data["X"])
    phi_score = np.mean(signs * data["phi"])
    assert abs(psi-g-phi_score) < 1e-12
    s = root_n * psi
    w = -root_n * np.mean((TAU - (data["epsilon"] < 0)) * v)
    a = frozen_h_adjustment(r, v)
    pop = population_metrics(model, delta, c, c["integration_seed"] + seed_for(c, n, rep))
    scaled_pop = root_n * pop["population_score"]
    empirical = s - w - scaled_pop
    taylor = scaled_pop - J * root_n * delta
    linear = J * root_n * delta + w
    loss0, loss_a = check_loss(r), check_loss(r-a*v)
    if loss_a > loss0 + 1e-8:
        raise AssertionError("LP increased the stated frozen-h objective")
    score_se = root_n * pop["population_score_se"]
    score_gap = root_n * abs(pop["population_score_refinement_gap"])
    row = dict(n=n, rep=rep, stage=stage, seed=seed_for(c, n, rep), theta_hat=theta,
               theta_error=delta, root_n_theta_error=root_n*delta,
               S=s, E=empirical, T=taylor, W=w, R_score=linear, R_theta=linear/J,
               Psi_hat=float(psi), root_n_G=root_n*g, root_n_phi_score=root_n*phi_score,
               a_hat=a, root_n_a=root_n*a, final_training_loss=loss0,
               frozen_h_loss_gain=loss0-loss_a, n_frozen_h_loss_gain=n*(loss0-loss_a),
               exact_zero_residual_count=int(np.sum(r == 0)),
               near_zero_residual_count=int(np.sum(np.abs(r) <= 1e-8)),
               root_n_tie_score_bound=root_n*np.sum(np.abs(v[r == 0]))/n,
               selected_polish_epoch=selected_epoch,
               scaled_population_integration_se=score_se,
               scaled_population_refinement_gap=score_gap,
               integration_adequate=bool(max(3*score_se, score_gap) <= c["integration_score_se_tolerance"]),
               decomposition_error=linear-(s-empirical-taylor), **pop)
    row["sqrt_n_prediction_error_squared"] = root_n*(SIGMA_V**2*delta**2 + pop["h_l2_squared"])
    row["n_quarter_h_l2"] = n**0.25 * math.sqrt(pop["h_l2_squared"])
    row["n_quarter_m_l2"] = n**0.25 * math.sqrt(pop["m_l2_squared"])
    return row


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def run_identity(c):
    validate_config(c)
    identity = dict(config=c, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__, pandas=pd.__version__, torch=str(torch.__version__)))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return identity, digest


def run_directory(base, c):
    _, digest = run_identity(c)
    return Path(base) / "run_expansion_v2" / f"{c['profile']}_{digest}"


def initialize_run(base, c):
    identity, _ = run_identity(c)
    folder = run_directory(base, c)
    manifest = folder / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Run manifest mismatch; refusing to combine incompatible results")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, identity)
    return folder


def result_path(folder, n, rep):
    return Path(folder) / "replications" / f"n_{n}_rep_{rep:04d}" / "result.json"


def read_pair(path, n, rep):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(rows) != 2 or {r["stage"] for r in rows} != {"baseline", "polished"}:
        raise ValueError(f"Incomplete stage pair: {path}")
    if any(r["n"] != n or r["rep"] != rep for r in rows):
        raise ValueError(f"Replication key mismatch: {path}")
    for row in rows:
        if not (Path(path).parent / f"{row['stage']}.pt").exists():
            raise ValueError(f"Missing model checkpoint for {path}")
    return rows


def collect_results(folder, c):
    rows = []
    for n in c["n_values"]:
        for rep in range(1, c["reps"]+1):
            path = result_path(folder, n, rep)
            if path.exists():
                rows.extend(read_pair(path, n, rep))
    return pd.DataFrame(rows)


def run_experiment(base, c):
    folder = initialize_run(base, c)
    for n in c["n_values"]:
        for rep in range(1, c["reps"]+1):
            path = result_path(folder, n, rep)
            if path.exists():
                read_pair(path, n, rep)
                continue
            seed = seed_for(c, n, rep)
            data = generate_sample(n, seed)
            rows = []
            for stage, model, selected_epoch in fit_stages(data, c, seed):
                row = diagnose(model, data, c, n, rep, stage, selected_epoch)
                rows.append(row)
                path.parent.mkdir(parents=True, exist_ok=True)
                destination = path.parent / f"{stage}.pt"
                temp = destination.with_suffix(".pt.tmp")
                torch.save(dict(state_dict=model.state_dict(), hidden=hidden_for(c, n),
                                n=n, rep=rep, stage=stage, seed=seed), temp)
                os.replace(temp, destination)
            atomic_json(path, rows)  # Completion only after both diagnostic rows and checkpoints.
            print(f"n={n} rep={rep}/{c['reps']}  |S| {abs(rows[0]['S']):.4f} -> {abs(rows[1]['S']):.4f}", flush=True)
    results = collect_results(folder, c)
    results.to_csv(folder / "replication_results.csv", index=False)
    return folder, results


def mean_interval(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite Monte Carlo observations")
    mean = float(x.mean())
    if len(x) == 1:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    margin = float(t.ppf(0.975, len(x)-1) * se)
    return mean, se, mean-margin, mean+margin


def wilson_interval(successes, count):
    z = norm.ppf(0.975)
    p = successes/count
    den = 1 + z*z/count
    mid = (p + z*z/(2*count))/den
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count**2))/den
    return p, max(0.0, mid-half), min(1.0, mid+half)


def bias_summary(df, planned_reps=None):
    rows = []
    for (n, stage), g in df.groupby(["n", "stage"]):
        x = np.sqrt(n)*g["theta_error"].to_numpy()
        bias, se, lo, hi = mean_interval(x)
        coverage, cover_lo, cover_hi = wilson_interval(int((np.abs(x) <= norm.ppf(.975)*ASYMPTOTIC_SD).sum()), len(x))
        rows.append(dict(n=n, stage=stage, completed_reps=len(g), planned_reps=planned_reps,
                         root_n_bias=bias, mc_se=se, ci_low=lo, ci_high=hi,
                         root_n_sd=float(np.std(x, ddof=1)), root_n_mae=float(np.mean(np.abs(x))),
                         root_n_rmse=float(np.sqrt(np.mean(x*x))),
                         oracle_wald_coverage=coverage, coverage_ci_low=cover_lo, coverage_ci_high=cover_hi))
    return pd.DataFrame(rows)


def legacy_diagnostics(base):
    """Recover truth scores and total expansion residuals, never the missing fitted m."""
    folder = Path(base) / "run"
    if not (folder / "replication_results.csv").exists():
        return pd.DataFrame()
    old_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    expected_dgp = {
        "Z": "Uniform(0,1)^2",
        "phi_star": "0.5 + 0.3*sin(2*pi*Z1) + 0.3*Z1*Z2",
        "X": "phi_star(Z) + N(0,0.5^2)",
        "m0": "0.5*sin(2*pi*Z1) + 0.5*(Z2-0.5)^2 + 0.3*cos(2*pi*Z1*Z2)",
        "epsilon": "N(0,1)", "Y": "theta0*X + m0(Z) + epsilon",
    }
    if old_config["DGP"] != expected_dgp or old_config["TAU"] != TAU or old_config["THETA0"] != THETA0:
        raise ValueError("Legacy DGP differs; cannot reconstruct truth scores")
    df = pd.read_csv(folder / "replication_results.csv")
    if df.duplicated(["n", "rep"]).any():
        raise ValueError("Duplicate legacy rows")
    if not np.allclose(df.theta_error, df.theta_hat-THETA0):
        raise ValueError("Inconsistent legacy theta errors")
    df["stage"] = "legacy_baseline"
    w = []
    for row in df.itertuples():
        expected_seed = old_config["BASE_SEED"] + 1_000_003*row.n + 1_009*row.rep
        if int(row.seed) != expected_seed:
            raise ValueError("Legacy seed mismatch")
        data = generate_sample(int(row.n), int(row.seed))
        w.append(-math.sqrt(row.n)*np.mean((TAU-(data["epsilon"]<0))*data["V"]))
    df["W"] = w
    df["S"] = df.Sn
    df["R_score"] = J*np.sqrt(df.n)*df.theta_error + df.W
    df["R_theta"] = df.R_score/J
    df["E_plus_T"] = df.S-df.R_score
    return df


def summarize(df, c):
    if df.empty:
        return {}
    if df.duplicated(["n", "rep", "stage"]).any():
        raise ValueError("Duplicate replication/stage keys")
    metrics = ["S", "E", "T", "R_score", "R_theta", "W", "root_n_G", "root_n_phi_score",
               "root_n_a", "sqrt_n_prediction_error_squared", "n_quarter_h_l2", "n_quarter_m_l2",
               "n_frozen_h_loss_gain"]
    summaries, tails, integration, paired = [], [], [], []
    for (n, stage), g in df.groupby(["n", "stage"]):
        for metric in metrics:
            x = g[metric].to_numpy()
            signed, signed_se, _, _ = mean_interval(x)
            absolute, se, lo, hi = mean_interval(np.abs(x))
            summaries.append(dict(n=n, stage=stage, metric=metric, completed_reps=len(x),
                                  planned_reps=c["reps"], signed_mean=signed, signed_mc_se=signed_se,
                                  mean_abs=absolute, mean_abs_mc_se=se, mean_abs_ci_low=max(0, lo),
                                  mean_abs_ci_high=hi, median_abs=float(np.median(np.abs(x))),
                                  q90_abs=float(np.quantile(np.abs(x), .9)), rms=float(np.sqrt(np.mean(x*x)))))
        for metric in ["S", "E", "T", "R_score", "R_theta"]:
            thresholds = c["theta_thresholds"] if metric == "R_theta" else c["score_thresholds"]
            for epsilon in thresholds:
                probability, lo, hi = wilson_interval(int((g[metric].abs()>epsilon).sum()), len(g))
                tails.append(dict(n=n, stage=stage, metric=metric, epsilon=epsilon,
                                  probability=probability, ci_low=lo, ci_high=hi, completed_reps=len(g)))
        integration.append(dict(n=n, stage=stage, completed_reps=len(g),
                                inadequate_integrations=int((~g.integration_adequate.astype(bool)).sum()),
                                max_scaled_se=g.scaled_population_integration_se.max(),
                                max_scaled_refinement_gap=g.scaled_population_refinement_gap.max(),
                                max_decomposition_error=g.decomposition_error.abs().max()))
    for n, g in df.groupby("n"):
        for metric in ["S", "E", "T", "R_theta", "root_n_G", "root_n_a", "final_training_loss"]:
            pivot = g.pivot(index="rep", columns="stage", values=metric)
            if pivot.isna().any().any() or set(pivot.columns) != {"baseline", "polished"}:
                raise ValueError("Unpaired optimizer comparison")
            delta = pivot.polished-pivot.baseline if metric == "final_training_loss" else pivot.polished.abs()-pivot.baseline.abs()
            mean, se, lo, hi = mean_interval(delta)
            paired.append(dict(n=n, metric=metric, pairs=len(delta), mean_change=mean, mc_se=se,
                               ci_low=lo, ci_high=hi, fraction_improved=float((delta<0).mean())))
    return dict(bias=bias_summary(df, c["reps"]), terms=pd.DataFrame(summaries),
                tails=pd.DataFrame(tails), integration=pd.DataFrame(integration), paired=pd.DataFrame(paired))


def save_summaries(folder, tables):
    for name, table in tables.items():
        table.to_csv(Path(folder) / f"{name}_summary.csv", index=False)


def plot_diagnostics(tables, c, folder):
    import matplotlib.pyplot as plt
    folder = Path(folder) / "figures"
    folder.mkdir(exist_ok=True, parents=True)
    paths = []

    def finish(fig, name):
        fig.tight_layout()
        path = folder / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for stage, g in tables["bias"].groupby("stage"):
        g = g.sort_values("n")
        axes[0].errorbar(g.n, g.root_n_bias, yerr=[g.root_n_bias-g.ci_low, g.ci_high-g.root_n_bias], marker="o", capsize=4, label=stage)
        axes[1].plot(g.n, g.root_n_sd, "o-", label=stage)
    axes[0].axhspan(-c["bias_tolerance"], c["bias_tolerance"], color="gray", alpha=.15, label="chosen bias tolerance")
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(title="Scaled signed bias: 95% Monte Carlo intervals", ylabel=r"$\sqrt{n}\,\widehat{E}(\hat\theta-\theta_0)$")
    axes[1].axhline(ASYMPTOTIC_SD, color="black", linestyle="--", label=r"$\sqrt{2\pi}$ benchmark")
    axes[1].set(title="Root-n sampling SD (should not vanish)", ylabel=r"$\mathrm{SD}[\sqrt{n}(\hat\theta-\theta_0)]$")
    for ax in axes:
        ax.set_xlabel("n"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    finish(fig, "bias_and_sampling_sd.png")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, ["S", "E", "T", "R_score", "R_theta", "root_n_a"]):
        selected = tables["terms"].query("metric == @metric")
        for stage, g in selected.groupby("stage"):
            g = g.sort_values("n")
            ax.errorbar(g.n, g.mean_abs, yerr=[g.mean_abs-g.mean_abs_ci_low, g.mean_abs_ci_high-g.mean_abs], marker="o", capsize=3, label=stage)
        ax.set(title=metric + ": mean absolute value", xlabel="n")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    finish(fig, "expansion_terms.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric in zip(axes, ["S", "E", "R_theta"]):
        eps = c["theta_thresholds"][1] if metric == "R_theta" else c["score_thresholds"][1]
        for stage, g in tables["tails"].query("metric == @metric and epsilon == @eps").groupby("stage"):
            g = g.sort_values("n")
            ax.errorbar(g.n, g.probability, yerr=[g.probability-g.ci_low, g.ci_high-g.probability], marker="o", capsize=3, label=stage)
        ax.set(title=f"P(|{metric}| > {eps}): Wilson intervals", xlabel="n", ylim=(-.02, 1.02))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    finish(fig, "exceedance_probabilities.png")
    return paths
