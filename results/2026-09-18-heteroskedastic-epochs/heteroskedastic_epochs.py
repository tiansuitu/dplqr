"""DPLQR-only heteroskedasticity x epoch-drift experiment.

Recycles Case 6 (2026-09-15) hyperparameters / continuous-trajectory idea,
Simulation I/II covariates + deep m(.) from the Sept 4 notebook, and
networks/loss from repo-root dqAux.py.

Question: which stronger-than-paper heteroskedastic scales / error laws make
theta_hat drift from theta_0 as epochs grow, even at large n?

DPLQR only -- no LQR / PLAQR.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from scipy.stats import norm, t as student_t
from sklearn.preprocessing import StandardScaler
from torchtuples import Model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dqAux import checkLoss, dqNetSparse  # noqa: E402

THETA0 = np.array([1.0, -1.0], dtype=float)
TAU = 0.5
HP = dict(depth=2, width=32, batch_size=128, lr=0.005)
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8
TRAIN_FRACTION = 0.8

# Affine-in-X slopes of base sigma (before scale multiplier).
AFFINE_SLOPES = {
    "paper_case6": np.array([1.0 / 3.0, 1.0 / 3.0]),
    "mild_prior": np.array([0.1, 0.1]),
    "mild_x10": np.array([10.0, 10.0]),
    "paper_case4": np.array([0.2, 0.2]),
    "case4_unscaled": np.array([1.0, 1.0]),
    "case6_amp_phi": np.array([1.0, 1.0]),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def nonlinear_truth(z: np.ndarray, case: int = 3) -> np.ndarray:
    if case != 3:
        raise ValueError("This experiment uses deep case-3 m(.) only.")
    inside = (
        np.exp(z[:, 0] * (1 + z[:, 1] - np.pi * z[:, 2] * z[:, 3]) / 2) * (z[:, 4] + 0.2)
        + z[:, 4] * (z[:, 3] - 0.3) / (np.abs(2 * z[:, 3] - 1) + 1)
        + 2 * np.sin(z[:, 4]) * np.abs(z[:, 4] * z[:, 5] - 0.6)
        + np.log(z[:, 5] + z[:, 6] * z[:, 7])
    )
    return 0.61 * inside


def generate_covariates(n: int, rng: np.random.Generator):
    cov = np.full((10, 10), 0.5)
    np.fill_diagonal(cov, 1.0)
    z_tilde = 2 * norm.cdf(rng.multivariate_normal(np.zeros(10), cov, size=n))
    z = z_tilde[:, :8]
    x = np.column_stack((z_tilde[:, 8] > 1, z_tilde[:, 9])).astype(float)
    return x, z


def sigma_paper_case6(x, z):
    """Simulation II Case 6 (paper)."""
    return (x[:, 0] + x[:, 1]) / 3.0 + 3.0 * norm.cdf(np.sum(z - 1.0, axis=1) / 5.0)


def sigma_mild_prior(x, z):
    """Mild 2026-09-15 formula with /10 on the linear X piece."""
    return (x[:, 0] + x[:, 1]) / 10.0 + norm.cdf(np.mean(z - 1.0, axis=1))


def sigma_mild_x10(x, z):
    """Same mild formula but /10 replaced by *10 on the X piece."""
    return 10.0 * (x[:, 0] + x[:, 1]) + norm.cdf(np.mean(z - 1.0, axis=1))


def sigma_paper_case4(x, z):
    """Simulation II Case 4; paper typo x1+x1 -> x1+x2."""
    return (x[:, 0] + x[:, 1] + z.sum(axis=1)) / 5.0


def sigma_case4_unscaled(x, z):
    """Case 4 without the /5 divisor."""
    return 0.05 + (x[:, 0] + x[:, 1] + z.sum(axis=1))


def sigma_case6_amp_phi(x, z):
    """Case 6 with Phi term amplified and linear X undivided."""
    return (x[:, 0] + x[:, 1]) + 30.0 * norm.cdf(np.sum(z - 1.0, axis=1) / 5.0)


def sigma_steep_x(x, z):
    return 0.25 + np.exp(1.5 * (x[:, 0] + x[:, 1]))


def sigma_steep_x_hard(x, z):
    """Even steeper exponential X-driven scale."""
    return 0.1 + np.exp(3.0 * (x[:, 0] + x[:, 1]))


def sigma_interaction(x, z):
    return 0.2 + np.abs(x[:, 0] + x[:, 1]) * (1.0 + np.abs(z[:, 0] * z[:, 1])) ** 2


def sigma_x_multiplicative(x, z):
    """Strong multiplicative hetero in X."""
    return 0.2 + 8.0 * (1.0 + 4.0 * x[:, 0]) * (0.5 + x[:, 1])


def sigma_z_spike(x, z):
    """Hetero spikes when Z coordinates are jointly large."""
    return 0.2 + 5.0 * np.exp(np.sum(np.maximum(z - 1.2, 0.0), axis=1))


SIGMA = {
    "paper_case6": sigma_paper_case6,
    "mild_prior": sigma_mild_prior,
    "mild_x10": sigma_mild_x10,
    "paper_case4": sigma_paper_case4,
    "case4_unscaled": sigma_case4_unscaled,
    "case6_amp_phi": sigma_case6_amp_phi,
    "steep_x": sigma_steep_x,
    "steep_x_hard": sigma_steep_x_hard,
    "interaction": sigma_interaction,
    "x_multiplicative": sigma_x_multiplicative,
    "z_spike": sigma_z_spike,
}


@dataclass(frozen=True)
class Design:
    name: str
    sigma: str
    scale: float
    error_law: str


def draw_errors(n: int, law: str, rng: np.random.Generator) -> np.ndarray:
    if law == "t3":
        return rng.standard_t(df=3, size=n)
    if law == "t3_unit":
        return rng.standard_t(df=3, size=n) / np.sqrt(3.0)
    if law == "normal":
        return rng.normal(size=n)
    if law == "t5_unit":
        return rng.standard_t(df=5, size=n) / np.sqrt(5.0 / 3.0)
    if law == "laplace":
        return rng.laplace(size=n) / np.sqrt(2.0)
    raise ValueError(law)


def quantile_of_law(tau: float, law: str) -> float:
    if law == "t3":
        return float(student_t.ppf(tau, df=3))
    if law == "t3_unit":
        return float(student_t.ppf(tau, df=3) / np.sqrt(3.0))
    if law == "normal":
        return float(norm.ppf(tau))
    if law == "t5_unit":
        return float(student_t.ppf(tau, df=5) / np.sqrt(5.0 / 3.0))
    if law == "laplace":
        b = 1.0 / np.sqrt(2.0)
        if tau == 0.5:
            return 0.0
        if tau < 0.5:
            return float(b * np.log(2.0 * tau))
        return float(-b * np.log(2.0 * (1.0 - tau)))
    raise ValueError(law)


def make_sigma(design: Design):
    base = SIGMA[design.sigma]

    def fn(x, z):
        s = design.scale * base(x, z)
        if (not np.isfinite(s).all()) or np.any(s <= 0):
            raise FloatingPointError(design.name)
        return s

    return fn


def simulate(n: int, design: Design, data_seed: int):
    rng = np.random.default_rng(data_seed)
    x, z = generate_covariates(n, rng)
    m = nonlinear_truth(z, 3)
    sigma = make_sigma(design)(x, z)
    eps = draw_errors(n, design.error_law, rng)
    y = x @ THETA0 + m + sigma * eps
    order = rng.permutation(n)
    n_fit = int(TRAIN_FRACTION * n)
    tr, va = order[:n_fit], order[n_fit:]
    scaler = StandardScaler().fit(z[tr])
    return dict(
        x_tr=x[tr], z_tr=z[tr], y_tr=y[tr],
        x_va=x[va], z_va=z[va], y_va=y[va],
        scaler=scaler, n_fit=n_fit, n_val=len(va),
        mean_sigma=float(np.mean(sigma)),
        p10_sigma=float(np.quantile(sigma, 0.1)),
        p90_sigma=float(np.quantile(sigma, 0.9)),
        hetero_ratio=float(np.quantile(sigma, 0.9) / max(np.quantile(sigma, 0.1), 1e-8)),
    )


def tensors(data):
    sc = data["scaler"]
    z_tr = torch.tensor(sc.transform(data["z_tr"]), dtype=torch.float32)
    z_va = torch.tensor(sc.transform(data["z_va"]), dtype=torch.float32)
    x_tr = torch.tensor(data["x_tr"], dtype=torch.float32)
    x_va = torch.tensor(data["x_va"], dtype=torch.float32)
    y_tr = torch.tensor(data["y_tr"].reshape(-1, 1), dtype=torch.float32)
    y_va = torch.tensor(data["y_va"].reshape(-1, 1), dtype=torch.float32)
    return (x_tr, z_tr), y_tr, (x_va, z_va), y_va


def new_model(fit_seed: int) -> Model:
    seed_all(fit_seed)
    net = dqNetSparse(
        2, 8, torch.zeros((1, 2), dtype=torch.float32),
        [HP["depth"], HP["width"]], sparseRatio=0.5,
    )
    net.linLinear.reset_parameters()
    opt = tt.optim.AdamW(
        lr=HP["lr"], betas=ADAM_BETAS, eps=ADAM_EPS, decoupled_weight_decay=0.0,
    )
    model = Model(net, checkLoss(tau=TAU), optimizer=opt, device="cpu")
    model.optimizer.set_lr(HP["lr"])
    return model


def theta_of(model: Model) -> np.ndarray:
    return model.net.linLinear.weight.detach().cpu().numpy().reshape(-1).astype(float)


def full_check_loss(model: Model, inputs, target) -> float:
    net = model.net
    was = net.training
    net.eval()
    loss_fn = checkLoss(tau=TAU)
    total = 0.0
    try:
        with torch.no_grad():
            for start in range(0, len(target), HP["batch_size"]):
                stop = min(start + HP["batch_size"], len(target))
                pred = net(*(v[start:stop] for v in inputs))
                total += float(loss_fn(pred, target[start:stop])) * (stop - start)
    finally:
        net.train(was)
    return total / max(len(target), 1)


class EpochCheckpoints(tt.callbacks.Callback):
    def __init__(self, wanted, on_epoch):
        self.wanted = set(int(e) for e in wanted)
        self.on_epoch = on_epoch
        self._epoch = 0

    def on_epoch_end(self):
        self._epoch += 1
        if self._epoch in self.wanted:
            self.on_epoch(self._epoch)


def fit_with_checkpoints(model, train_in, train_y, val_in, val_y, max_epochs, checkpoints, log_row):
    rows = []

    def grab(epoch):
        th = theta_of(model)
        rows.append(log_row(
            epoch, th,
            full_check_loss(model, train_in, train_y),
            full_check_loss(model, val_in, val_y),
        ))

    model.fit(
        train_in, train_y,
        batch_size=HP["batch_size"],
        epochs=max_epochs,
        callbacks=[EpochCheckpoints(checkpoints, grab)],
        val_data=(val_in, val_y),
        verbose=False,
    )
    if max_epochs in set(checkpoints) and (not rows or rows[-1]["epoch"] != max_epochs):
        grab(max_epochs)
    return rows


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_designs(verify: bool) -> list[Design]:
    """Smoke grid stresses stronger-than-paper heteroskedasticity.

    Keeps smoke-scale n/epochs/reps, but sweeps forms that undo the paper's
    small divisors (/5, /10) and amplify X- or Z-driven scale.
    """
    if verify:
        return [
            Design("mild_s1_t3", "mild_prior", 1.0, "t3"),
            Design("paper6_s1_t3", "paper_case6", 1.0, "t3"),
            Design("paper4_s1_t3", "paper_case4", 1.0, "t3"),
            Design("mild_x10_s1_t3", "mild_x10", 1.0, "t3"),
            Design("mild_s10_t3", "mild_prior", 10.0, "t3"),
            Design("paper6_s10_t3", "paper_case6", 10.0, "t3"),
            Design("paper4_s10_t3", "paper_case4", 10.0, "t3"),
            Design("case4_unscaled_t3", "case4_unscaled", 1.0, "t3"),
            Design("case6_amp_phi_t3", "case6_amp_phi", 1.0, "t3"),
            Design("steep_s1_t3", "steep_x", 1.0, "t3"),
            Design("steep_hard_t3", "steep_x_hard", 1.0, "t3"),
            Design("interact_s1_t3", "interaction", 1.0, "t3"),
            Design("interact_s5_t3", "interaction", 5.0, "t3"),
            Design("x_mult_t3", "x_multiplicative", 1.0, "t3"),
            Design("z_spike_t3", "z_spike", 1.0, "t3"),
            Design("paper6_s10_normal", "paper_case6", 10.0, "normal"),
            Design("mild_x10_normal", "mild_x10", 1.0, "normal"),
        ]
    out = []
    for sigma in (
        "mild_prior", "mild_x10", "paper_case6", "paper_case4",
        "case4_unscaled", "case6_amp_phi", "steep_x", "steep_x_hard",
        "interaction", "x_multiplicative", "z_spike",
    ):
        for scale in (1.0, 2.0, 5.0, 10.0):
            for law in ("t3", "normal"):
                out.append(Design(f"{sigma}_s{scale:g}_{law}", sigma, scale, law))
    return out


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    keys = ["design", "sigma", "scale", "error_law", "n_total", "epoch"]
    rows = []
    for key, g in raw.groupby(keys, sort=True):
        design, sigma, scale, law, n_total, epoch = key
        q = quantile_of_law(TAU, law)
        slopes = AFFINE_SLOPES.get(sigma)
        theta_tau = THETA0 + q * (slopes * float(scale)) if slopes is not None else None
        for j in (1, 2):
            col = f"theta_hat_{j}"
            err0 = g[col].to_numpy() - THETA0[j - 1]
            rec = dict(
                design=design, sigma=sigma, scale=float(scale), error_law=law,
                n_total=int(n_total), epoch=int(epoch), coef=j, Q=len(g),
                mean_theta=float(g[col].mean()),
                bias_vs_theta0=float(err0.mean()),
                rmse_vs_theta0=float(np.sqrt(np.mean(err0**2))),
                mean_train_check=float(g["train_check_loss"].mean()),
                mean_val_check=float(g["val_check_loss"].mean()),
                mean_hetero_ratio=float(g["hetero_ratio"].mean()),
            )
            if theta_tau is not None:
                errt = g[col].to_numpy() - theta_tau[j - 1]
                rec.update(
                    theta_tau=float(theta_tau[j - 1]),
                    bias_vs_theta_tau=float(errt.mean()),
                    rmse_vs_theta_tau=float(np.sqrt(np.mean(errt**2))),
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def run(args):
    torch.set_num_threads(1)
    out = HERE / ("smoke" if args.verify else args.output_name)
    out.mkdir(parents=True, exist_ok=True)

    designs = default_designs(args.verify)
    if args.designs:
        keep = set(args.designs.split(","))
        designs = [d for d in designs if d.name in keep]
        if not designs:
            raise SystemExit(f"No designs matched --designs {args.designs}")

    n_list = [int(x) for x in args.n_list.split(",")]
    epochs = [int(x) for x in args.epochs.split(",")]
    max_epochs = max(epochs)

    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        verify=bool(args.verify),
        tau=TAU,
        theta0=THETA0.tolist(),
        hp=HP,
        adam_betas=list(ADAM_BETAS),
        adam_eps=ADAM_EPS,
        train_fraction=TRAIN_FRACTION,
        n_list=n_list,
        epochs=epochs,
        max_epochs=max_epochs,
        reps=args.reps,
        base_seed=args.seed,
        designs=[asdict(d) for d in designs],
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        note=(
            "Primary metric: bias/RMSE of theta_hat vs structural theta_0 across epochs. "
            "bias_vs_theta_tau only for affine-in-X sigma families."
        ),
    )
    atomic_json(config, out / "run_config.json")

    raw_rows = []
    t0 = time.time()
    for design in designs:
        for n_total in n_list:
            for rep in range(1, args.reps + 1):
                data_seed = int(args.seed + (hash((design.name, n_total, rep)) % 1_000_000_000))
                fit_seed = data_seed + 17
                data = simulate(n_total, design, data_seed)
                train_in, train_y, val_in, val_y = tensors(data)
                model = new_model(fit_seed)

                def log_row(
                    epoch, th, tr_loss, va_loss,
                    _d=design, _n=n_total, _r=rep, _data=data, _ds=data_seed, _fs=fit_seed,
                ):
                    return dict(
                        design=_d.name, sigma=_d.sigma, scale=_d.scale, error_law=_d.error_law,
                        n_total=_n, n_fit=_data["n_fit"], n_val=_data["n_val"],
                        replication=_r, data_seed=_ds, fit_seed=_fs, epoch=int(epoch),
                        theta_hat_1=float(th[0]), theta_hat_2=float(th[1]),
                        error_theta1=float(th[0] - THETA0[0]),
                        error_theta2=float(th[1] - THETA0[1]),
                        train_check_loss=float(tr_loss),
                        val_check_loss=float(va_loss),
                        mean_sigma=_data["mean_sigma"],
                        hetero_ratio=_data["hetero_ratio"],
                        p10_sigma=_data["p10_sigma"],
                        p90_sigma=_data["p90_sigma"],
                    )

                rows = fit_with_checkpoints(
                    model, train_in, train_y, val_in, val_y, max_epochs, epochs, log_row,
                )
                raw_rows.extend(rows)
                print(
                    f"[{design.name}] n={n_total} rep={rep}/{args.reps} "
                    f"hetero={data['hetero_ratio']:.2f} "
                    f"theta={rows[-1]['theta_hat_1']:.3f},{rows[-1]['theta_hat_2']:.3f}",
                    flush=True,
                )

    raw = pd.DataFrame(raw_rows)
    atomic_csv(raw, out / "raw_checkpoints.csv")
    summary = summarize(raw)
    atomic_csv(summary, out / "summary.csv")

    drift_rows = []
    if not summary.empty:
        for (design, n_total, coef), g in summary.groupby(["design", "n_total", "coef"]):
            g = g.sort_values("epoch")
            first, last = g.iloc[0], g.iloc[-1]
            drift_rows.append(dict(
                design=design, n_total=int(n_total), coef=int(coef),
                epoch_first=int(first["epoch"]), epoch_last=int(last["epoch"]),
                abs_bias_first=abs(float(first["bias_vs_theta0"])),
                abs_bias_last=abs(float(last["bias_vs_theta0"])),
                abs_bias_delta=abs(float(last["bias_vs_theta0"])) - abs(float(first["bias_vs_theta0"])),
                rmse_first=float(first["rmse_vs_theta0"]),
                rmse_last=float(last["rmse_vs_theta0"]),
                mean_hetero_ratio=float(first["mean_hetero_ratio"]),
            ))
    drift = pd.DataFrame(drift_rows)
    atomic_csv(drift, out / "epoch_drift.csv")
    elapsed = time.time() - t0
    atomic_json(dict(elapsed_seconds=elapsed, rows=len(raw), output=str(out)), out / "run_meta.json")
    print(f"Done in {elapsed:.1f}s -> {out}")
    if not drift.empty:
        print(drift.sort_values("abs_bias_delta", ascending=False).head(12).to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true", help="Tiny smoke grid")
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--n-list", default=None)
    p.add_argument("--epochs", default=None)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--designs", default=None)
    p.add_argument("--output-name", default="run")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify:
        args.reps = args.reps or 2
        args.n_list = args.n_list or "400"
        args.epochs = args.epochs or "20,60"
    else:
        args.reps = args.reps or 20
        args.n_list = args.n_list or "1000,2000"
        args.epochs = args.epochs or "200,500,1000,2000"
    run(args)


if __name__ == "__main__":
    main()
