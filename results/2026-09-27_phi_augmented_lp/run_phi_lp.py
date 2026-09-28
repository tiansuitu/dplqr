"""Oracle phi*-augmented exact-ERM diagnostic (diagnostic only; uses oracle phi*).

For every saved fit in run_expansion_v2/main_26a7078945fc (same seeds, same data),
hold m_hat fixed and solve the check-loss LP
    min_{theta,t}  sum_i rho_tau(Y_i - m_hat(Z_i) - theta X_i - t phi*(Z_i)),
i.e. exact ERM along both the theta and the phi* direction. The resulting fit
(theta_tilde, m_hat + t_tilde phi*) is re-diagnosed with the unchanged
diagnostic_helpers.diagnose (S, E, T, W, R_score, root_n_G, root_n_phi_score, ...).
Contrast: 'theta_lp' solves the LP in theta only (t = 0).
Writes only inside this folder. Does not modify the source run.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "2026-09-23_psi_n_thm33"
RUN = SRC / "run_expansion_v2" / "main_26a7078945fc"
sys.path.insert(0, str(SRC))
import diagnostic_helpers as dh  # noqa: E402

TAU = dh.TAU


def lp_fit(y_off, design):
    """min sum rho_tau(y_off - design @ b); returns b."""
    n, p = design.shape
    a = sparse.hstack([sparse.csr_matrix(design), sparse.eye(n), -sparse.eye(n)], format="csr")
    cost = np.r_[np.zeros(p), np.full(n, TAU), np.full(n, 1 - TAU)]
    fit = linprog(cost, A_eq=a, b_eq=y_off,
                  bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x[:p]


class PhiShift(nn.Module):
    def __init__(self, base, t):
        super().__init__()
        self.base, self.t = base, float(t)

    def forward(self, z):
        z1, z2 = z[:, 0], z[:, 1]
        phi = 0.5 + 0.3 * torch.sin(2 * math.pi * z1) + 0.3 * z1 * z2
        return self.base(z) + self.t * phi.unsqueeze(-1)


class Wrapped(nn.Module):
    def __init__(self, theta, m_net):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta), dtype=torch.float64))
        self.m_net = m_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None, help="limit reps per n (smoke)")
    ap.add_argument("--n", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="run")
    args = ap.parse_args()

    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    c = manifest["config"]
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    n_values = args.n or c["n_values"]
    reps = args.reps or c["reps"]
    rows, t0 = [], time.time()
    for n in n_values:
        for rep in range(1, reps + 1):
            folder = RUN / "replications" / f"n_{n}_rep_{rep:04d}"
            saved = {r["stage"]: r for r in json.loads((folder / "result.json").read_text(encoding="utf-8"))}
            seed = dh.seed_for(c, n, rep)
            data = dh.generate_sample(n, seed)
            for stage in ["baseline", "polished"]:
                ck = torch.load(folder / f"{stage}.pt", map_location="cpu", weights_only=False)
                assert ck["seed"] == seed and ck["n"] == n and ck["rep"] == rep
                model = dh.JointModel(ck["hidden"])
                model.load_state_dict(ck["state_dict"])
                model.eval()
                # 1) reproduce the saved Adam diagnostics exactly
                r_adam = dh.diagnose(model, data, c, n, rep, stage, saved[stage]["selected_polish_epoch"])
                for k in ["theta_hat", "S", "E", "T", "root_n_phi_score"]:
                    if abs(r_adam[k] - saved[stage][k]) > 1e-7:
                        raise AssertionError(f"reproduction mismatch n={n} rep={rep} {stage} {k}")
                m_hat = dh.predict_m(model, data["Z"])
                y_off = data["Y"] - m_hat
                loss_adam = dh.check_loss(y_off - r_adam["theta_hat"] * data["X"])
                variants = [("adam", r_adam, 0.0)]
                # 2) theta-only LP (t = 0)
                th = lp_fit(y_off, data["X"].reshape(-1, 1))[0]
                variants.append(("theta_lp", dh.diagnose(Wrapped(th, model.m_net), data, c, n, rep, stage, -1), 0.0))
                # 3) joint (theta, t) LP along X and phi*
                th2, tt = lp_fit(y_off, np.column_stack([data["X"], data["phi"]]))
                m2 = PhiShift(model.m_net, tt)
                variants.append(("phi_lp", dh.diagnose(Wrapped(th2, m2), data, c, n, rep, stage, -1), tt))
                for name, row, tval in variants:
                    row = dict(row)
                    row["variant"], row["t_hat"] = name, tval
                    row["loss_minus_adam"] = row["final_training_loss"] - loss_adam
                    if name != "adam" and row["loss_minus_adam"] > 1e-10:
                        raise AssertionError("LP did not lower the check loss")
                    rows.append(row)
            if rep % 10 == 0:
                print(f"n={n} rep={rep} elapsed={time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "replication_results.csv", index=False)

    # summaries
    metrics = ["root_n_theta_error", "S", "E", "T", "W", "R_score", "R_theta", "root_n_G",
               "root_n_phi_score", "t_hat", "exact_zero_residual_count", "loss_minus_adam", "n_quarter_m_l2"]
    summ = []
    for (n, stage, var), g in df.groupby(["n", "stage", "variant"]):
        for m in metrics:
            x = g[m].to_numpy(float)
            summ.append(dict(n=n, stage=stage, variant=var, metric=m, reps=len(x),
                             mean=x.mean(), mc_se=x.std(ddof=1) / math.sqrt(len(x)),
                             mean_abs=np.abs(x).mean()))
        x = g["root_n_theta_error"].to_numpy()
        summ.append(dict(n=n, stage=stage, variant=var, metric="E_over_J", reps=len(x),
                         mean=(g["E"] / dh.J).mean(), mc_se=(g["E"] / dh.J).std(ddof=1) / math.sqrt(len(x)),
                         mean_abs=(g["E"] / dh.J).abs().mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    # paired changes vs Adam, same seed
    paired = []
    for (n, stage), g in df.groupby(["n", "stage"]):
        piv = {m: g.pivot(index="rep", columns="variant", values=m) for m in
               ["root_n_theta_error", "S", "root_n_phi_score", "root_n_G", "E"]}
        for var in ["theta_lp", "phi_lp"]:
            for m, p in piv.items():
                d = p[var] - p["adam"]
                da = p[var].abs() - p["adam"].abs()
                paired.append(dict(n=n, stage=stage, variant=var, metric=m, pairs=len(d),
                                   mean_change=d.mean(), mc_se=d.std(ddof=1) / math.sqrt(len(d)),
                                   mean_abs_change=da.mean(), frac_abs_reduced=float((da < 0).mean())))
    pd.DataFrame(paired).to_csv(out / "paired_vs_adam.csv", index=False)
    print("done", out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
