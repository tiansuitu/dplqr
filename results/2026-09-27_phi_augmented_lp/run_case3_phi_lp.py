"""Case 3 oracle phi*-orthogonalization test (diagnostic only; oracle phi*).

For each saved Case 3 fit (run/replications, epochs=1000 by default), hold m_hat fixed and solve
    adam    : theta from Adam (as saved)
    theta_lp: min_theta      P_n rho_tau(Y - m_hat - X theta)                 (re-profiled, t=0)
    phi_lp  : min_{theta,t}  P_n rho_tau(Y - m_hat - X theta - phi*(Z) t)       (joint LP, X and phi*)
and report sqrt(N)(theta - theta0), scores, and Wald coverage with three density choices:
f0 (oracle t3 density at 0), f_eff = mean f_t3(m_hat - m0) (oracle curvature under the bad m_hat),
and f_hat = kernel density at 0 of the fitted residuals (practical). Sandwich uses X~ = X - phi*.
Writes only inside results/2026-09-27_phi_augmented_lp/case3_run/. Source run untouched.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import torch

HERE = Path(__file__).resolve().parent
C3 = HERE.parent / "2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(C3))
import psi_n_if_helpers as H  # noqa: E402

TAU, TH0, F0 = H.TAU, H.THETA_0, H.ORACLE_F0
Z975 = norm.ppf(0.975)


def scores(r, x, xt, phi):
    s = TAU - (r < 0)
    return (-(s[:, None] * xt).mean(0), -(s[:, None] * x).mean(0), (s[:, None] * phi).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n", type=int, nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out", default="case3_run")
    a = ap.parse_args()
    _root, bridge, orig = H.load_stack(C3)
    avail = [(n, e, r) for n, e, r in H.list_available(C3) if e == a.epochs
             and (a.n is None or n in a.n) and (a.reps is None or r <= a.reps)]
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for k, (n, ep, rep) in enumerate(avail, 1):
        path = H.replication_dir(C3, n, ep, rep)
        fits = torch.load(path / "fits.pt", map_location="cpu", weights_only=False)
        data = H.regenerate(orig, n, rep)
        if data.data_sha256 != fits["data_sha256"]:
            raise ValueError(f"hash mismatch {path}")
        net = H.load_main_net(fits["stages"]["main"]["state"])
        th_adam, m_hat, _ = H.predict_train(net, data)
        x, z, y = data.x_train, data.z_train, data.y_train
        N = len(y); rn = math.sqrt(N)
        phi = H.true_phi_star_case3(z)
        xt = x - phi
        m0 = orig.nonlinear_truth(z, 3)
        f_eff = float(np.mean(student_t.pdf(m_hat - m0, df=3)))
        ginv = np.linalg.inv(xt.T @ xt / N)
        off = y - m_hat
        th_prof = H.check_linprog(off, x)
        b = H.check_linprog(off, np.column_stack([x, phi]))
        th_phi, t_hat = b[:2], b[2:]
        for var, th, m in [("adam", th_adam, m_hat), ("theta_lp", th_prof, m_hat),
                           ("phi_lp", th_phi, m_hat + phi @ t_hat)]:
            r = y - x @ th - m
            so, sg, sp = scores(r, x, xt, phi)
            f_hat = float(bridge.residual_density_zero(r))
            row = dict(n=n, epochs=ep, rep=rep, N=N, variant=var, f_eff=f_eff, f_hat=f_hat,
                       loss=float(np.mean(np.where(r >= 0, TAU * r, (TAU - 1) * r))),
                       m_l2=float(np.sqrt(np.mean((m_hat - m0) ** 2))))
            for j in range(2):
                row[f"sn_xi_{j+1}"] = rn * (th[j] - TH0[j])
                row[f"sn_orth_score_{j+1}"] = rn * so[j]
                row[f"sn_G_{j+1}"] = rn * sg[j]
                row[f"sn_phi_score_{j+1}"] = rn * sp[j]
                row[f"t_hat_{j+1}"] = float(t_hat[j]) if var == "phi_lp" else 0.0
                for lab, f in [("f0", F0), ("feff", f_eff), ("fhat", f_hat)]:
                    se = math.sqrt(TAU * (1 - TAU) * ginv[j, j] / N) / f
                    row[f"se_{lab}_{j+1}"] = se
                    row[f"cover_{lab}_{j+1}"] = float(abs(th[j] - TH0[j]) <= Z975 * se)
            rows.append(row)
        if k % 50 == 0:
            print(f"{k}/{len(avail)} n={n} rep={rep} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "replication_results.csv", index=False)
    summ = []
    for (n, var), g in df.groupby(["n", "variant"]):
        d = dict(n=n, variant=var, reps=len(g), mean_f_eff=g.f_eff.mean(), mean_f_hat=g.f_hat.mean(),
                 mean_m_l2=g.m_l2.mean(), mean_loss=g.loss.mean())
        for j in (1, 2):
            x = g[f"sn_xi_{j}"]
            d[f"sn_bias_{j}"] = x.mean(); d[f"sn_bias_mcse_{j}"] = x.std(ddof=1) / math.sqrt(len(x))
            d[f"sn_sd_{j}"] = x.std(ddof=1)
            d[f"sn_se_f0_{j}"] = (g[f"se_f0_{j}"] * np.sqrt(g.N)).mean()
            d[f"sn_se_feff_{j}"] = (g[f"se_feff_{j}"] * np.sqrt(g.N)).mean()
            d[f"sn_se_fhat_{j}"] = (g[f"se_fhat_{j}"] * np.sqrt(g.N)).mean()
            for lab in ("f0", "feff", "fhat"):
                d[f"cover_{lab}_{j}"] = g[f"cover_{lab}_{j}"].mean()
            for s in ("sn_orth_score", "sn_G", "sn_phi_score", "t_hat"):
                d[f"mean_{s}_{j}"] = g[f"{s}_{j}"].mean()
                d[f"mean_abs_{s}_{j}"] = g[f"{s}_{j}"].abs().mean()
        summ.append(d)
    pd.DataFrame(summ).to_csv(out / "summary_by_n.csv", index=False)
    paired = []
    for n, g in df.groupby("n"):
        p = g.pivot(index="rep", columns="variant", values="sn_xi_1")
        p2 = g.pivot(index="rep", columns="variant", values="sn_xi_2")
        for var in ("theta_lp", "phi_lp"):
            for j, pp in ((1, p), (2, p2)):
                dd = pp[var] - pp["adam"]
                paired.append(dict(n=n, variant=var, coord=j, mean_change=dd.mean(),
                                   mcse=dd.std(ddof=1) / math.sqrt(len(dd))))
    pd.DataFrame(paired).to_csv(out / "paired_vs_adam.csv", index=False)
    print("done", out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
