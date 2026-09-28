"""Task B: toy design, joint (theta, t) check-loss LP with a cross-fitted phi_hat instead of oracle phi*.

For each saved toy fit (run_expansion_v2/main_26a7078945fc, stage 'polished', reps 1..50 on disk):
  * m_hat = in-sample Adam m_net prediction (held fixed, as in 2026-09-27_phi_augmented_lp/run_phi_lp.py).
  * phi_hat(Z) = E[X|Z] by 2-fold cross-fitting on the rep's own sample (fit fold A -> predict B, and
    vice versa), plain least squares. Learners: MLPRegressor (32,32) with early stopping [primary],
    GradientBoostingRegressor [robustness].
  * Joint LP  min_{theta,t} sum rho_tau(Y - m_hat - theta X - t phi_hat)  (lp_fit reused from run_phi_lp.py).
  * Comparators: Adam theta_hat, and the oracle joint LP with (X, phi*) (recomputed; checked against the
    old replication_results.csv).
Scores (same formulas as diagnostic_helpers.diagnose, with fitted m~ = m_hat + t*reg):
  S = sqrt(n) * (-P_n[(tau - 1{r<0}) V]),  root_n_phi_score = sqrt(n) P_n[(tau - 1{r<0}) phi*].
SE (matches dplqr_standard_errors in the Sept-4 code): var = tau(1-tau) / (f(0)^2 * Omega * n),
  Omega = np.cov(X_tilde, ddof=1).
  feasible: X_tilde = X - phi_hat (MLP phi_hat for adam / phistar_lp rows, the row's own phi_hat otherwise),
            f_hat(0) = R stats::density of that row's training residuals (source_bridge.residual_density_zero).
  oracle:   X_tilde = X - phi* = V, f0 = dnorm(0).
  pop_oracle: the fixed population SD used by diagnostic_helpers.bias_summary.
Writes only into this script's folder (or --out subfolder).
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, importlib.util, json, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
SRC = RESULTS / "2026-09-23_psi_n_thm33"
RUN = SRC / "run_expansion_v2" / "main_26a7078945fc"
OLD = RESULTS / "2026-09-27_phi_augmented_lp"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(OLD))
import diagnostic_helpers as dh  # noqa: E402
import run_phi_lp as rpl  # noqa: E402  (lp_fit)

spec = importlib.util.spec_from_file_location(
    "xfit_bridge", RESULTS / "2026-09-10-asymptotic-normality" / "source_bridge.py")
bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
residual_density_zero = bridge.residual_density_zero

TAU, THETA0 = dh.TAU, dh.THETA0
F0 = float(norm.pdf(0.0))
Z975 = float(norm.ppf(0.975))
STAGE = "polished"


def learner(name, seed):
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(32, 32), early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=20, max_iter=2000, learning_rate_init=1e-3, alpha=1e-4,
                            random_state=seed)
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    raise ValueError(name)


def crossfit_phi(z, x, name, seed):
    n = len(x)
    perm = np.random.default_rng(seed).permutation(n)
    folds = [np.sort(perm[: n // 2]), np.sort(perm[n // 2:])]
    out = np.empty(n)
    for a, b in [(0, 1), (1, 0)]:
        model = learner(name, seed + 17 * a).fit(z[folds[a]], x[folds[a]])
        out[folds[b]] = model.predict(z[folds[b]])
    return out


def row_metrics(data, theta, m_tilde, phihat_se, loss_adam):
    n = len(data["Y"]); rn = math.sqrt(n)
    r = data["Y"] - theta * data["X"] - m_tilde
    signs = TAU - (r < 0)
    f_hat = residual_density_zero(r)
    omega_feas = float(np.var(data["X"] - phihat_se, ddof=1))
    omega_orac = float(np.var(data["V"], ddof=1))
    se_f = math.sqrt(TAU * (1 - TAU) / (f_hat ** 2 * omega_feas * n))
    se_o = math.sqrt(TAU * (1 - TAU) / (F0 ** 2 * omega_orac * n))
    err = theta - THETA0
    loss = dh.check_loss(r)
    return dict(theta_hat=float(theta), root_n_theta_error=rn * err,
                S=rn * float(-np.mean(signs * data["V"])),
                root_n_phi_score=rn * float(np.mean(signs * data["phi"])),
                root_n_G=-rn * float(np.mean(signs * data["X"])),
                root_n_phihat_se_score=rn * float(np.mean(signs * phihat_se)),
                near_zero_residuals=int(np.sum(np.abs(r) <= 1e-8)),
                train_loss=loss, loss_minus_adam=loss - loss_adam,
                f_hat0=f_hat, omega_feasible=omega_feas, omega_oracle=omega_orac,
                se_feasible=se_f, se_oracle=se_o,
                cover_feasible=bool(abs(err) <= Z975 * se_f),
                cover_oracle=bool(abs(err) <= Z975 * se_o),
                cover_pop_oracle=bool(abs(rn * err) <= Z975 * dh.ASYMPTOTIC_SD))


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(df):
    rows = []
    for (n, meth), g in df.groupby(["n", "method"], sort=True):
        x = g.root_n_theta_error.to_numpy()
        b, se = mean_se(x)
        rows.append(dict(n=n, method=meth, reps=len(g), root_n_bias=b, mcse=se,
                         root_n_sd=float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                         root_n_rmse=float(np.sqrt(np.mean(x ** 2))),
                         coverage_feasible=g.cover_feasible.mean(), coverage_oracle=g.cover_oracle.mean(),
                         coverage_pop_oracle=g.cover_pop_oracle.mean(),
                         mean_root_n_se_feasible=float((np.sqrt(n) * g.se_feasible).mean()),
                         mean_root_n_se_oracle=float((np.sqrt(n) * g.se_oracle).mean()),
                         pop_root_n_sd=dh.ASYMPTOTIC_SD, mean_f_hat0=g.f_hat0.mean(),
                         mean_abs_root_n_phi_score=g.root_n_phi_score.abs().mean(),
                         mean_abs_S=g.S.abs().mean(), mean_S=g.S.mean(),
                         mean_abs_root_n_G=g.root_n_G.abs().mean(),
                         mean_t_hat=g.t_hat.mean(), mean_phi_l2=g.phi_l2.mean(),
                         mean_m_l2=g.m_l2.mean(), mean_product=g["root_n_product"].mean()))
    return pd.DataFrame(rows)


def paired(df):
    out = []
    for n, g in df.groupby("n", sort=True):
        piv = {m: g.pivot(index="rep", columns="method", values=m)
               for m in ["root_n_theta_error", "root_n_phi_score", "S"]}
        for lrn in ["phihat_lp_mlp", "phihat_lp_gb"]:
            for ref in ["adam", "phistar_lp"]:
                for m, p in piv.items():
                    d = p[lrn] - p[ref]
                    da = p[lrn].abs() - p[ref].abs()
                    mu, se = mean_se(d); mua, sea = mean_se(da)
                    out.append(dict(n=n, method=lrn, minus=ref, metric=m, pairs=len(d),
                                    mean_shift=mu, se=se, z=mu / se if se and se > 0 else float("nan"),
                                    mean_abs_shift=mua, se_abs=sea, frac_abs_reduced=float((da < 0).mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--n", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--cache-only", action="store_true", help="fill the per-rep cache, skip the outputs")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    c = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["config"]
    out = (HERE / args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(OLD / "run" / "replication_results.csv")
    old = old[old.stage == STAGE].set_index(["n", "rep", "variant"])
    n_values = args.n or c["n_values"]; reps = args.reps or c["reps"]
    rows, t0 = [], time.time()
    cache = out / "_cache"; cache.mkdir(exist_ok=True)
    for n in n_values:
        for rep in range(1, reps + 1):
            cpath = cache / f"n_{n}_rep_{rep:04d}.json"
            if cpath.exists():
                rows.extend(json.loads(cpath.read_text(encoding="utf-8")))
                continue
            rep_rows = []
            seed = dh.seed_for(c, n, rep)
            data = dh.generate_sample(n, seed)
            ck = torch.load(RUN / "replications" / f"n_{n}_rep_{rep:04d}" / f"{STAGE}.pt",
                            map_location="cpu", weights_only=False)
            assert ck["seed"] == seed and ck["n"] == n and ck["rep"] == rep
            model = dh.JointModel(ck["hidden"]); model.load_state_dict(ck["state_dict"]); model.eval()
            m_hat = dh.predict_m(model, data["Z"])
            theta_adam = float(model.theta.detach())
            y_off = data["Y"] - m_hat
            loss_adam = dh.check_loss(y_off - theta_adam * data["X"])
            m_l2 = float(np.sqrt(np.mean((m_hat - data["m0"]) ** 2)))
            ph = {k: crossfit_phi(data["Z"], data["X"], k, seed + 7919) for k in ["mlp", "gb"]}
            l2 = {k: float(np.sqrt(np.mean((v - data["phi"]) ** 2))) for k, v in ph.items()}
            fits = [("adam", theta_adam, 0.0, m_hat, "mlp", np.nan)]
            th_s, t_s = rpl.lp_fit(y_off, np.column_stack([data["X"], data["phi"]]))
            fits.append(("phistar_lp", th_s, t_s, m_hat + t_s * data["phi"], "mlp", 0.0))
            for k in ["mlp", "gb"]:
                th, tt = rpl.lp_fit(y_off, np.column_stack([data["X"], ph[k]]))
                fits.append((f"phihat_lp_{k}", th, tt, m_hat + tt * ph[k], k, l2[k]))
            for meth, th, tt, mtil, se_key, phi_l2 in fits:
                row = dict(n=n, rep=rep, seed=seed, method=meth, t_hat=float(tt), se_phihat=se_key)
                row.update(row_metrics(data, th, mtil, ph[se_key], loss_adam))
                row.update(m_l2=m_l2, phi_l2=phi_l2,
                           root_n_product=math.sqrt(n) * m_l2 * phi_l2 if np.isfinite(phi_l2) else np.nan,
                           phi_l2_mlp=l2["mlp"], phi_l2_gb=l2["gb"])
                ov = {"adam": "adam", "phistar_lp": "phi_lp"}.get(meth)
                if ov is not None:
                    o = old.loc[(n, rep, ov)]
                    row.update(old_theta_hat=float(o.theta_hat), old_S=float(o.S),
                               old_root_n_phi_score=float(o.root_n_phi_score))
                    if abs(o.theta_hat - th) > 1e-8:
                        raise AssertionError(f"theta mismatch vs old CSV n={n} rep={rep} {meth}")
                if meth != "adam" and row["loss_minus_adam"] > 1e-10:
                    raise AssertionError("LP did not lower the check loss")
                rep_rows.append(row)
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(rep_rows, default=lambda o: o.item()), encoding="utf-8")
            tmp.replace(cpath)  # per-rep cache makes the run resumable
            rows.extend(rep_rows)
            if rep % 10 == 0 or reps <= 5:
                print(f"n={n} rep={rep} elapsed={time.time()-t0:.0f}s", flush=True)
    if args.cache_only:
        print(f"cache filled for n={n_values} {time.time()-t0:.0f}s"); return
    df = pd.DataFrame(rows)
    df.to_csv(out / "replication_results.csv", index=False)
    s = summarize(df); s.to_csv(out / "summary.csv", index=False)
    p = paired(df); p.to_csv(out / "paired.csv", index=False)
    chk = df.dropna(subset=["old_S"])
    print("max |S - old S| =", float((chk.S - chk.old_S).abs().max()),
          " max |phi score - old| =", float((chk.root_n_phi_score - chk.old_root_n_phi_score).abs().max()))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(s[["n", "method", "reps", "root_n_bias", "mcse", "root_n_sd", "root_n_rmse", "coverage_feasible",
             "coverage_oracle", "coverage_pop_oracle", "mean_abs_root_n_phi_score", "mean_abs_S",
             "mean_phi_l2", "mean_m_l2", "mean_product", "mean_f_hat0", "mean_t_hat"]].round(4).to_string())
    print(p[p.metric == "root_n_theta_error"].round(4).to_string())
    print(f"done {out} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
