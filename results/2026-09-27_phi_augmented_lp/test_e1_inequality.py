"""Test ZW supplement p.5 inequality |dL*_n/dzeta at zeta_hat| <= (2/n) sum 1{r_i=0}|X~_i| per replication.
LHS = |P_n psi_tau(r) X~|, RHS = (2/n) sum_{|r_i|<tol} |X~_i|, X~ = X - phi*(Z), psi_tau(r)=tau-1{r<0}.
Fits: 'adam' (saved) and 'phi_lp' (joint (theta,t) LP on X and phi*, m_hat fixed). Toy (polished) and Case 3 (1000 epochs)."""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
RES = HERE.parent
TOLS = [1e-8, 1e-6]
TAU = 0.5


def ineq(r, xt, n):
    psi = TAU - (r < 0)
    lhs = np.abs((psi[:, None] * xt).mean(0))
    out = dict(lhs=lhs)
    for tol in TOLS:
        tie = np.abs(r) < tol
        out[tol] = (2.0 / n * np.abs(xt[tie]).sum(0), int(tie.sum()))
    return out


def record(rows, design, n, rep, fit, r, xt):
    o = ineq(r, xt, len(r)); rn = math.sqrt(len(r))
    for j in range(xt.shape[1]):
        row = dict(design=design, n=n, rep=rep, fit=fit, coord=j + 1, sn_lhs=rn * o["lhs"][j])
        for tol in TOLS:
            rhs, k = o[tol]
            row[f"sn_rhs_{tol:g}"] = rn * rhs[j]; row[f"ties_{tol:g}"] = k
            row[f"holds_{tol:g}"] = bool(o["lhs"][j] <= rhs[j] + 1e-15)
        rows.append(row)


rows = []
# ---- toy ----
sys.path.insert(0, str(RES / "2026-09-23_psi_n_thm33"))
import diagnostic_helpers as dh
sys.path.insert(0, str(HERE))
from run_phi_lp import lp_fit
RUN = RES / "2026-09-23_psi_n_thm33" / "run_expansion_v2" / "main_26a7078945fc"
c = json.loads((RUN / "manifest.json").read_text())["config"]
for n in c["n_values"]:
    for rep in range(1, c["reps"] + 1):
        ck = torch.load(RUN / "replications" / f"n_{n}_rep_{rep:04d}" / "polished.pt", map_location="cpu", weights_only=False)
        model = dh.JointModel(ck["hidden"]); model.load_state_dict(ck["state_dict"]); model.eval()
        d = dh.generate_sample(n, dh.seed_for(c, n, rep))
        m = dh.predict_m(model, d["Z"]); th = float(model.theta.detach())
        xt = (d["X"] - d["phi"]).reshape(-1, 1)
        record(rows, "toy", n, rep, "adam", d["Y"] - th * d["X"] - m, xt)
        b = lp_fit(d["Y"] - m, np.column_stack([d["X"], d["phi"]]))
        record(rows, "toy", n, rep, "phi_lp", d["Y"] - b[0] * d["X"] - m - b[1] * d["phi"], xt)
print("toy done", flush=True)
# ---- case 3 ----
C3 = RES / "2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(C3))
import psi_n_if_helpers as H
_, _, orig = H.load_stack(C3)
for n, ep, rep in H.list_available(C3):
    if ep != 1000: continue
    fits = torch.load(H.replication_dir(C3, n, ep, rep) / "fits.pt", map_location="cpu", weights_only=False)
    data = H.regenerate(orig, n, rep)
    net = H.load_main_net(fits["stages"]["main"]["state"])
    th, m_hat, _ = H.predict_train(net, data)
    x, y = data.x_train, data.y_train
    phi = H.true_phi_star_case3(data.z_train); xt = x - phi
    record(rows, "case3", n, rep, "adam", y - x @ th - m_hat, xt)
    b = H.check_linprog(y - m_hat, np.column_stack([x, phi]))
    record(rows, "case3", n, rep, "phi_lp", y - x @ b[:2] - m_hat - phi @ b[2:], xt)
print("case3 done", flush=True)
df = pd.DataFrame(rows)
out = HERE / "e1_inequality"; out.mkdir(exist_ok=True)
df.to_csv(out / "by_replication.csv", index=False)
agg = df.groupby(["design", "fit", "coord", "n"]).agg(
    reps=("rep", "size"), frac_holds_1e8=("holds_1e-08", "mean"), frac_holds_1e6=("holds_1e-06", "mean"),
    median_sn_lhs=("sn_lhs", "median"), median_sn_rhs_1e8=("sn_rhs_1e-08", "median"),
    median_ties_1e8=("ties_1e-08", "median"), median_ties_1e6=("ties_1e-06", "median")).reset_index()
agg.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(agg.round(4).to_string(index=False))
