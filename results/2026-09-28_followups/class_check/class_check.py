"""Task A: how far the saved fits sit outside the Zhong-Wang (Schmidt-Hieber) sparse bounded class.

ZW class: all weights |w| <= 1, ||f||_inf <= F, sparsity s, size growing with n.
For every saved fit we report weight magnitudes (overall, per layer, weights/biases separately),
parameter counts (total / |w| > 1e-8), the range of m_hat on training Z and on 100k fresh Z draws
vs. the range and sup of m0, and the architecture per n.

Toy: results/2026-09-23_psi_n_thm33/run_expansion_v2/main_26a7078945fc (stages baseline, polished;
     reps are numbered 1..50 on disk).
Case 3: results/2026-09-11_asymptotic_normality_case3/run (epochs=1000, first 50 reps per n; network
     'main' stage). Masked (sparse) hidden layers are evaluated on the EFFECTIVE weights weight*mask;
     the parametric part (theta / linLinear) is excluded from the network-class check.
Only reads existing artefacts; writes by_fit.csv and summary.csv next to this script.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
TOY_SRC = RESULTS / "2026-09-23_psi_n_thm33"
TOY_RUN = TOY_SRC / "run_expansion_v2" / "main_26a7078945fc"
C3_EXP = RESULTS / "2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(TOY_SRC))
sys.path.insert(0, str(C3_EXP))
import diagnostic_helpers as dh  # noqa: E402
import psi_n_if_helpers as ph  # noqa: E402

N_FRESH = 100_000
NZ_TOL = 1e-8
REPS = 50


def layer_stats(layers):
    """layers: list of (name, W ndarray, b ndarray) in forward order (effective weights)."""
    row, allv = {}, []
    wmax, bmax = 0.0, 0.0
    for k, (name, w, b) in enumerate(layers, start=1):
        row[f"L{k}_w_maxabs"] = float(np.abs(w).max())
        row[f"L{k}_b_maxabs"] = float(np.abs(b).max())
        row[f"L{k}_frac_gt1"] = float(np.mean(np.abs(np.r_[w.ravel(), b.ravel()]) > 1))
        wmax, bmax = max(wmax, row[f"L{k}_w_maxabs"]), max(bmax, row[f"L{k}_b_maxabs"])
        allv.append(w.ravel()); allv.append(b.ravel())
    allv = np.abs(np.concatenate(allv))
    widths = [layers[0][1].shape[1]] + [w.shape[0] for _, w, _ in layers]
    row.update(arch="-".join(map(str, widths)), n_affine_layers=len(layers),
               n_hidden_layers=len(layers) - 1, max_hidden_width=max(widths[1:-1]),
               total_params=int(allv.size), nonzero_params=int((allv > NZ_TOL).sum()),
               max_abs_weight=wmax, max_abs_bias=bmax, max_abs_param=float(allv.max()),
               count_gt1=int((allv > 1).sum()), frac_gt1=float(np.mean(allv > 1)),
               count_ge_0p999=int((allv >= 0.999).sum()), frac_ge_0p999=float(np.mean(allv >= 0.999)))
    return row


def range_stats(mh_tr, m0_tr, mh_fr, m0_fr, sup_ref):
    return dict(mhat_train_min=float(mh_tr.min()), mhat_train_max=float(mh_tr.max()),
                m0_train_min=float(m0_tr.min()), m0_train_max=float(m0_tr.max()),
                mhat_fresh_min=float(mh_fr.min()), mhat_fresh_max=float(mh_fr.max()),
                m0_fresh_min=float(m0_fr.min()), m0_fresh_max=float(m0_fr.max()),
                sup_abs_m0_ref=sup_ref,
                mhat_fresh_max_over_sup_m0=float(mh_fr.max() / sup_ref),
                mhat_fresh_min_over_sup_m0=float(mh_fr.min() / sup_ref),
                sup_abs_mhat_fresh_over_sup_m0=float(np.abs(mh_fr).max() / sup_ref),
                mhat_train_max_over_sup_m0=float(mh_tr.max() / sup_ref),
                mhat_train_min_over_sup_m0=float(mh_tr.min() / sup_ref))


def toy_rows():
    c = json.loads((TOY_RUN / "manifest.json").read_text(encoding="utf-8"))["config"]
    g = np.linspace(0, 1, 2001)
    gz = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
    sup_grid = float(np.abs(dh.m0_fn(gz)).max())
    z_fr = np.random.default_rng(20260928).uniform(0, 1, (N_FRESH, 2))
    m0_fr = dh.m0_fn(z_fr)
    print(f"toy sup|m0| grid={sup_grid:.5f}  m0 grid range=[{dh.m0_fn(gz).min():.4f},{dh.m0_fn(gz).max():.4f}]", flush=True)
    rows = []
    for n in c["n_values"]:
        for rep in range(1, REPS + 1):
            folder = TOY_RUN / "replications" / f"n_{n}_rep_{rep:04d}"
            data = dh.generate_sample(n, dh.seed_for(c, n, rep))
            for stage in ["baseline", "polished"]:
                ck = torch.load(folder / f"{stage}.pt", map_location="cpu", weights_only=False)
                assert ck["n"] == n and ck["rep"] == rep and ck["seed"] == dh.seed_for(c, n, rep)
                model = dh.JointModel(ck["hidden"]); model.load_state_dict(ck["state_dict"]); model.eval()
                lin = [(nm, m.weight.detach().numpy(), m.bias.detach().numpy())
                       for nm, m in model.m_net.named_children() if isinstance(m, torch.nn.Linear)]
                row = dict(design="toy", stage=stage, n=n, n_train=n, rep=rep,
                           theta_hat=float(model.theta.detach()), n_masked=0)
                row.update(layer_stats(lin))
                row.update(range_stats(dh.predict_m(model, data["Z"]), data["m0"],
                                       dh.predict_m(model, z_fr), m0_fr, sup_grid))
                rows.append(row)
        print(f"toy n={n} done", flush=True)
    return rows


def case3_rows():
    _root, _bridge, original = ph.load_stack(C3_EXP)
    avail = sorted((n, r) for n, e, r in ph.list_available(C3_EXP) if e == 1000)
    rng = np.random.default_rng(20260928)
    _x, z_fr = original.generate_covariates(N_FRESH, rng)
    m0_fr = original.nonlinear_truth(z_fr, 3)
    # m0 is unbounded below on (0,2)^8 (log(z5 + z6 z7) -> -inf); use an empirical sup over 1e6 draws.
    _x, z_big = original.generate_covariates(1_000_000, np.random.default_rng(20260929))
    m0_big = original.nonlinear_truth(z_big, 3)
    sup_emp = float(np.abs(m0_big).max())
    print(f"case3 empirical sup|m0| (1e6 draws)={sup_emp:.4f}; range=[{m0_big.min():.4f},{m0_big.max():.4f}]; "
          f"q0.001/q0.999=[{np.quantile(m0_big,.001):.4f},{np.quantile(m0_big,.999):.4f}]", flush=True)
    rows = []
    for n in sorted({a[0] for a in avail}):
        reps = sorted(r for nn, r in avail if nn == n)[:REPS]
        for rep in reps:
            path = ph.replication_dir(C3_EXP, n, 1000, rep)
            fits = torch.load(path / "fits.pt", map_location="cpu", weights_only=False)
            data = ph.regenerate(original, n, rep)
            if data.data_sha256 != fits["data_sha256"]:
                raise ValueError(f"data hash mismatch {path}")
            state = fits["stages"]["main"]["state"]
            net = ph.load_main_net(state)
            theta, mh_tr, _ = ph.predict_train(net, data)
            mh_fr = ph.predict_m_on_z(net, data, z_fr)
            m0_tr = original.nonlinear_truth(data.z_train, 3)
            sd = {k: v.detach().double().numpy() for k, v in state.items()}
            names = ["nonparLinear1"] + [f"nonparLinearList.{i}" for i in range(len(net.nonparLinearList))] + ["nonparLinearEnd"]
            lin, n_masked = [], 0
            for nm in names:
                w = sd[f"{nm}.weight"]
                if f"{nm}.mask" in sd:
                    w = w * sd[f"{nm}.mask"]; n_masked += int((sd[f"{nm}.mask"] == 0).sum())
                lin.append((nm, w, sd[f"{nm}.bias"]))
            row = dict(design="case3", stage="main", n=n, n_train=len(data.y_train), rep=rep,
                       theta_hat=float(theta[0]), theta_hat_2=float(theta[1]), n_masked=n_masked,
                       raw_max_abs_nonpar=float(max(np.abs(sd[k]).max() for k in sd if k.startswith("nonpar") and not k.endswith("mask"))))
            row.update(layer_stats(lin))
            row.update(range_stats(mh_tr, m0_tr, mh_fr, m0_fr, sup_emp))
            rows.append(row)
        print(f"case3 n={n} done ({len(reps)} reps)", flush=True)
    return rows


def summarize(df):
    skip = {"design", "stage", "n", "rep", "arch"}
    out = []
    for (d, s, n), g in df.groupby(["design", "stage", "n"], sort=True):
        base = dict(design=d, stage=s, n=n, reps=len(g), archs=";".join(sorted(g.arch.unique())))
        for col in df.columns:
            if col in skip or not np.issubdtype(df[col].dtype, np.number):
                continue
            x = g[col].dropna()
            if x.empty:
                continue
            out.append(dict(base, metric=col, median=float(x.median()), max=float(x.max()), min=float(x.min())))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    rows = toy_rows()
    print(f"toy elapsed {time.time()-t0:.0f}s", flush=True)
    rows += case3_rows()
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "by_fit.csv", index=False)
    summ = summarize(df)
    summ.to_csv(HERE / "summary.csv", index=False)
    key = ["max_abs_param", "max_abs_weight", "max_abs_bias", "frac_gt1", "total_params", "nonzero_params",
           "mhat_fresh_min", "mhat_fresh_max", "m0_fresh_min", "m0_fresh_max", "sup_abs_m0_ref",
           "sup_abs_mhat_fresh_over_sup_m0"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summ[summ.metric.isin(key)].pivot_table(index=["design", "stage", "n"], columns="metric",
                                                     values="median").round(4).to_string())
    print(summ[summ.metric.isin(key)].pivot_table(index=["design", "stage", "n"], columns="metric",
                                                     values="max").round(4).to_string())
    print("archs:", {k: list(v) for k, v in df.groupby(["design", "n"]).arch.unique().items()})
    lay = [c for c in df.columns if c.startswith("L") and ("maxabs" in c or "frac" in c)] + ["count_ge_0p999", "raw_max_abs_nonpar"]
    print(df.groupby(["design", "stage", "n"])[lay].median().round(3).to_string())
    print(df.groupby(["design", "stage", "n"])[lay].max().round(3).to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
