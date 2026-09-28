"""Case 3: cross-fitted m_hat (2 folds) + oracle phi* joint (theta, t) check-loss LP.

For each saved Case-3 replication (epochs=200) we regenerate the identical dataset
(psi_n_if_helpers.regenerate; hash checked against fits.pt), split ALL n observations
into 2 folds (fixed seed), and on each fold k:
  * train the joint (theta, m) DPLQR net with exactly the original Case-3 "main" stage:
    seed_all(seed) -> dqNetSparse(2, 8, 0, [2, 32], sparseRatio=0.5) -> linLinear.reset_parameters()
    -> fit_fixed (torchtuples Adam, lr=0.005, batch 128, 200 epochs, shuffle, no early stopping;
    validation is monitor-only) -> clip_neural_weights.  fit_fixed/EpochCounter are extracted
    verbatim (ast) from asymptotic_normality_case3.ipynb; the scaler is fit on fold-k Z only.
  * on the other fold, hold m_hat_k fixed and solve
      phi_lp  : min_{theta,t} sum rho(Y - m_hat_k - X theta - phi*(Z) t)
      theta_lp: min_theta     sum rho(Y - m_hat_k - X theta)
Estimates: DML1-style average over folds (primary) and a DML2 pooled LP (extra), plus the
average of the two in-fold Adam thetas ("adam_half").
Writes only into results/2026-09-28_followups/case3_xfit_m_200ep/.
"""
from __future__ import annotations
import argparse, ast, csv, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C3 = ROOT / "results" / "2026-09-11_asymptotic_normality_case3"
NB = C3 / "asymptotic_normality_case3.ipynb"
sys.path.insert(0, str(C3))

EPOCHS = 200
HP = dict(depth=2, width=32, batch_size=128, lr=0.005)   # = run/run_config.json identity.hp
FOLD_SEED_OFFSET = 7_000_000      # fold split rng = default_rng(data_seed + 7_000_000)
FOLD_FIT_OFFSET = 100_000         # fold-k training seed = fit_seed + 100_000 * (k + 1)
OUT_CSV = HERE / "replication_results.csv"
ERR_LOG = HERE / "errors.log"

G = {}  # per-process globals


def init_worker(threads=1):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    import psi_n_if_helpers as H
    root, bridge, orig = H.load_stack(C3)
    import dqAux
    from torchtuples import Model
    import torchtuples as tt
    # extract EpochCounter + fit_fixed verbatim from the Case-3 notebook (never run its cells)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("EpochCounter", "fit_fixed"):
                if node.name not in found:
                    found[node.name] = node
    assert set(found) == {"EpochCounter", "fit_fixed"}
    ns = dict(np=np, math=math, torch=torch, tt=tt, Model=Model, HP=dict(HP))
    exec(compile(ast.Module(body=[found["EpochCounter"], found["fit_fixed"]], type_ignores=[]), str(NB), "exec"), ns)
    G.update(H=H, bridge=bridge, orig=orig, dqAux=dqAux, torch=torch, Model=Model,
             fit_fixed=ns["fit_fixed"])


def train_main(x, z_scaled, y, xv, zv_scaled, yv, seed):
    """Exact replica of the Case-3 'main' stage (fit_replication.stage + main_net)."""
    torch, dq, orig = G["torch"], G["dqAux"], G["orig"]
    orig.seed_all(seed)
    net = dq.dqNetSparse(2, 8, torch.zeros((1, 2), dtype=torch.float32),
                         [HP["depth"], HP["width"]], sparseRatio=0.5)
    net.linLinear.reset_parameters()
    train_x = orig.tensor_pair(x, z_scaled)
    val_x = orig.tensor_pair(xv, zv_scaled)
    model, details = G["fit_fixed"](net, dq.checkLoss(tau=0.5), train_x, y, val_x, yv, EPOCHS)
    orig.clip_neural_weights(model.net)
    model.net.eval()
    return model.net, details


def net_theta_m(net, x, z_scaled):
    torch = G["torch"]
    with torch.no_grad():
        m = net(torch.zeros((len(z_scaled), 2), dtype=torch.float32),
                torch.tensor(z_scaled, dtype=torch.float32)).numpy().reshape(-1).astype(float)
    th = net.linLinear.weight.detach().numpy().reshape(-1).astype(float)
    return th, m


def verify_original(n, rep):
    """Retrain the ORIGINAL main stage (0.8n train, fit_seed) and compare with saved fits.pt."""
    H, torch = G["H"], G["torch"]
    data = H.regenerate(G["orig"], n, rep)
    fits = torch.load(H.replication_dir(C3, n, EPOCHS, rep) / "fits.pt", map_location="cpu", weights_only=False)
    t0 = time.time()
    net, _ = train_main(data.x_train, data.scaler.transform(data.z_train), data.y_train,
                        data.x_val, data.scaler.transform(data.z_val), data.y_val, data.fit_seed)
    secs = time.time() - t0
    saved = fits["stages"]["main"]["state"]
    new = net.state_dict()
    maxdiff = max(float((new[k].float() - saved[k].float()).abs().max()) for k in saved)
    return dict(max_abs_state_diff=maxdiff, seconds=secs,
                theta_new=net.linLinear.weight.detach().numpy().ravel().tolist(),
                theta_saved=saved["linLinear.weight"].numpy().ravel().tolist())


def one_rep(task):
    n, rep, deadline = task
    if deadline is not None and time.time() > deadline:
        return dict(n=n, rep=rep, skipped=True)
    try:
        return _one_rep(n, rep)
    except Exception:
        return dict(n=n, rep=rep, error=traceback.format_exc())


def _one_rep(n, rep):
    from scipy.stats import t as student_t
    from sklearn.preprocessing import StandardScaler
    H, orig, torch, bridge = G["H"], G["orig"], G["torch"], G["bridge"]
    TAU, TH0, F0 = H.TAU, H.THETA_0, H.ORACLE_F0
    Z975 = 1.959963984540054
    t_start = time.time()
    data = H.regenerate(orig, n, rep)
    fits = torch.load(H.replication_dir(C3, n, EPOCHS, rep) / "fits.pt", map_location="cpu", weights_only=False)
    if fits["data_sha256"] != data.data_sha256:
        raise ValueError("data hash mismatch")
    x, z, y = data.x, data.z, data.y
    phi = H.true_phi_star_case3(z)
    m0 = orig.nonlinear_truth(z, 3)
    eps = y - x @ TH0 - m0
    perm = np.random.default_rng(data.data_seed + FOLD_SEED_OFFSET).permutation(n)
    folds = [np.sort(perm[: n // 2]), np.sort(perm[n // 2:])]

    fold = []
    for k in (0, 1):
        I, O = folds[k], folds[1 - k]          # I = training fold, O = held-out fold
        sc = StandardScaler().fit(z[I])
        seed = data.fit_seed + FOLD_FIT_OFFSET * (k + 1)
        t0 = time.time()
        # validation = held-out fold, MONITOR ONLY (no effect on weights, as in the original)
        net, det = train_main(x[I], sc.transform(z[I]), y[I], x[O], sc.transform(z[O]), y[O], seed)
        fit_s = time.time() - t0
        th_adam, m_in = net_theta_m(net, x[I], sc.transform(z[I]))
        _, m_out = net_theta_m(net, x[O], sc.transform(z[O]))
        off = y[O] - m_out
        th_tlp = H.check_linprog(off, x[O])
        b = H.check_linprog(off, np.column_stack([x[O], phi[O]]))
        fold.append(dict(I=I, O=O, seed=seed, fit_s=fit_s, det=det, th_adam=th_adam, m_in=m_in, m_out=m_out,
                         th_tlp=th_tlp, th_plp=b[:2], t_hat=b[2:]))

    # pooled DML2 LP over all n with fold-specific offsets
    O_all = np.concatenate([f["O"] for f in fold])
    m_all = np.concatenate([f["m_out"] for f in fold])       # m_hat_{-k(i)}(Z_i), aligned with O_all
    xO, phiO, yO, m0O, epsO = x[O_all], phi[O_all], y[O_all], m0[O_all], eps[O_all]
    bp = H.check_linprog(yO - m_all, np.column_stack([xO, phiO]))

    xt = xO - phiO
    ginv = np.linalg.inv(xt.T @ xt / n)
    d_out = m_all - m0O
    f_eff = float(np.mean(student_t.pdf(d_out, df=3)))
    m_in_all = np.concatenate([f["m_in"] for f in fold])
    I_all = np.concatenate([f["I"] for f in fold])
    d_in = m_in_all - m0[I_all]
    # in-fold Adam training residuals (overfitting diagnostic)
    r_in = np.concatenate([y[f["I"]] - x[f["I"]] @ f["th_adam"] - f["m_in"] for f in fold])

    def resid(key, with_t):
        rs = []
        for f in fold:
            r = y[f["O"]] - x[f["O"]] @ f[key] - f["m_out"]
            if with_t:
                r = r - phi[f["O"]] @ f["t_hat"]
            rs.append(r)
        return np.concatenate(rs)

    variants = {
        "xfit_phi_lp": (np.mean([f["th_plp"] for f in fold], 0), resid("th_plp", True), resid("th_plp", False),
                        np.mean([f["t_hat"] for f in fold], 0)),
        "xfit_theta_lp": (np.mean([f["th_tlp"] for f in fold], 0), resid("th_tlp", False), None, np.zeros(2)),
        "adam_half": (np.mean([f["th_adam"] for f in fold], 0), resid("th_adam", False), None, np.zeros(2)),
        "xfit_phi_lp_pooled": (bp[:2], yO - xO @ bp[:2] - m_all - phiO @ bp[2:], yO - xO @ bp[:2] - m_all, bp[2:]),
    }
    keymap = {"xfit_phi_lp": "th_plp", "xfit_theta_lp": "th_tlp", "adam_half": "th_adam"}
    rows = []
    N_old = int(0.8 * n)
    common = dict(n=n, rep=rep, N_old=N_old, n_total=n, n_fold_train=len(folds[0]),
                  data_seed=data.data_seed, fold_seed=data.data_seed + FOLD_SEED_OFFSET,
                  fit_seed_f1=fold[0]["seed"], fit_seed_f2=fold[1]["seed"],
                  f0=F0, f_eff=f_eff,
                  m_l2_ho=float(np.sqrt(np.mean(d_out ** 2))), m_mean_ho=float(d_out.mean()),
                  m_l2_in=float(np.sqrt(np.mean(d_in ** 2))),
                  mean_abs_eps_ho=float(np.abs(epsO).mean()),
                  mean_abs_resid_adam_in=float(np.abs(r_in).mean()),
                  mean_abs_eps_in=float(np.abs(eps[I_all]).mean()),
                  fit_seconds_f1=fold[0]["fit_s"], fit_seconds_f2=fold[1]["fit_s"],
                  last_train_loss_f1=fold[0]["det"]["last_epoch_train_loss"],
                  last_train_loss_f2=fold[1]["det"]["last_epoch_train_loss"],
                  last_ho_loss_f1=fold[0]["det"]["last_epoch_validation_loss"],
                  last_ho_loss_f2=fold[1]["det"]["last_epoch_validation_loss"])
    for var, (th, r, r_noT, t_hat) in variants.items():
        fhat = float(bridge.residual_density_zero(r))
        fhat_noT = float(bridge.residual_density_zero(r_noT)) if r_noT is not None else fhat
        s = TAU - (r < 0)
        orth = -(s[:, None] * xt).mean(0)
        row = dict(common, variant=var, fhat=fhat, fhat_noT=fhat_noT,
                   mean_abs_resid_ho=float(np.abs(r).mean()), median_resid_ho=float(np.median(r)))
        for j in range(2):
            err = th[j] - TH0[j]
            row[f"theta_{j+1}"] = float(th[j])
            row[f"sn_xi_{j+1}"] = math.sqrt(N_old) * err       # sqrt(0.8 n) scale = old convention
            row[f"rn_xi_{j+1}"] = math.sqrt(n) * err           # sqrt(n) scale (xfit uses all n)
            row[f"t_hat_{j+1}"] = float(t_hat[j])
            row[f"sn_orth_score_{j+1}"] = math.sqrt(n) * float(orth[j])
            if var in keymap:
                row[f"theta_f1_{j+1}"] = float(fold[0][keymap[var]][j])
                row[f"theta_f2_{j+1}"] = float(fold[1][keymap[var]][j])
            else:
                row[f"theta_f1_{j+1}"] = row[f"theta_f2_{j+1}"] = float("nan")
            for lab, f in (("f0", F0), ("feff", f_eff), ("fhat", fhat)):
                se = math.sqrt(TAU * (1 - TAU) * ginv[j, j] / n) / f
                row[f"se_{lab}_{j+1}"] = se
                row[f"cover_{lab}_{j+1}"] = float(abs(err) <= Z975 * se)
        rows.append(row)
    for row in rows:
        row["rep_seconds"] = time.time() - t_start
    return dict(n=n, rep=rep, rows=rows)


def done_keys():
    if not OUT_CSV.exists():
        return set()
    import pandas as pd
    d = pd.read_csv(OUT_CSV, usecols=["n", "rep", "variant"])
    return set(map(tuple, d.loc[d.variant == "xfit_phi_lp", ["n", "rep"]].to_numpy().tolist()))


def append_rows(rows):
    new = not OUT_CSV.exists()
    if not new:
        with OUT_CSV.open("r", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
    else:
        header = list(rows[0].keys())
    with OUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="raise")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush(); os.fsync(fh.fileno())


def build_tasks(plan):
    """plan: list of (n, max_rep). Reps in saved-fit order (1, 2, ...), interleaved across n.
    Phase 1: reps 1..100 for every n; phase 2: reps 101..max for n <= 1000;
    phase 3: reps 101..max for n >= 2000 (so partial results are balanced and usable)."""
    import psi_n_if_helpers as H
    avail = {(n, r) for n, e, r in H.list_available(C3) if e == EPOCHS}
    tasks = []
    maxr = max(m for _, m in plan)
    phases = [((1, 100), lambda n: True), ((101, maxr), lambda n: n <= 1000), ((101, maxr), lambda n: n > 1000)]
    for (lo, hi), keep in phases:
        for rep in range(lo, hi + 1):
            for n, m in plan:
                if keep(n) and rep <= m and (n, rep) in avail:
                    tasks.append((n, rep))
    return tasks


def keep_awake():
    """Ask Windows not to sleep while this process runs (auto-cleared on exit)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def log(msg):
    print(msg, flush=True)
    with (HERE / "run_log.txt").open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", default=["500:200", "1000:200", "2000:100", "4000:100"],
                    help="n:max_rep pairs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--minutes", type=float, default=None, help="wall budget; later tasks are skipped")
    ap.add_argument("--verify", action="store_true", help="retrain original rep and compare to fits.pt")
    a = ap.parse_args()
    plan = [tuple(int(v) for v in p.split(":")) for p in a.plan]
    if a.verify:
        init_worker(1)
        for n, rep in plan:
            print("verify", n, rep, verify_original(n, rep), flush=True)
        return
    done = done_keys()
    tasks = [t for t in build_tasks(plan) if t not in done]
    deadline = time.time() + 60 * a.minutes if a.minutes else None
    log(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} keep_awake={keep_awake()}: "
        f"{len(done)} done, {len(tasks)} to run, workers={a.workers}, plan={a.plan}, minutes={a.minutes}")
    t0 = time.time()
    if a.workers <= 1:
        init_worker(1)
        it = (one_rep((n, r, deadline)) for n, r in tasks)
        pool = None
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(1,))
        it = pool.imap_unordered(one_rep, [(n, r, deadline) for n, r in tasks], chunksize=1)
    k = skipped = 0
    for res in it:
        if res.get("skipped"):
            skipped += 1; continue
        if "error" in res:
            with ERR_LOG.open("a", encoding="utf-8") as fh:
                fh.write(f"n={res['n']} rep={res['rep']}\n{res['error']}\n")
            log(f"ERROR n={res['n']} rep={res['rep']}")
            continue
        append_rows(res["rows"])
        k += 1
        r0 = res["rows"][0]
        log(f"[{time.time()-t0:7.0f}s] n={res['n']} rep={res['rep']} done={k} "
              f"rep_s={r0['rep_seconds']:.0f} fits={r0['fit_seconds_f1']:.0f}/{r0['fit_seconds_f2']:.0f} "
              f"sn_xi1(phi_lp)={r0['sn_xi_1']:.2f}")
    if pool is not None:
        pool.close(); pool.join()
    log(f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}: {k} reps written, {skipped} skipped by deadline, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
