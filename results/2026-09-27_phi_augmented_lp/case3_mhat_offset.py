"""Split m_hat - m0 into mean offset and SD over Z (train and held-out), Case 3 saved fits."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
C3 = HERE.parent / "2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(C3))
import psi_n_if_helpers as H
_, _, orig = H.load_stack(C3)
EP = 1000; REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(99)
xe, ze, ye = H.independent_eval_sample(orig, 50000, 20260928)
m0e = orig.nonlinear_truth(ze, 3)
rows = []
for n, ep, rep in H.list_available(C3):
    if ep != EP or rep > REPS: continue
    path = H.replication_dir(C3, n, ep, rep)
    fits = torch.load(path / "fits.pt", map_location="cpu", weights_only=False)
    data = H.regenerate(orig, n, rep)
    net = H.load_main_net(fits["stages"]["main"]["state"])
    th, m_hat, _ = H.predict_train(net, data)
    m0 = orig.nonlinear_truth(data.z_train, 3)
    d = m_hat - m0
    de = H.predict_m_on_z(net, data, ze) - m0e
    r = data.y_train - data.x_train @ th - m_hat
    eps = data.y_train - data.x_train @ H.THETA_0 - m0
    rows.append(dict(n=n, rep=rep, mean_d=d.mean(), sd_d=d.std(), l2_d=math.sqrt(np.mean(d*d)),
                     median_d=np.median(d), mean_d_eval=de.mean(), sd_d_eval=de.std(),
                     l2_d_eval=math.sqrt(np.mean(de*de)), median_resid=np.median(r),
                     mean_abs_resid=np.abs(r).mean(), mean_abs_eps=np.abs(eps).mean(),
                     median_eps=np.median(eps)))
df = pd.DataFrame(rows)
df.to_csv(HERE / "case3_run" / "mhat_offset_by_rep.csv", index=False)
g = df.groupby("n").mean(numeric_only=True).drop(columns="rep")
g["share_l2sq_from_mean"] = (df.assign(s=df.mean_d**2/df.l2_d**2).groupby("n").s.mean())
g["sd_of_mean_d_across_reps"] = df.groupby("n").mean_d.std()
g.to_csv(HERE / "case3_run" / "mhat_offset_summary.csv")
pd.set_option("display.width", 250)
print(g.round(3).T.to_string())
