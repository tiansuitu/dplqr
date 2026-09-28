"""Eigenvalues of oracle Sigma2 = f0 E[(X-phi*)(X-phi*)'] for Zhong-Wang Sim I Case 3 (homoskedastic t3)."""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
C3 = HERE.parent / "2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(C3))
import psi_n_if_helpers as H  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
_, _, orig = H.load_stack(C3)
out = {}
for seed in [1, 2]:
    rng = np.random.default_rng(20260927 + seed)
    x, z, y, _ = orig.generate_dataset(N, 3, rng)
    x = np.asarray(x, float); z = np.asarray(z, float)
    phi = H.true_phi_star_case3(z)
    xt = x - phi
    f0 = H.ORACLE_F0
    s2 = f0 * xt.T @ xt / N
    exx = x.T @ x / N
    covx = np.cov(x.T)
    covxt = np.cov(xt.T)
    corr_xt = np.corrcoef(xt.T)
    # check phi* is E[X|Z]: regress residual on [1, Z] and on phi* (should be ~0)
    D = np.column_stack([np.ones(N), z, phi])
    b = np.linalg.lstsq(D, xt, rcond=None)[0]
    r2 = 1 - ((xt - D @ b)**2).sum(0) / ((xt - xt.mean(0))**2).sum(0)
    ev_s2 = np.linalg.eigvalsh(s2); ev_exx = np.linalg.eigvalsh(exx)
    ev_covx = np.linalg.eigvalsh(covx); ev_covxt = np.linalg.eigvalsh(covxt)
    out[seed] = dict(
        N=N, f0=f0,
        Sigma2=s2.round(5).tolist(), eig_Sigma2=ev_s2.round(5).tolist(),
        eig_EXX=ev_exx.round(5).tolist(),
        ratio_min_Sigma2_over_min_EXX=float(ev_s2[0] / ev_exx[0]),
        ratio_min_Sigma2_over_f0_min_EXX=float(ev_s2[0] / (f0 * ev_exx[0])),
        ratio_min_CovXtilde_over_min_CovX=float(ev_covxt[0] / ev_covx[0]),
        mean_Xtilde=xt.mean(0).round(5).tolist(),
        var_share_Xtilde_over_X=(np.diag(covxt) / np.diag(covx)).round(4).tolist(),
        eig_corr_Xtilde=np.linalg.eigvalsh(corr_xt).round(4).tolist(),
        corr_Xtilde12=float(corr_xt[0, 1]),
        R2_Xtilde_on_Z_phi=r2.round(5).tolist(),
        cond_Sigma2=float(ev_s2[-1] / ev_s2[0]),
    )
    print(json.dumps(out[seed], indent=1), flush=True)
(HERE / "sigma2_eigen_case3.json").write_text(json.dumps(out, indent=2))
