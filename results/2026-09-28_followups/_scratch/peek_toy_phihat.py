import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).resolve().parents[1] / "toy_xfit_phihat"
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
df = pd.read_csv(here / "replication_results.csv")
s = pd.read_csv(here / "summary.csv"); p = pd.read_csv(here / "paired.csv")
print("rows", len(df), df.groupby(["n","method"]).size().unstack().to_string())
print("dup (n,rep,method):", int(df.duplicated(["n","rep","method"]).sum()), " reps per n:", df.groupby("n").rep.nunique().to_dict())
print(s[["n","method","reps","root_n_bias","mcse","root_n_sd","root_n_rmse","coverage_feasible","coverage_oracle","coverage_pop_oracle",
         "mean_root_n_se_feasible","mean_root_n_se_oracle","pop_root_n_sd","mean_f_hat0","mean_abs_root_n_phi_score","mean_abs_S","mean_S",
         "mean_t_hat","mean_phi_l2","mean_m_l2","mean_product"]].round(4).to_string())
print(p.round(4).to_string())
chk = df.dropna(subset=["old_S"]).copy()
chk["dS"] = (chk.S - chk.old_S).abs(); chk["dphi"] = (chk.root_n_phi_score - chk.old_root_n_phi_score).abs()
chk["dth"] = (chk.theta_hat - chk.old_theta_hat).abs()
print(chk.groupby(["n","method"])[["dS","dphi","dth"]].max().to_string())
print("rows with dS>1e-6:", (chk.dS > 1e-6).sum(), "of", len(chk))
print(chk.sort_values("dS", ascending=False)[["n","rep","method","dS","dphi","dth","near_zero_residuals"]].head(8).to_string())
print(df.groupby(["n","method"]).near_zero_residuals.agg(["mean","max"]).unstack().to_string())
print("phi_l2 mlp vs gb mean:", df[df.method=="adam"].groupby("n")[["phi_l2_mlp","phi_l2_gb"]].mean().round(4).to_string())
print("max loss_minus_adam non-adam:", df[df.method!="adam"].loss_minus_adam.max())
print("nan check:", df[["theta_hat","se_feasible","f_hat0"]].isna().sum().to_dict())
