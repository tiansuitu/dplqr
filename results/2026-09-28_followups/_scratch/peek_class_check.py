import pandas as pd
from pathlib import Path
here = Path(__file__).resolve().parents[1] / "class_check"
s = pd.read_csv(here / "summary.csv")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
print(s.groupby(["design","stage","n"]).agg(reps=("reps","first"), archs=("archs","first")).to_string())
keys = ["total_params","nonzero_params","n_masked","max_abs_param","max_abs_weight","max_abs_bias","frac_gt1","count_gt1",
        "count_ge_0p999","frac_ge_0p999","mhat_train_min","mhat_train_max","m0_train_min","m0_train_max",
        "mhat_fresh_min","mhat_fresh_max","m0_fresh_min","m0_fresh_max","sup_abs_m0_ref","sup_abs_mhat_fresh_over_sup_m0",
        "mhat_train_max_over_sup_m0","n_hidden_layers","max_hidden_width","raw_max_abs_nonpar"]
for stat in ["median","max","min"]:
    print("=====", stat)
    p = s[s.metric.isin(keys)].pivot_table(index=["design","stage","n"], columns="metric", values=stat)
    print(p.round(4).T.to_string())
b = pd.read_csv(here / "by_fit.csv")
print("by_fit rows", len(b), b.groupby(["design","stage","n"]).size().to_dict())
print("frac of fits with max_abs_param>1:", b.assign(o=b.max_abs_param>1+1e-12).groupby(["design","stage","n"]).o.mean().round(3).to_dict())
print("case3 frac fits with any param exactly >=0.999:", b[b.design=="case3"].assign(o=b.count_ge_0p999>0).groupby("n").o.mean().to_dict())
print("toy frac fits with mhat fresh sup > sup m0:", b[b.design=="toy"].assign(o=b.sup_abs_mhat_fresh_over_sup_m0>1).groupby(["stage","n"]).o.mean().to_dict())
