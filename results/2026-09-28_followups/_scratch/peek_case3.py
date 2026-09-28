import sys, ast, json
from pathlib import Path
R = Path(r"C:\Users\Tiansui Tu\Documents\GitHub\dplqr")
E = R/"results"/"2026-09-11_asymptotic_normality_case3"
sys.path.insert(0, str(E))
import psi_n_if_helpers as ph
root, bridge, orig = ph.load_stack(E)
nodes = bridge.selected_source()
for k in ["nonlinear_truth","generate_covariates","generate_dataset","clip_neural_weights","dplqr_standard_errors","train_dplqr_once","fit_dplqr"]:
    print("=====", k); print(ast.unparse(nodes[k]))
print("===== density_only.R"); print((R/"results"/"2026-09-10-asymptotic-normality"/"density_only.R").read_text())
av = ph.list_available(E)
from collections import Counter
print(Counter((n,e) for n,e,r in av))
print("reps per n sample:", {n: sorted(r for nn,e,r in av if nn==n)[:5] for n in sorted(set(a[0] for a in av))})
import torch
n,e,r = av[0]
f = torch.load(ph.replication_dir(E,n,e,r)/"fits.pt", map_location="cpu", weights_only=False)
def show(d, pre=""):
    for k,v in d.items():
        if isinstance(v, dict): print(pre+str(k)+":"); show(v, pre+"  ")
        elif hasattr(v, "shape"): print(pre+str(k), tuple(v.shape))
        else: print(pre+str(k), repr(v)[:100])
show(f)
print((E/"run"/"run_config.json").read_text()[:3000])
