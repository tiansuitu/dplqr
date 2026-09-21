"""Confirm Theorist mechanism: Case 6 c=20 with vs without x1 in sigma."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("het", HERE / "heteroskedastic_epochs.py")
het = importlib.util.module_from_spec(spec)
sys.modules["het"] = het
spec.loader.exec_module(het)


def sigma_case6_no_x1(x, z):
    return (x[:, 1] / 3.0) + 3.0 * norm.cdf(np.sum(z - 1.0, axis=1) / 5.0)


def sigma_case6_x1_only(x, z):
    return (x[:, 0] / 3.0) + 3.0 * norm.cdf(np.sum(z - 1.0, axis=1) / 5.0)


het.SIGMA = dict(getattr(het, "SIGMA"))
het.SIGMA["case6_no_x1"] = sigma_case6_no_x1
het.SIGMA["case6_x1_only"] = sigma_case6_x1_only

C = 20.0
DESIGNS = [
    het.Design("case6_full_c20", "paper_case6", C, "t3"),
    het.Design("case6_nox1_c20", "case6_no_x1", C, "t3"),
    het.Design("case6_x1only_c20", "case6_x1_only", C, "t3"),
]
OUT = HERE / "smoke_case6_x1_ablation"
FIG = OUT / "figures"
LABELS = {
    "case6_full_c20": "full Case6 (x1+x2)",
    "case6_nox1_c20": "no x1 in σ",
    "case6_x1only_c20": "x1-only linear in σ",
}


def run():
    class Args:
        verify = False
        reps = 20
        n_list = "400"
        epochs = "10,20,40,60,80,100,150,200"
        seed = 20260918
        designs = ",".join(d.name for d in DESIGNS)
        output_name = "smoke_case6_x1_ablation"

    orig = het.default_designs
    het.default_designs = lambda verify: list(DESIGNS)
    try:
        het.run(Args())
    finally:
        het.default_designs = orig


def plot():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT / "summary.csv")
    drift = pd.read_csv(OUT / "epoch_drift.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, coef in zip(axes, (1, 2)):
        sub = summary[summary["coef"] == coef]
        for design, g in sub.groupby("design"):
            g = g.sort_values("epoch")
            ax.plot(
                g["epoch"],
                g["bias_vs_theta0"].abs(),
                marker="o",
                ms=4,
                label=LABELS.get(design, design),
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\overline{\hat\theta}-\theta_0|$")
        ax.set_title(rf"$\theta_{coef}$ at c=20")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Ablation: remove x1 from σ (n=400, 20 reps)")
    fig.tight_layout()
    p1 = FIG / "x1_ablation_abs_bias_vs_epoch.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    d = drift.copy()
    d["label"] = d["design"].map(LABELS).fillna(d["design"]) + " / th" + d["coef"].astype(str)
    d = d.sort_values("abs_bias_delta", ascending=True)
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in d["abs_bias_delta"]]
    ax.barh(d["label"], d["abs_bias_delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta|$bias$|$ (epoch 200 - 10)")
    ax.set_title("Red = |bias| grew with epochs")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    p2 = FIG / "x1_ablation_abs_bias_delta.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    print("FIGURES")
    print(p1)
    print(p2)
    print("DRIFT")
    print(
        drift.sort_values(["coef", "abs_bias_delta"], ascending=[True, False])[
            ["design", "coef", "abs_bias_first", "abs_bias_last", "abs_bias_delta"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
    plot()
