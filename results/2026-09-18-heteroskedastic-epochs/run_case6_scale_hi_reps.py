"""Case 6 shape; c in {1,10,20}; 20 reps; |bias| vs epoch through 200."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("het", HERE / "heteroskedastic_epochs.py")
het = importlib.util.module_from_spec(spec)
sys.modules["het"] = het
spec.loader.exec_module(het)

SCALES = (1.0, 10.0, 20.0)
DESIGNS = [het.Design(f"case6_c{c:g}_t3", "paper_case6", float(c), "t3") for c in SCALES]
OUT = HERE / "smoke_case6_scale_reps20"
FIG = OUT / "figures"


def run():
    class Args:
        verify = False
        reps = 20
        n_list = "400"
        epochs = "10,20,40,60,80,100,150,200"
        seed = 20260918
        designs = ",".join(d.name for d in DESIGNS)
        output_name = "smoke_case6_scale_reps20"

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
    raw = pd.read_csv(OUT / "raw_checkpoints.csv")
    theta0 = het.THETA0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, coef in zip(axes, (1, 2)):
        sub = summary[summary["coef"] == coef]
        for design, g in sub.groupby("design"):
            g = g.sort_values("epoch")
            label = design.replace("case6_", "").replace("_t3", "")
            ax.plot(g["epoch"], g["bias_vs_theta0"].abs(), marker="o", ms=4, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\overline{\hat\theta}-\theta_0|$")
        ax.set_title(rf"$\theta_{coef}$")
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="Case6 scale", fontsize=9)
    fig.suptitle("Case 6 shape only: |bias| vs epoch (n=400, 20 reps, t3)")
    fig.tight_layout()
    p1 = FIG / "abs_bias_vs_epoch.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # MCSE bands on |bias| approx via sd of means - use raw to get MC SE of mean
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, coef, j in zip(axes, (1, 2), (1, 2)):
        col = f"theta_hat_{j}"
        for design, g in raw.groupby("design"):
            rows = []
            for epoch, ge in g.groupby("epoch"):
                err = ge[col].to_numpy() - theta0[j - 1]
                bias = err.mean()
                mcse = err.std(ddof=1) / (len(err) ** 0.5)
                rows.append((epoch, abs(bias), mcse))
            rows = sorted(rows)
            ep = [r[0] for r in rows]
            ab = [r[1] for r in rows]
            se = [r[2] for r in rows]
            label = design.replace("case6_", "").replace("_t3", "")
            ax.plot(ep, ab, marker="o", ms=4, label=label)
            ax.fill_between(ep, [a - s for a, s in zip(ab, se)], [a + s for a, s in zip(ab, se)], alpha=0.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\overline{\hat\theta}-\theta_0|$ (± MCSE)")
        ax.set_title(rf"$\theta_{coef}$")
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="c", fontsize=9)
    fig.suptitle("Same |bias| with Monte Carlo SE bands")
    fig.tight_layout()
    p2 = FIG / "abs_bias_vs_epoch_mcse.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    d = drift.copy()
    d["label"] = (
        d["design"].str.replace("case6_", "", regex=False).str.replace("_t3", "", regex=False)
        + " / th"
        + d["coef"].astype(str)
    )
    d = d.sort_values("abs_bias_delta", ascending=True)
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in d["abs_bias_delta"]]
    ax.barh(d["label"], d["abs_bias_delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta|$bias$|$ (epoch 200 - epoch 10)")
    ax.set_title("Red = |bias| grew with more epochs (20 reps)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    p3 = FIG / "abs_bias_delta.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    print("FIGURES")
    for p in (p1, p2, p3):
        print(p)
    print("DRIFT")
    print(
        drift.sort_values(["coef", "abs_bias_delta"], ascending=[True, False])[
            ["design", "coef", "abs_bias_first", "abs_bias_last", "abs_bias_delta", "mean_hetero_ratio"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run()
    plot()
