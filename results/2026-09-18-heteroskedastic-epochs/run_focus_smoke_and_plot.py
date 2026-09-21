"""Focused smoke on positive-drift hetero family + figures."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("het", HERE / "heteroskedastic_epochs.py")
het = importlib.util.module_from_spec(spec)
sys.modules["het"] = het
spec.loader.exec_module(het)

FOCUS = [
    "mild_s1_t3",
    "paper6_s1_t3",
    "paper6_s10_t3",
    "paper4_s10_t3",
    "mild_x10_s1_t3",
    "interact_s1_t3",
    "z_spike_t3",
    "steep_s1_t3",
]

OUT = HERE / "smoke_focus"
FIG = OUT / "figures"


def run_focus():
    verify_designs = {d.name: d for d in het.default_designs(True)}
    missing = [n for n in FOCUS if n not in verify_designs]
    if missing:
        raise SystemExit(f"Missing designs: {missing}")

    class Args:
        verify = False
        reps = 3
        n_list = "400"
        epochs = "10,20,40,60,80,100"
        seed = 20260918
        designs = ",".join(FOCUS)
        output_name = "smoke_focus"

    # Force the focused Design objects even when verify=False.
    orig = het.default_designs
    het.default_designs = lambda verify: [verify_designs[n] for n in FOCUS]
    try:
        het.run(Args())
    finally:
        het.default_designs = orig


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT / "summary.csv")
    drift = pd.read_csv(OUT / "epoch_drift.csv")
    raw = pd.read_csv(OUT / "raw_checkpoints.csv")
    theta0 = het.THETA0

    # 1) |bias| vs epoch
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, coef in zip(axes, (1, 2)):
        sub = summary[summary["coef"] == coef]
        for design, g in sub.groupby("design"):
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g["bias_vs_theta0"].abs(), marker="o", ms=4, label=design)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"|bias| vs $\theta_0$")
        ax.set_title(rf"$\theta_{coef}$")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Focused smoke: absolute bias vs epoch (n=400, 3 reps)")
    fig.tight_layout()
    p1 = FIG / "abs_bias_vs_epoch.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # 2) mean theta_hat path
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, coef, truth in zip(axes, (1, 2), theta0):
        sub = summary[summary["coef"] == coef]
        for design, g in sub.groupby("design"):
            g = g.sort_values("epoch")
            ax.plot(g["epoch"], g["mean_theta"], marker="o", ms=4, label=design)
        ax.axhline(truth, color="k", ls="--", lw=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(rf"mean $\hat\theta_{coef}$")
        ax.set_title(rf"$\hat\theta_{coef}$ path (dashed = $\theta_0$)")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    p2 = FIG / "theta_hat_vs_epoch.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # 3) delta |bias| bars
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = drift.copy()
    d["label"] = d["design"] + " / coef" + d["coef"].astype(str)
    d = d.sort_values("abs_bias_delta", ascending=True)
    colors = np.where(d["abs_bias_delta"].to_numpy() > 0, "#d62728", "#2ca02c")
    ax.barh(d["label"], d["abs_bias_delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta$|bias| (last epoch - first)")
    ax.set_title("Red = bias grew with more epochs")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    p3 = FIG / "abs_bias_delta.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # 4) replication spaghetti for top positive-drift designs on coef1
    top = (
        drift[drift["coef"] == 1]
        .sort_values("abs_bias_delta", ascending=False)
        .head(4)["design"]
        .tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, design in zip(axes.ravel(), top):
        sub = raw[raw["design"] == design]
        for rep, g in sub.groupby("replication"):
            g = g.sort_values("epoch")
            ax.plot(
                g["epoch"],
                (g["theta_hat_1"] - theta0[0]).abs(),
                alpha=0.9,
                label=f"rep {rep}",
            )
        ax.set_title(design)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"|$\hat\theta_1 - \theta_{0,1}$|")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(r"Replication paths for strongest positive-drift designs")
    fig.tight_layout()
    p4 = FIG / "replication_abs_error_theta1.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    print("FIGURES")
    for p in (p1, p2, p3, p4):
        print(p)
    print("DRIFT_COEF1")
    print(
        drift[drift["coef"] == 1]
        .sort_values("abs_bias_delta", ascending=False)[
            ["design", "abs_bias_first", "abs_bias_last", "abs_bias_delta", "mean_hetero_ratio"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_focus()
    make_figures()
