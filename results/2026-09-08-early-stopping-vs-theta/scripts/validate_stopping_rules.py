"""Check replay against actual callbacks and torchtuples validation scoring.

No neural-network optimization is performed. A scalar parameter is set to the
epoch number to make restored checkpoint identity directly observable. Optional
dense trajectory CSVs are checked against the notebook callback as well.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchtuples as tt

from stopping_rules import select_stopping_epoch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dqAux import checkLoss  # noqa: E402


def load_notebook_callback():
    path = ROOT / "results/2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "EarlyStopInMemory":
                namespace = dict(math=math, np=np, torch=torch, tt=tt)
                code = ast.Module(body=[node], type_ignores=[])
                exec(compile(code, str(path), "exec"), namespace)
                return namespace["EarlyStopInMemory"]
    raise RuntimeError("Could not find the authoritative EarlyStopInMemory callback")


class EpochMarkerModel:
    def __init__(self):
        self.net = torch.nn.Linear(1, 1, bias=False)
        self.val_metrics = SimpleNamespace(scores={"loss": {"score": []}})
        with torch.no_grad():
            self.net.weight.zero_()

    def save_model_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model_weights(self, path):
        self.net.load_state_dict(torch.load(path, weights_only=True))


def replay_actual_callback(losses, patience, max_epochs, callback_kind="notebook"):
    model = EpochMarkerModel()
    with tempfile.TemporaryDirectory(prefix="dplqr-stopping-test-") as temporary:
        if callback_kind == "notebook":
            callback = load_notebook_callback()(patience)
        elif callback_kind == "torchtuples":
            callback = tt.callbacks.EarlyStopping(
                patience=patience, min_delta=0.0,
                file_path=Path(temporary) / "weights.pt", load_best=True,
            )
        else:
            raise ValueError(callback_kind)
        callback.give_model(model)
        callback.on_fit_start()
        reason = "max_epochs"
        for epoch, loss in enumerate(losses[:max_epochs], start=1):
            with torch.no_grad():
                model.net.weight.fill_(epoch)
            model.val_metrics.scores["loss"]["score"].append(loss)
            if callback.on_epoch_end():
                reason = "early_stopping"
                break
        callback.on_fit_end()
        return {
            "stop_epoch": epoch,
            "selected_epoch": int(model.net.weight.item()),
            "stopping_reason": reason,
        }


def validate_batch_scoring():
    # The repository uses 200 validation observations and batches 128 + 72.
    # Losses are deliberately unequal to expose equal-batch versus equal-row
    # weighting; the frozen network's output is zero throughout.
    net = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        net.weight.zero_()
    model = tt.Model(net, checkLoss(tau=0.5), device="cpu")
    x = torch.zeros((200, 1), dtype=torch.float32)
    y = torch.cat((torch.ones((128, 1)), torch.full((72, 1), 4.0)))
    direct_batches = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(y), 128):
            direct_batches.append(checkLoss(0.5)(net(x[start:start + 128]), y[start:start + 128]))
    direct = torch.stack(direct_batches).mean().item()
    loader = model.make_dataloader((x, y), batch_size=128, shuffle=False)
    library = model.score_in_batches_dataloader(loader)["loss"]
    empirical = checkLoss(0.5)(torch.zeros_like(y), y).item()
    assert direct == library == 1.25
    assert not math.isclose(empirical, library, abs_tol=1e-6)
    return dict(
        validation_observations=200, batch_sizes=[128, 72],
        direct_float32_mean_batch_loss=direct,
        actual_torchtuples_loss=library,
        full_sample_empirical_loss=empirical,
        convention="Unweighted mean of float32 batch means; no validation shuffle",
    )


def validate_synthetic():
    fixtures = [
        ("monotone_improvement", [5., 4., 3., 2., 1.], 2, 5, (5, 5, "max_epochs")),
        ("two_bad_epochs", [5., 4., 4.5, 4.6, 3.], 2, 5, (4, 2, "early_stopping")),
        ("tie_keeps_earliest", [1., 1., 1., 0.5], 2, 4, (3, 1, "early_stopping")),
        ("improvement_resets_counter", [3., 4., 2., 4., 5., 1.], 2, 6, (5, 3, "early_stopping")),
        ("improvement_smaller_than_1e-8", [1., 1. - 1e-12, 1., 1.], 2, 4, (4, 2, "early_stopping")),
        ("cap_before_patience", [3., 2., 4., 5., 1.], 3, 3, (3, 2, "max_epochs")),
        ("callback_and_cap_same_epoch", [1., 2., 3.], 2, 3, (3, 1, "early_stopping")),
        ("single_epoch_cap", [9., 1.], 5, 1, (1, 1, "max_epochs")),
        ("patience_one", [2., 1., 1., 0.5], 1, 4, (3, 2, "early_stopping")),
    ]
    results = []
    for name, losses, patience, cap, expected in fixtures:
        replay = select_stopping_epoch(losses, patience, cap)
        actual = replay_actual_callback(losses, patience, cap)
        assert tuple(replay[k] for k in actual) == expected, (name, replay)
        assert {k: replay[k] for k in actual} == actual, (name, replay, actual)
        assert replay["best_validation_loss"] == min(losses[:replay["stop_epoch"]])
        library = replay_actual_callback(losses, patience, cap, "torchtuples")
        assert library["stop_epoch"] == actual["stop_epoch"]
        assert library["stopping_reason"] == actual["stopping_reason"]
        results.append(dict(name=name, validation_losses=losses, replay=replay,
                            actual_notebook_callback=actual, actual_torchtuples_callback=library))

    # Cap truncation only alters what is observable, never the trajectory.
    losses = [5., 4., 3., 4., 5., 6., 7., 0.]
    cap_results = [select_stopping_epoch(losses, 2, cap) for cap in (2, 3, 4, 5, 6, 8)]
    assert [(row["stop_epoch"], row["selected_epoch"]) for row in cap_results] == [
        (2, 2), (3, 3), (4, 3), (5, 3), (5, 3), (5, 3),
    ]
    assert list(inspect.signature(select_stopping_epoch).parameters) == [
        "validation_losses", "patience", "max_epochs",
    ]
    # Rejected inputs cannot silently yield a selected model.
    rejected = []
    for name, losses, patience, cap, error in [
        ("insufficient_dense_epochs", [1.], 2, 3, ValueError),
        ("nonfinite_loss", [1., float("nan")], 2, 2, FloatingPointError),
        ("zero_patience", [1.], 0, 1, ValueError),
        ("noninteger_cap", [1.], 1, 1.5, ValueError),
    ]:
        try:
            select_stopping_epoch(losses, patience, cap)
        except error:
            rejected.append(name)
        else:
            raise AssertionError(f"Did not reject {name}")
    return dict(fixtures=results, cap_truncation=cap_results, rejected_inputs=rejected,
                function_inputs=["validation_losses", "patience", "max_epochs"],
                no_truth_test_or_oracle_inputs=True)


def validate_trajectory_csv(path):
    import pandas as pd

    raw = pd.read_csv(path)
    rows = []
    column = "stopping_validation_check_loss"
    if column not in raw:
        raise ValueError(f"Dense trajectory must contain {column}")
    # Two preselected trajectories suffice to check actual-path integration.
    for (case, replicate), frame in list(raw.groupby(["case", "replicate"]))[:2]:
        frame = frame.sort_values("epoch")
        ceiling = int(frame["epoch"].max())
        np.testing.assert_array_equal(frame["epoch"].to_numpy(), np.arange(1, ceiling + 1))
        losses = frame[column].tolist()
        configurations = {(patience, ceiling) for patience in (5, 15, 30, 50, 100, 200)}
        configurations.update((15, cap) for cap in (25, 50, 100, 200, 500, 1000) if cap <= ceiling)
        for patience, cap in sorted(configurations):
            replay = select_stopping_epoch(losses, patience, cap)
            actual = replay_actual_callback(losses, patience, cap)
            assert {k: replay[k] for k in actual} == actual
            rows.append(dict(case=int(case), replicate=int(replicate), **replay))
    return dict(source=str(path.resolve()), checked_configurations=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-csv", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(status="passed", authoritative_callback="main notebook EarlyStopInMemory",
                  synthetic=validate_synthetic(), batch_scoring=validate_batch_scoring(),
                  exact_tie_discrepancy=("Notebook keeps earliest best checkpoint; installed "
                      "torchtuples EarlyStopping rewrites its checkpoint on an equal best loss, "
                      "so it restores the latest tied best. Their patience counters agree."))
    if args.trajectory_csv:
        result["actual_trajectories"] = validate_trajectory_csv(args.trajectory_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
