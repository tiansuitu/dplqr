"""Read-only statistical/checkpoint checks of a completed run; no model fits."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_normality import HERE, atomic_json, collect_results


def validate(output):
    metadata = json.loads((output / "run_config.json").read_text())
    assert metadata["status"] == "complete" and metadata["last_error"] is None
    q = metadata["requested_repetitions"]
    expected = set(itertools.product(metadata["requested_cases"], metadata["requested_sample_sizes"], range(1, q+1)))
    raw = pd.read_csv(output / "raw_replications.csv", float_precision="round_trip")
    keys = ["case", "n", "rep"]
    assert set(raw[keys].itertuples(index=False, name=None)) == expected
    assert len(raw) == len(expected) and raw.method.eq("DPLQR").all() and raw.tau.eq(.5).all()
    assert raw.n_train.eq((.8*raw.n).astype(int)).all() and (raw.n_train+raw.n_val).eq(raw.n).all()
    assert np.isfinite(raw.select_dtypes(include="number")).all().all()
    assert (raw[["se_theta1", "se_theta2", "density_zero"]] > 0).all().all()
    np.testing.assert_array_equal(raw.setting_seed, metadata["identity"]["seed"] + raw.case*10_000_000 + raw.n*1000 + raw.rep)
    np.testing.assert_allclose(raw.t_theta1, (raw.theta1-1)/raw.se_theta1, rtol=1e-14, atol=1e-14)
    for column, theta in [(1, 1), (2, -1)]:
        estimate, se = raw[f"theta{column}"], raw[f"se_theta{column}"]
        lower, upper = raw[f"lower_theta{column}"], raw[f"upper_theta{column}"]
        np.testing.assert_allclose(lower, estimate-norm.ppf(.975)*se, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(upper, estimate+norm.ppf(.975)*se, rtol=1e-14, atol=1e-14)
        assert raw[f"covered_theta{column}"].eq((lower <= theta) & (theta <= upper)).all()
    np.testing.assert_allclose(raw.root_n_error_theta1, np.sqrt(raw.n)*(raw.theta1-1), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(raw.root_n_train_error_theta1, np.sqrt(raw.n_train)*(raw.theta1-1), rtol=1e-14, atol=1e-14)
    # Authoritative result files and all 3 saved-network hashes must agree.
    reconstructed = pd.DataFrame(collect_results(output, metadata["identity_sha256"]))
    pd.testing.assert_frame_equal(raw.sort_values(keys).reset_index(drop=True),
                                  reconstructed.sort_values(keys).reset_index(drop=True), check_exact=True)
    summary = pd.read_csv(output / "summary.csv", float_precision="round_trip").set_index(["case", "n"])
    assert len(summary) == len(expected)//q and summary.completed_repetitions.eq(q).all()
    for (case, n), group in raw.groupby(["case", "n"]):
        row = summary.loc[(case, n)]
        estimates = group.theta1.to_numpy()
        error = estimates-1
        studentized = error/group.se_theta1.to_numpy()
        bias = sum(error)/q
        sd = np.sqrt(sum((estimates-estimates.mean())**2)/(q-1)) if q > 1 else np.nan
        checks = dict(bias_theta1=bias, sqrt_n_bias_theta1=np.sqrt(n)*bias,
                      sd_theta1=sd, sqrt_n_sd_theta1=np.sqrt(n)*sd,
                      sqrt_n_train_sd_theta1=np.sqrt(int(.8*n))*sd,
                      mean_se_theta1=group.se_theta1.mean(), rmse_theta1=np.sqrt(sum(error**2)/q),
                      t_mean=studentized.mean(), t_sd=studentized.std(ddof=1) if q > 1 else np.nan,
                      coverage_95_theta1=(np.abs(studentized) <= norm.ppf(.975)).mean())
        for name, value in checks.items():
            np.testing.assert_allclose(row[name], value, rtol=1e-12, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(row[["t_q025", "t_q500", "t_q975"]].to_numpy(dtype=float),
                                   np.quantile(studentized, [.025, .5, .975]), rtol=1e-12, atol=1e-12)
        assert row.coverage_95_wilson_lower <= row.coverage_95_theta1 <= row.coverage_95_wilson_upper
    quantiles = pd.read_csv(output / "quantiles.csv")
    assert len(quantiles) == 3*len(summary)
    figures = list((output / "figures").glob("*.png"))
    assert len(figures) == 5*len(metadata["requested_cases"])
    smoke_count = 0
    smoke_meta = json.loads((HERE / "smoke-test/run_config.json").read_text())
    if smoke_meta["identity"] == metadata["identity"]:
        smoke = pd.read_csv(HERE / "smoke-test/raw_replications.csv", float_precision="round_trip")
        smoke = smoke.set_index(keys)
        overlap = smoke.index.intersection(raw.set_index(keys).index)
        saved = raw.set_index(keys).loc[overlap].reset_index()
        pd.testing.assert_frame_equal(smoke.loc[overlap].reset_index()[raw.columns], saved[raw.columns], check_exact=True)
        smoke_count = len(overlap)
    result = dict(status="passed", replications=len(raw), groups=len(summary), q=q,
                  raw_and_summary_formulas="passed", checkpoint_hashes_checked=3*len(raw),
                  completed_smoke_rows_reused_unchanged=smoke_count, diagnostic_figures=len(figures),
                  normal_quantile_rows=len(quantiles), model_fits_performed=0)
    atomic_json(result, output / "validation.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "pilot")
