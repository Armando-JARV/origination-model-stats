"""Unit tests for drift detection functions using synthetic data and Lending Club data."""

import numpy as np
import polars as pl
import pytest
from drift.metrics_computation import (
    compare_datasets_drift,
    compute_drift_summary,
    compute_ks,
    compute_psi,
    compute_psi_table,
    compute_temporal_drift,
)


def test_psi_identical_distribution():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=10_000)
    tgt = np.random.normal(loc=0.0, scale=1.0, size=10_000)

    psi = compute_psi(ref, tgt, n_bins=10)
    assert psi < 0.02, f"Expected PSI for identical distribution to be near 0, got {psi}"


def test_ks_identical_distribution():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=5_000)
    tgt = np.random.normal(loc=0.0, scale=1.0, size=5_000)

    stat, pval = compute_ks(ref, tgt)
    assert stat < 0.05, f"Expected small KS statistic, got {stat}"
    assert pval > 0.05, f"Expected high p-value for identical distributions, got {pval}"


def test_psi_and_ks_with_drift():
    np.random.seed(42)
    ref = np.random.normal(loc=0.0, scale=1.0, size=10_000)
    # Significant shift: mean moved by 1.2 std dev
    tgt = np.random.normal(loc=1.2, scale=1.2, size=10_000)

    psi = compute_psi(ref, tgt, n_bins=10)
    stat, pval = compute_ks(ref, tgt)

    assert psi > 0.25, f"Expected PSI > 0.25 (significant drift), got {psi}"
    assert stat > 0.3, f"Expected large KS statistic, got {stat}"
    assert pval < 1e-10, f"Expected near-zero p-value, got {pval}"


def test_psi_polars_series():
    ref_series = pl.Series("feature", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    tgt_series = pl.Series("feature", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    psi = compute_psi(ref_series, tgt_series, n_bins=5)
    assert psi < 0.01


def test_psi_table_structure():
    np.random.seed(42)
    ref = pl.Series("ref", np.random.exponential(scale=2.0, size=1_000))
    tgt = pl.Series("tgt", np.random.exponential(scale=3.0, size=1_000))

    table = compute_psi_table(ref, tgt, n_bins=10)

    assert isinstance(table, pl.DataFrame)
    expected_cols = {"bin", "ref_count", "target_count", "ref_pct", "target_pct", "psi_contribution"}
    assert set(table.columns) == expected_cols
    assert len(table) <= 10
    assert table["psi_contribution"].sum() >= 0.0


def test_tied_quantiles_handling():
    # Distribution with many 0s (typical for delinq_2yrs or sparse credit features)
    ref = pl.Series("tied", [0.0] * 800 + [1.0] * 100 + [2.0] * 100)
    tgt = pl.Series("tied", [0.0] * 500 + [1.0] * 250 + [2.0] * 250)

    psi = compute_psi(ref, tgt, n_bins=10)
    stat, pval = compute_ks(ref, tgt)

    assert not np.isnan(psi)
    assert not np.isnan(stat)
    assert psi > 0.0


def test_compute_drift_summary():
    np.random.seed(42)
    df_ref = pl.DataFrame({
        "var_stable": np.random.normal(0, 1, 1000),
        "var_drifted": np.random.normal(0, 1, 1000),
    })
    df_tgt = pl.DataFrame({
        "var_stable": np.random.normal(0, 1, 1000),
        "var_drifted": np.random.normal(2, 1.5, 1000),
    })

    summary = compute_drift_summary(df_ref, df_tgt, n_bins=10)

    assert isinstance(summary, pl.DataFrame)
    assert summary.filter(pl.col("feature") == "var_stable")["drift_alert"][0] == "No Drift"
    assert summary.filter(pl.col("feature") == "var_drifted")["drift_alert"][0] == "Significant Drift"


def test_compare_datasets_drift():
    ref = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 100})
    tgt1 = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 100})
    tgt2 = pl.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0] * 100})

    res = compare_datasets_drift(ref, {"test": tgt1, "validation": tgt2}, columns=["x"])
    assert "comparison" in res.columns
    assert set(res["comparison"].to_list()) == {"test", "validation"}
    assert res.filter(pl.col("comparison") == "test")["drift_alert"][0] == "No Drift"
    assert res.filter(pl.col("comparison") == "validation")["drift_alert"][0] == "Significant Drift"


def test_compute_temporal_drift():
    ref = pl.DataFrame({
        "issue_d": ["2012-01-01"] * 500,
        "score": np.random.normal(0, 1, 500),
    })
    monitoring = pl.DataFrame({
        "issue_d": ["2013-01-01"] * 250 + ["2013-04-01"] * 250,
        "score": list(np.random.normal(0, 1, 250)) + list(np.random.normal(2, 1, 250)),
    })

    temp = compute_temporal_drift(ref, monitoring, date_col="issue_d", freq="3mo", columns=["score"])
    assert len(temp) == 2
    assert "window" in temp.columns
    # First window should be stable, second shifted
    assert temp["drift_alert"][0] == "No Drift"
    assert temp["drift_alert"][1] == "Significant Drift"

