"""Drift detection module for credit origination models using Polars.

Provides functions to compute:
- Population Stability Index (PSI) with configurable bins
- Two-Sample Kolmogorov-Smirnov (KS) test
- Multi-feature drift summary table
- Detailed bin breakdown for PSI
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import polars as pl
from scipy import stats


def compute_psi(
    reference: pl.Series | np.ndarray | Sequence[float],
    target: pl.Series | np.ndarray | Sequence[float],
    n_bins: int = 10,
    epsilon: float = 1e-4,
    strategy: str = "quantile",
) -> float:
    """Calculate the Population Stability Index (PSI) between reference and target distributions.

    Parameters
    ----------
    reference : pl.Series | np.ndarray | Sequence[float]
        The baseline or expected distribution (e.g., training data).
    target : pl.Series | np.ndarray | Sequence[float]
        The comparison or actual distribution (e.g., test or validation data).
    n_bins : int, default 10
        Number of bins for partitioning the distribution.
    epsilon : float, default 1e-4
        Small constant added to zero proportions to prevent log(0) or division by zero.
    strategy : str, default "quantile"
        Binning strategy: "quantile" (equal frequency on reference) or "uniform" (equal width).

    Returns
    -------
    float
        The calculated PSI statistic.

    Notes
    -----
    Interpretation rule-of-thumb:
    - PSI < 0.10: No significant shift / stable
    - 0.10 <= PSI < 0.25: Moderate shift / warning
    - PSI >= 0.25: Significant shift / action required
    """
    table = compute_psi_table(
        reference=reference,
        target=target,
        n_bins=n_bins,
        epsilon=epsilon,
        strategy=strategy,
    )
    return float(table["psi_contribution"].sum())


def compute_psi_table(
    reference: pl.Series | np.ndarray | Sequence[float],
    target: pl.Series | np.ndarray | Sequence[float],
    n_bins: int = 10,
    epsilon: float = 1e-4,
    strategy: str = "quantile",
) -> pl.DataFrame:
    """Calculate detailed bin breakdown for Population Stability Index (PSI).

    Parameters
    ----------
    reference : pl.Series | np.ndarray | Sequence[float]
        Expected baseline distribution.
    target : pl.Series | np.ndarray | Sequence[float]
        Actual comparison distribution.
    n_bins : int, default 10
        Number of bins.
    epsilon : float, default 1e-4
        Small constant to avoid division by zero or log of zero.
    strategy : str, default "quantile"
        "quantile" or "uniform".

    Returns
    -------
    pl.DataFrame
        DataFrame with bin edges, counts, expected/actual proportions, and PSI contribution.
    """
    if isinstance(reference, pl.Series):
        ref_clean = reference.drop_nulls().to_numpy()
    else:
        ref_clean = np.asarray(reference, dtype=float)
        ref_clean = ref_clean[~np.isnan(ref_clean)]

    if isinstance(target, pl.Series):
        tgt_clean = target.drop_nulls().to_numpy()
    else:
        tgt_clean = np.asarray(target, dtype=float)
        tgt_clean = tgt_clean[~np.isnan(tgt_clean)]

    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        raise ValueError("Reference and target must both contain non-null values.")

    # Determine bin edges based on reference distribution
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        raw_edges = np.quantile(ref_clean, quantiles)
        # Deduplicate edges in case of tied values (e.g. lots of zeros)
        edges = np.unique(raw_edges)
        if len(edges) < 2:
            edges = np.array([ref_clean.min() - 1e-6, ref_clean.max() + 1e-6])
    elif strategy == "uniform":
        edges = np.linspace(ref_clean.min(), ref_clean.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}. Choose 'quantile' or 'uniform'.")

    # Set outer edges to infinity to catch out-of-range observations in target
    bin_cutoffs = edges.copy()
    bin_cutoffs[0] = -np.inf
    bin_cutoffs[-1] = np.inf

    # Bin counts
    ref_counts, _ = np.histogram(ref_clean, bins=bin_cutoffs)
    tgt_counts, _ = np.histogram(tgt_clean, bins=bin_cutoffs)

    ref_total = len(ref_clean)
    tgt_total = len(tgt_clean)

    # Proportions
    ref_pct = ref_counts / ref_total
    tgt_pct = tgt_counts / tgt_total

    # Apply epsilon smoothing for 0% bins
    ref_pct_smoothed = np.where(ref_pct == 0, epsilon, ref_pct)
    tgt_pct_smoothed = np.where(tgt_pct == 0, epsilon, tgt_pct)

    # Re-normalize after smoothing
    ref_pct_smoothed = ref_pct_smoothed / ref_pct_smoothed.sum()
    tgt_pct_smoothed = tgt_pct_smoothed / tgt_pct_smoothed.sum()

    # PSI contribution per bin: (Actual - Expected) * ln(Actual / Expected)
    psi_contrib = (tgt_pct_smoothed - ref_pct_smoothed) * np.log(tgt_pct_smoothed / ref_pct_smoothed)

    # Format human-readable bin labels
    bin_labels = []
    num_bins_actual = len(ref_counts)
    for i in range(num_bins_actual):
        lower = f"{edges[i]:.4g}" if i > 0 else f"<= {edges[1]:.4g}"
        upper = f"{edges[i+1]:.4g}" if i < num_bins_actual - 1 else f"> {edges[-2]:.4g}"
        if i == 0:
            bin_labels.append(f"[-inf, {edges[1]:.4g}]")
        elif i == num_bins_actual - 1:
            bin_labels.append(f"({edges[-2]:.4g}, +inf]")
        else:
            bin_labels.append(f"({edges[i]:.4g}, {edges[i+1]:.4g}]")

    return pl.DataFrame({
        "bin": bin_labels,
        "ref_count": ref_counts,
        "target_count": tgt_counts,
        "ref_pct": ref_pct,
        "target_pct": tgt_pct,
        "psi_contribution": psi_contrib,
    })


def compute_ks(
    reference: pl.Series | np.ndarray | Sequence[float],
    target: pl.Series | np.ndarray | Sequence[float],
) -> tuple[float, float]:
    """Calculate the two-sample Kolmogorov-Smirnov (KS) statistic and p-value.

    Parameters
    ----------
    reference : pl.Series | np.ndarray | Sequence[float]
        Baseline distribution.
    target : pl.Series | np.ndarray | Sequence[float]
        Comparison distribution.

    Returns
    -------
    tuple[float, float]
        (ks_statistic, p_value)
        - ks_statistic: Maximum vertical distance between empirical CDFs [0.0, 1.0].
        - p_value: Two-tailed p-value testing null hypothesis that both samples come
          from the same continuous distribution.
    """
    if isinstance(reference, pl.Series):
        ref_clean = reference.drop_nulls().to_numpy()
    else:
        ref_clean = np.asarray(reference, dtype=float)
        ref_clean = ref_clean[~np.isnan(ref_clean)]

    if isinstance(target, pl.Series):
        tgt_clean = target.drop_nulls().to_numpy()
    else:
        tgt_clean = np.asarray(target, dtype=float)
        tgt_clean = tgt_clean[~np.isnan(tgt_clean)]

    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        raise ValueError("Reference and target must both contain non-null values.")

    res = stats.ks_2samp(ref_clean, tgt_clean)
    return float(res.statistic), float(res.pvalue)


def compute_drift_summary(
    reference: pl.DataFrame,
    target: pl.DataFrame,
    columns: list[str] | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> pl.DataFrame:
    """Compute PSI and KS across multiple columns between reference and target DataFrames.

    Parameters
    ----------
    reference : pl.DataFrame
        Baseline DataFrame (e.g., train).
    target : pl.DataFrame
        Target DataFrame (e.g., test or validation).
    columns : list[str] | None, default None
        List of numeric column names to evaluate. If None, auto-selects numeric columns.
    n_bins : int, default 10
        Number of bins for PSI calculation.
    epsilon : float, default 1e-4
        Epsilon for zero-frequency bins.

    Returns
    -------
    pl.DataFrame
        Summary table with columns: [feature, psi, ks_statistic, ks_pvalue, ref_count, target_count, drift_alert]
    """
    if columns is None:
        numeric_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
        columns = [
            col for col, dtype in zip(reference.columns, reference.dtypes)
            if dtype in numeric_types and col in target.columns
        ]

    records = []
    for col in columns:
        ref_s = reference[col].drop_nulls()
        tgt_s = target[col].drop_nulls()

        if len(ref_s) == 0 or len(tgt_s) == 0:
            continue

        # If constant column, skip or handle gracefully
        if ref_s.min() == ref_s.max() and tgt_s.min() == tgt_s.max() and ref_s.min() == tgt_s.min():
            records.append({
                "feature": col,
                "psi": 0.0,
                "ks_statistic": 0.0,
                "ks_pvalue": 1.0,
                "ref_count": len(ref_s),
                "target_count": len(tgt_s),
                "drift_alert": "No Drift",
            })
            continue

        try:
            psi_val = compute_psi(ref_s, tgt_s, n_bins=n_bins, epsilon=epsilon)
            ks_stat, ks_pval = compute_ks(ref_s, tgt_s)

            if psi_val >= 0.25:
                alert = "Significant Drift"
            elif psi_val >= 0.10:
                alert = "Moderate Drift"
            else:
                alert = "No Drift"

            records.append({
                "feature": col,
                "psi": round(psi_val, 6),
                "ks_statistic": round(ks_stat, 6),
                "ks_pvalue": ks_pval,
                "ref_count": len(ref_s),
                "target_count": len(tgt_s),
                "drift_alert": alert,
            })
        except Exception as exc:
            records.append({
                "feature": col,
                "psi": float("nan"),
                "ks_statistic": float("nan"),
                "ks_pvalue": float("nan"),
                "ref_count": len(ref_s),
                "target_count": len(tgt_s),
                "drift_alert": f"Error: {exc}",
            })

    return pl.DataFrame(records)


def compare_datasets_drift(
    reference: pl.DataFrame,
    targets: dict[str, pl.DataFrame],
    columns: list[str] | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> pl.DataFrame:
    """Compare a reference DataFrame against multiple target datasets.

    Parameters
    ----------
    reference : pl.DataFrame
        Baseline reference population (e.g., train).
    targets : dict[str, pl.DataFrame]
        Dictionary mapping target names to target DataFrames (e.g. {"test": test_df, "validation": val_df}).
    columns : list[str] | None, default None
        List of numeric features to evaluate.
    n_bins : int, default 10
        Number of bins for PSI calculation.
    epsilon : float, default 1e-4
        Epsilon smoothing parameter.

    Returns
    -------
    pl.DataFrame
        Consolidated summary DataFrame with comparison column.
    """
    summaries = []
    for name, target_df in targets.items():
        summary = compute_drift_summary(
            reference=reference,
            target=target_df,
            columns=columns,
            n_bins=n_bins,
            epsilon=epsilon,
        )
        summary = summary.with_columns(pl.lit(name).alias("comparison"))
        summaries.append(summary)

    if not summaries:
        return pl.DataFrame()

    full_summary = pl.concat(summaries)
    # Reorder columns with comparison first
    cols = ["comparison"] + [c for c in full_summary.columns if c != "comparison"]
    return full_summary.select(cols)


def compute_temporal_drift(
    reference: pl.DataFrame,
    monitoring: pl.DataFrame,
    date_col: str = "issue_d",
    freq: str = "3mo",
    columns: list[str] | None = None,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> pl.DataFrame:
    """Compute PSI and KS across temporal windows against a reference baseline DataFrame.

    Parameters
    ----------
    reference : pl.DataFrame
        Baseline reference population (e.g., train).
    monitoring : pl.DataFrame
        Target monitoring population (e.g., test, validation, or concatenated).
    date_col : str, default "issue_d"
        Date column name used to partition the data into temporal windows.
    freq : str, default "3mo"
        Window frequency string for Polars truncate (e.g., "3mo" for quarters, "1mo" for months).
    columns : list[str] | None, default None
        Features to compute drift for.
    n_bins : int, default 10
        Number of bins for PSI calculation.
    epsilon : float, default 1e-4
        Epsilon smoothing parameter.

    Returns
    -------
    pl.DataFrame
        Summary table with [window, feature, psi, ks_statistic, ks_pvalue, ref_count, target_count, drift_alert]
    """
    # Ensure date column is Date type
    df = monitoring
    if df[date_col].dtype != pl.Date:
        df = df.with_columns(pl.col(date_col).str.to_date().alias("_date_parsed"))
        date_feature = "_date_parsed"
    else:
        date_feature = date_col

    df = df.with_columns(pl.col(date_feature).dt.truncate(freq).alias("window"))

    unique_windows = df["window"].drop_nulls().unique().sort()

    window_summaries = []
    for w in unique_windows:
        sub_df = df.filter(pl.col("window") == w)
        summary = compute_drift_summary(
            reference=reference,
            target=sub_df,
            columns=columns,
            n_bins=n_bins,
            epsilon=epsilon,
        )
        summary = summary.with_columns(pl.lit(str(w)).alias("window"))
        window_summaries.append(summary)

    if not window_summaries:
        return pl.DataFrame()

    full = pl.concat(window_summaries)
    cols = ["window"] + [c for c in full.columns if c != "window"]
    return full.select(cols)

