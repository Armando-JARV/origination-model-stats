# AGENTS.md — Agent & Developer Guide

## 1. Project Overview & Context

This repository contains the codebase and research for the **Credit Model Drift Detection** final project.
The goal of the project is to evaluate whether conventional drift monitoring metrics (Population Stability Index / PSI, two-sample Kolmogorov-Smirnov / KS test) reliably anticipate credit model performance degradation (AUC drops), using statistical inference (bootstrap, jackknife, MLE logistic regression, and Bayesian Beta-Binomial models).

- **Data Source:** Lending Club loan dataset (Kaggle), cleaned and partitioned chronologically.
- **Key Question:** Do statistical drift metrics reliably signal model degradation, and how sound are fixed thresholds (e.g. $PSI > 0.25$)?

---

## 2. Technology Stack & Environment

Agents and contributors working in this repository must adhere to the following stack:

* **Python:** `3.12+`
* **Package & Environment Manager:** `uv`
* **Data Processing:** `polars` (>= 2.2.0). **Do NOT use pandas.** Always use native Polars expressions and data structures (`pl.DataFrame`, `pl.Series`).
* **Statistics & Numerics:** `scipy` (>= 1.18.1), `numpy` (>= 2.5)
* **Testing:** `pytest` (>= 9.2.0)

### Essential Commands
```bash
# Sync / install environment
uv sync

# Run tests
uv run pytest

# Execute a Python script
uv run python path/to/script.py
```

---

## 3. Data & Partition Strategy

The cleaned dataset spans loans from `2008-06-01` to `2013-06-01` (`tag = 2008_2013`) and is split chronologically by `issue_d`:

| Dataset Path | Time Window | Observations | Role in Project |
|---|---|---|---|
| `src/data/credito_2008_2013_train.csv` | 2008-06-01 – 2012-12-01 | 92,095 (~70%) | **Baseline / Reference Population (Expected)** |
| `src/data/credito_2008_2013_test.csv` | 2013-01-01 – 2013-04-01 | 32,125 (~24%) | **Monitoring Window 1 (Actual)** |
| `src/data/credito_2008_2013_validation.csv` | 2013-05-01 | 10,350 (~8%) | **Monitoring Window 2 (Actual)** |

### Direct Dataset Focus Rule
* In credit risk drift detection, **the Reference population is `train`**.
* All standard drift comparisons must be performed directly against `train`:
  1. `train` (reference) vs `test` (actual)
  2. `train` (reference) vs `validation` (actual)
* **Never evaluate only `test` vs `validation` for model drift.** Consecutive months in 2013 look stable relative to each other ($PSI < 0.04$), but both have drifted significantly from the 2008–2012 training distribution (e.g. `int_rate` $PSI = 0.265$ against `train`).

---

## 4. Repository Structure

```text
├── AGENTS.md                          # This agent guide
├── README.md                          # Project proposal, roadmap and deliverables
├── pyproject.toml                     # uv project configuration and dependencies
├── uv.lock                            # Dependency lockfile
├── notebooks/
│   └── research/
│       └── Curado_Data.ipynb          # Data cleaning, target creation and temporal split
├── src/
│   ├── data/                          # Cleaned CSV files (train, test, validation)
│   ├── drift/                         # Drift detection package
│   │   ├── __init__.py                # Package exports
│   │   └── metrics_computation.py     # PSI, KS, multi-feature summary, comparison helpers
│   ├── training/                      # Model training routines (MLE logistic regression)
│   └── testing/                       # Model evaluation routines
└── tests/
    └── test_drift.py                  # Unit and synthetic data tests for drift metrics
```

---

## 5. Drift Module Specification (`src/drift/metrics_computation.py`)

All drift functions are designed to accept `polars.Series` or `polars.DataFrame`:

* `compute_psi(reference, target, n_bins=10, epsilon=1e-4, strategy="quantile") -> float`:
  - Quantile-based bin edges computed on `reference`.
  - Edge deduplication for repeated values (e.g., zero-heavy features).
  - Outermost bin edges set to $(-\infty, +\infty)$ to capture out-of-range monitoring values.
  - Epsilon smoothing (`1e-4`) applied to 0-frequency bins before re-normalizing.
* `compute_psi_table(reference, target, n_bins=10, ...) -> pl.DataFrame`:
  - Detailed breakdown with columns: `bin`, `ref_count`, `target_count`, `ref_pct`, `target_pct`, `psi_contribution`.
* `compute_ks(reference, target) -> tuple[float, float]`:
  - Returns `(ks_statistic, p_value)` using two-sample Kolmogorov-Smirnov test.
* `compare_datasets_drift(reference, targets={"test": df_test, "validation": df_val}, columns=[...]) -> pl.DataFrame`:
  - High-level function returning a Polars DataFrame comparing `train` directly to `test` and `validation`.
* `compute_temporal_drift(reference, monitoring, date_col="issue_d", freq="3mo", ...) -> pl.DataFrame`:
  - Partitions monitoring data into temporal chunks (e.g., quarterly) and evaluates drift against `reference`.

---

## 6. Coding & Contribution Rules for Agents

1. **Polars Idioms:** Always prefer Polars native expressions (`df.select(...)`, `df.filter(...)`, `df.with_columns(...)`). Do not convert large DataFrames to Pandas or Python lists.
2. **Type Safety:** Include type annotations for all new functions and public APIs.
3. **No Regressions:** Always verify that `uv run pytest` passes after making code edits.
4. **Resampling Tasks (Upcoming Roadmap):**
   - For Bootstrap (Task 5) and Jackknife (Task 6), leverage the existing functions in `src/drift/metrics_computation.py`.
   - Maintain reproducible random seeds in tests and experiments.
