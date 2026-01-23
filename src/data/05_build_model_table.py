"""
Step 6: Build Model Table (spine + aggregated features)

Purpose:
- Join:
    spine.parquet (train + members)
    txn_features.parquet (per-user)
    log_features.parquet (per-user)
- Produce a single, model-ready table with 1 row per msno
- Fill missing values sensibly (absence of activity/transactions becomes signal)

How to run (from project root, with venv activated):
    python src/05_build_model_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import PATHS, ensure_dirs


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def _infer_fill_zero_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """
    Choose numeric columns to fill with 0.0 (counts/sums/ratios) while excluding id/label columns.
    We only fill numeric dtypes by default.
    """
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("-" * 80)

    ensure_dirs()

    # Inputs
    _assert_exists(PATHS.SPINE_PARQUET)
    _assert_exists(PATHS.TXN_FEATURES_PARQUET)
    _assert_exists(PATHS.LOG_FEATURES_PARQUET)

    # Output
    out_path = PATHS.PROCESSED_DIR / "model_table.parquet"

    # ------------------------------------------
    # Step 6.1) Load datasets
    # ------------------------------------------
    print(f"Loading spine: {PATHS.SPINE_PARQUET}")
    spine = pd.read_parquet(PATHS.SPINE_PARQUET)

    print(f"Loading txn features: {PATHS.TXN_FEATURES_PARQUET}")
    txn = pd.read_parquet(PATHS.TXN_FEATURES_PARQUET)

    print(f"Loading log features: {PATHS.LOG_FEATURES_PARQUET}")
    logs = pd.read_parquet(PATHS.LOG_FEATURES_PARQUET)

    # Basic checks
    for name, df in [("spine", spine), ("txn", txn), ("logs", logs)]:
        if "msno" not in df.columns:
            raise ValueError(f"{name} is missing 'msno' column")
    if "is_churn" not in spine.columns:
        raise ValueError("spine is missing 'is_churn' label column")

    # Ensure uniqueness in feature tables (should be 1 row per msno)
    if txn["msno"].duplicated().any():
        raise ValueError("txn_features has duplicate msno rows — aggregation step may be wrong.")
    if logs["msno"].duplicated().any():
        raise ValueError("log_features has duplicate msno rows — aggregation step may be wrong.")

    print("-" * 80)

    # ------------------------------------------
    # Step 6.2) Join tables (LEFT joins from spine)
    # ------------------------------------------
    print("Joining spine + txn_features + log_features ...")
    model_df = spine.merge(txn, on="msno", how="left", validate="m:1")
    model_df = model_df.merge(logs, on="msno", how="left", validate="m:1")

    print("Model table shape:", model_df.shape)

    # Verify 1 row per msno (spine may contain dupes; warn)
    dup_msno = model_df["msno"].duplicated().sum()
    if dup_msno > 0:
        print(f"⚠️ Warning: model_table has {dup_msno:,} duplicate msno rows (coming from spine/train).")
        print("   Consider deduplicating train if needed before modeling.")

    print("-" * 80)

    # ------------------------------------------
    # Step 6.3) Missing value handling (sensible defaults)
    # ------------------------------------------
    # Philosophy:
    # - Missing txn/log features often mean "no activity" or "no transactions"
    # - For numeric aggregates, filling with 0 is reasonable and informative
    # - For member categorical fields, keep 'unknown' (already handled in spine step for gender)
    # - For member numeric fields like age, keep NaN for imputer later (or fill if you prefer)

    exclude_from_zero_fill = {"msno", "is_churn"}
    zero_fill_cols = _infer_fill_zero_columns(model_df, exclude=exclude_from_zero_fill)

    # Optional: exclude certain numeric columns you want to impute differently later
    # Example: 'bd' (age) should usually be imputed with median, not 0
    if "bd" in zero_fill_cols:
        zero_fill_cols.remove("bd")

    # Fill numeric aggregates with 0
    model_df[zero_fill_cols] = model_df[zero_fill_cols].fillna(0)

    # Ensure gender has no missing (if present)
    if "gender" in model_df.columns:
        model_df["gender"] = model_df["gender"].fillna("unknown").astype(str)

    print("Missingness after fills (top 15 columns by % missing):")
    missing_pct = (model_df.isnull().mean().sort_values(ascending=False) * 100).head(15)
    print(missing_pct)
    print("-" * 80)

    # ------------------------------------------
    # Step 6.4) Quick sanity prints
    # ------------------------------------------
    print("Label distribution (is_churn):")
    print(model_df["is_churn"].value_counts(normalize=True, dropna=False))

    numeric_cols = [c for c in model_df.columns if pd.api.types.is_numeric_dtype(model_df[c])]
    print(f"\nTotal numeric columns: {len(numeric_cols)}")
    print("Sample numeric columns:", numeric_cols[:15])

    # ------------------------------------------
    # Step 6.5) Save model-ready table
    # ------------------------------------------
    model_df.to_parquet(out_path, index=False)
    print(f"\n✅ Wrote model table: {out_path}")

    print(
        """
NEXT STEPS:
1) Create train/valid split (stratified by is_churn).
2) Build sklearn preprocessing pipeline (impute + one-hot encode).
3) Train baseline model (LogisticRegression) and evaluate ROC-AUC / PR-AUC.
"""
    )


if __name__ == "__main__":
    main()
