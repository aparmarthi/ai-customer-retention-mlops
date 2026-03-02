"""
shap_analysis.py

Goal:
  Explain the LightGBM champion model using SHAP (SHapley Additive exPlanations).

What it does:
  1) Loads the champion model from artifacts/champion/model.pkl
  2) Loads the model table (data/kkbox/processed/model_table.parquet)
  3) Recreates the champion time-based split using metrics.json (time_col + cutoff_quantile)
  4) Builds the validation feature matrix using artifacts/champion/feature_list.json
  5) Computes SHAP values on a sampled subset of validation rows (for speed)
  6) Writes:
       - reports/shap_summary.png (global feature impact plot)
       - reports/top_features.csv (ranked feature importance table)

Run:
  python -m src.evaluation.shap_analysis
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pandas.api.types import is_datetime64_any_dtype

# SHAP can be heavy; import after matplotlib backend is set
import shap


# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "kkbox" / "processed" / "model_table.parquet"

ART_DIR = PROJECT_ROOT / "artifacts" / "champion"
MODEL_PATH = ART_DIR / "model.pkl"
FEATURE_LIST_PATH = ART_DIR / "feature_list.json"
METRICS_PATH = ART_DIR / "metrics.json"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = REPORT_DIR / "shap_summary.png"
OUT_CSV = REPORT_DIR / "top_features.csv"


# ── config ─────────────────────────────────────────────────────────────────────
TARGET_COL = "is_churn"
ID_COL = "msno"

# SHAP compute can be slow on large validation sets; sampling is standard.
MAX_SHAP_ROWS = 20_000
RANDOM_SEED = 42


# ── feature prep (match your champion training logic) ──────────────────────────
def prepare_features_for_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Create X matrix restricted to feature_cols and ensure categorical columns
    are handled similarly to training (object -> category).

    Returns:
      X: pd.DataFrame with feature_cols
      categorical_cols: list of categorical cols (dtype category)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in dataframe: {missing[:10]} (and {len(missing)-10} more)")

    X = df[feature_cols].copy()

    categorical_cols: list[str] = []
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
            categorical_cols.append(c)

    # If any datetime columns sneak in, convert to integer epoch seconds
    for c in X.columns:
        if is_datetime64_any_dtype(X[c]):
            X[c] = (X[c].astype("int64") // 10**9).astype("int64")

    return X, categorical_cols


def _safe_datetime(col: pd.Series) -> pd.Series:
    """Best-effort to parse numeric date formats like 20170228.0 into datetime."""
    # Try direct datetime parse (handles already-datetime and YYYY-MM-DD strings)
    dt = pd.to_datetime(col, errors="coerce")

    # If many nulls and values look like YYYYMMDD floats/ints, try that format
    if dt.isna().mean() > 0.5:
        # Convert floats like 20170228.0 -> "20170228"
        as_str = pd.Series(col).astype("Int64", errors="ignore").astype(str)
        dt2 = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
        # Use dt2 where it succeeds
        dt = dt.fillna(dt2)

    return dt


def main() -> None:
    # ── load artifacts ─────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Champion model not found: {MODEL_PATH}")
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"feature_list.json not found: {FEATURE_LIST_PATH}")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found: {METRICS_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"model_table.parquet not found: {DATA_PATH}")

    print("Loading champion model …")
    model = joblib.load(MODEL_PATH)

    feature_cols = json.loads(FEATURE_LIST_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())

    time_col = metrics.get("time_col", "txn_last_date")
    split_q = float(metrics.get("cutoff_quantile", 0.80))

    print(f"Using time_col='{time_col}', cutoff_quantile={split_q}")

    # ── load data ──────────────────────────────────────────────────────────────
    print("Loading model table …")
    df = pd.read_parquet(DATA_PATH)

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in model table columns.")

    # Parse time_col
    df[time_col] = _safe_datetime(df[time_col])
    df = df.dropna(subset=[time_col])

    # Recreate time-based split
    cutoff = df[time_col].quantile(split_q)
    valid_df = df[df[time_col] > cutoff].copy()

    print(f"Validation rows (time holdout): {len(valid_df):,}")
    if len(valid_df) == 0:
        raise RuntimeError("Validation set is empty after time split. Check time_col parsing and cutoff.")

    # Build X for validation using the champion feature list
    X_valid, cat_cols = prepare_features_for_model(valid_df, feature_cols)

    # Sample for SHAP
    n = len(X_valid)
    sample_n = min(MAX_SHAP_ROWS, n)

    print(f"Sampling {sample_n:,} rows for SHAP (out of {n:,}) …")
    X_sample = X_valid.sample(n=sample_n, random_state=RANDOM_SEED)

    # ── compute SHAP ───────────────────────────────────────────────────────────
    print("Computing SHAP values (TreeExplainer) …")
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary classification sometimes returns a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    # ── global importance table ────────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # ── shap summary plot ──────────────────────────────────────────────────────
    print("Creating SHAP summary plot …")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()
    print(f"Saved: {OUT_PNG}")

    print("\n✅ SHAP analysis complete.")
    print("Next: open reports/shap_summary.png and reports/top_features.csv")


if __name__ == "__main__":
    main()