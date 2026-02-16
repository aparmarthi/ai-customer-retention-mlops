from __future__ import annotations

"""
Step 2: Logistic Regression baseline (with preprocessing)

Updates in this version:
- Adds PROJECT_ROOT to sys.path at top (as requested)
- Fixes pandas deprecation: replaces Series.view(...) with astype("int64")
- Handles datetime64 columns by converting to numeric seconds since epoch
- Adds StandardScaler for numeric features (critical for Logistic Regression)
- Prints a quick probability sanity check so you can see if the model is predicting "everyone churns"

Run:
    python src/models/02_logreg_baseline.py
"""

import sys
from pathlib import Path

# Add project root to import path so `import config` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ai-customer-retention-mlops/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.config import PATHS


@dataclass
class LogRegResults:
    n_rows: int
    n_features_raw: int
    n_numeric: int
    n_categorical: int
    n_datetime_converted: int
    datetime_cols: list[str]
    churn_rate_overall: float
    churn_rate_train: float
    churn_rate_valid: float
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    proba_min: float
    proba_p01: float
    proba_p05: float
    proba_p50: float
    proba_p95: float
    proba_p99: float
    proba_max: float
    proba_mean: float


def _convert_datetime_to_int_seconds(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert datetime64 columns to float64 seconds since epoch. NaT becomes NaN.
    This avoids sklearn dtype promotion issues and keeps missingness.
    """
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        return df, []

    df = df.copy()
    for c in dt_cols:
        s = df[c]
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        mask = s.notna()
        # datetime64[ns] -> int64 nanoseconds -> seconds
        out.loc[mask] = (s.loc[mask].astype("int64") // 1_000_000_000).astype("float64")
        df[c] = out
    return df, dt_cols


def _proba_summary(y_proba: np.ndarray) -> dict[str, float]:
    qs = np.quantile(y_proba, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "proba_min": float(np.min(y_proba)),
        "proba_p01": float(qs[0]),
        "proba_p05": float(qs[1]),
        "proba_p50": float(qs[2]),
        "proba_p95": float(qs[3]),
        "proba_p99": float(qs[4]),
        "proba_max": float(np.max(y_proba)),
        "proba_mean": float(np.mean(y_proba)),
    }


def main() -> None:
    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    if not model_table_path.exists():
        raise FileNotFoundError(f"Missing model table: {model_table_path}")

    print(f"Loading: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    required_cols = {"msno", "is_churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"model_table is missing required columns: {missing}")

    y = df["is_churn"].astype(int).values
    X = df.drop(columns=["is_churn", "msno"], errors="ignore")

    # Convert datetime columns -> numeric seconds
    X, dt_cols = _convert_datetime_to_int_seconds(X)

    n_rows = len(df)
    n_features_raw = X.shape[1]

    # Categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Rows: {n_rows:,}")
    print(f"Raw features: {n_features_raw:,}  | numeric={len(num_cols):,}  categorical={len(cat_cols):,}")
    if dt_cols:
        print(f"Converted datetime -> numeric seconds: {len(dt_cols)} columns")
        print("Datetime columns:", dt_cols)

    # Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    # Preprocess:
    # - median impute + missing indicators
    # - Standardize numeric (important for LR; use with_mean=False for sparse safety)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    print("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_valid)[:, 1]

    # Baseline threshold (we’ll tune later)
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    acc = float(accuracy_score(y_valid, y_pred))
    prec = float(precision_score(y_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_valid, y_pred, zero_division=0))
    f1 = float(f1_score(y_valid, y_pred, zero_division=0))
    roc = float(roc_auc_score(y_valid, y_proba))
    pr = float(average_precision_score(y_valid, y_proba))
    cm = confusion_matrix(y_valid, y_pred).tolist()

    ps = _proba_summary(y_proba)

    results = LogRegResults(
        n_rows=n_rows,
        n_features_raw=n_features_raw,
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        n_datetime_converted=len(dt_cols),
        datetime_cols=dt_cols,
        churn_rate_overall=churn_rate_overall,
        churn_rate_train=churn_rate_train,
        churn_rate_valid=churn_rate_valid,
        threshold=threshold,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
        **ps,
    )

    print("\n=== Logistic Regression Baseline (scaled numerics) ===")
    print(f"Churn rate (overall/train/valid): {results.churn_rate_overall:.4f} / {results.churn_rate_train:.4f} / {results.churn_rate_valid:.4f}")
    print(f"Threshold: {results.threshold:.2f}")
    print(f"Accuracy:   {results.accuracy:.4f}")
    print(f"Precision:  {results.precision:.4f}")
    print(f"Recall:     {results.recall:.4f}")
    print(f"F1:         {results.f1:.4f}")
    print(f"ROC-AUC:    {results.roc_auc:.4f}")
    print(f"PR-AUC:     {results.pr_auc:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(results.confusion_matrix)

    print("\nPredicted probability summary (for churn=1):")
    print(
        f"min={results.proba_min:.4f}  p01={results.proba_p01:.4f}  p05={results.proba_p05:.4f}  "
        f"p50={results.proba_p50:.4f}  p95={results.proba_p95:.4f}  p99={results.proba_p99:.4f}  "
        f"max={results.proba_max:.4f}  mean={results.proba_mean:.4f}"
    )

    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "logreg_baseline.json"
    out_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
