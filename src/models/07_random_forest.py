from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import OneHotEncoder

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS
from src.utils.run_logger import RunLogger


@dataclass
class RFResults:
    n_rows: int
    n_features_raw: int
    n_features_after_encoding: int
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


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to seconds since epoch (avoid deprecated .view)."""
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = (df[c].astype("int64") // 10**9).astype("float32")
    return df


def train_random_forest(
    random_state: int = 42,
    test_size: float = 0.2,
    threshold: float = 0.50,
) -> tuple[RFResults, dict[str, Any], Pipeline]:

    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    if not model_table_path.exists():
        raise FileNotFoundError(f"Missing model table: {model_table_path}")

    print(f"Loading: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    required_cols = {"msno", "is_churn"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"model_table is missing required columns: {missing}")

    n_rows = len(df)

    y = df["is_churn"].astype(int).values
    X = df.drop(columns=["is_churn", "msno"], errors="ignore").copy()

    X = _convert_datetime_columns(X)

    # Identify feature types
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "object" or str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Rows: {n_rows:,}")
    print(f"Raw features: {X.shape[1]} | numeric={len(num_cols)} categorical={len(cat_cols)}")
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    # Preprocess: one-hot encode categorical columns, passthrough numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    # RandomForest: use class_weight to address imbalance
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2000,
        min_samples_leaf=500,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
        bootstrap=True,
    )

    model = Pipeline(steps=[("prep", preprocessor), ("rf", rf)])

    print("Training RandomForest...")
    model.fit(X_train, y_train)

    # Estimate how many features after encoding (helpful for reporting)
    ohe = model.named_steps["prep"].named_transformers_.get("cat")
    n_ohe = int(sum(len(cats) for cats in ohe.categories_)) if cat_cols else 0
    n_features_after = len(num_cols) + n_ohe

    # Probabilities & predictions
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    acc = float(accuracy_score(y_valid, y_pred))
    prec = float(precision_score(y_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_valid, y_pred, zero_division=0))
    f1 = float(f1_score(y_valid, y_pred, zero_division=0))

    try:
        roc = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        roc = float("nan")

    try:
        pr = float(average_precision_score(y_valid, y_proba))
    except ValueError:
        pr = float("nan")

    cm = confusion_matrix(y_valid, y_pred).tolist()

    results = RFResults(
        n_rows=n_rows,
        n_features_raw=int(X.shape[1]),
        n_features_after_encoding=int(n_features_after),
        churn_rate_overall=churn_rate_overall,
        churn_rate_train=churn_rate_train,
        churn_rate_valid=churn_rate_valid,
        threshold=float(threshold),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
    )

    artifacts: dict[str, Any] = {
        "model_table": str(model_table_path),
        "categorical_cols": cat_cols,
        "numeric_cols_count": len(num_cols),
        "n_features_after_encoding": n_features_after,
    }

    return results, artifacts, model


def main() -> None:
    reports_dir = Path("reports")
    artifacts_dir = Path("artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger("reports/experiment_runs.jsonl")

    random_state = 42
    test_size = 0.2
    threshold = 0.50

    def train_fn():
        results, extra_artifacts, model = train_random_forest(
            random_state=random_state,
            test_size=test_size,
            threshold=threshold,
        )

        # Save model pipeline (includes encoder + RF)
        model_path = artifacts_dir / "random_forest_pipeline.joblib"
        dump(model, model_path)

        # Save metrics
        metrics_path = reports_dir / "random_forest_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        print("\n=== RandomForest Results ===")
        print(f"ROC-AUC: {results.roc_auc:.4f}")
        print(f"PR-AUC:  {results.pr_auc:.4f}")
        print(
            f"Precision/Recall/F1: "
            f"{results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}"
        )
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(results.confusion_matrix)
        print(f"Features raw -> encoded: {results.n_features_raw} -> {results.n_features_after_encoding}")

        return {
            "roc_auc": results.roc_auc,
            "pr_auc": results.pr_auc,
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "threshold": results.threshold,
            "n_rows": results.n_rows,
            "n_features_raw": results.n_features_raw,
            "n_features_after_encoding": results.n_features_after_encoding,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path),
            "artifact_metrics": str(metrics_path),
            **{f"extra_{k}": v for k, v in extra_artifacts.items()},
        }

    record = logger.log_run(
        model_name="random-forest",
        train_fn=train_fn,
        dataset={
            "name": "kkbox",
            "split": f"train/valid (test_size={test_size})",
        },
        params={
            "random_state": random_state,
            "test_size": test_size,
            "threshold": threshold,
            "n_estimators": 600,
            "class_weight": "balanced_subsample",
            "min_samples_split": 2000,
            "min_samples_leaf": 500,
            "one_hot": True,
        },
        notes="RandomForest baseline with one-hot encoding for categorical features; uses class_weight for imbalance.",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
