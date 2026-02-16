from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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

# Add project root to import path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ai-customer-retention-mlops/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS
from src.utils.run_logger import RunLogger  # you created this earlier


@dataclass
class LGBMResults:
    n_rows: int
    n_features: int
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
    best_iteration: int | None


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns to numeric seconds since epoch.
    Your pipeline already does something similar; this keeps it consistent.
    """
    dt_cols = [c for c in df.columns if c.endswith("_dt") or "datetime" in c.lower()]
    converted = []
    for c in dt_cols:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = (df[c].view("int64") // 10**9).astype("float32")  # seconds
            converted.append(c)
    if converted:
        print(f"Datetime columns converted to seconds: {converted}")
    return df


def _infer_categorical_columns(X: pd.DataFrame) -> list[str]:
    """
    Detect categorical columns (object/category/bool).
    LightGBM can handle categorical features if they are pandas 'category'.
    """
    cat_cols = []
    for c in X.columns:
        if str(X[c].dtype) in ("object", "category", "bool"):
            cat_cols.append(c)
    return cat_cols


def train_lightgbm(
    random_state: int = 42,
    test_size: float = 0.2,
    threshold: float = 0.50,
) -> tuple[LGBMResults, dict[str, Any], LGBMClassifier, np.ndarray, np.ndarray]:
    """
    Trains LightGBM and returns:
      - results dataclass
      - extra artifacts dict
      - trained model
      - y_valid probabilities (for ensemble)
      - y_valid labels (for debugging / sanity if needed)
    """
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

    # Identify categorical columns and convert them to pandas category
    cat_cols = _infer_categorical_columns(X)
    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype("category")

    print(f"Rows: {n_rows:,}")
    print(f"Raw features: {X.shape[1]} | categorical={len(cat_cols)}")
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")

    n_features = X.shape[1]

    # IMPORTANT: this split must match your ensemble script's split logic.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    # Handle imbalance: scale_pos_weight is often strong for churn
    # (neg/pos) computed on TRAIN
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=random_state,
        n_jobs=-1,
        # Note: We're not using early stopping callbacks yet.
        # eval_metric is still used for monitoring.
    )

    clf = LGBMClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        callbacks=[],
    )

    best_iter = getattr(clf, "best_iteration_", None)

    # Predict probabilities for VALIDATION (needed by ensemble)
    y_proba_valid = clf.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba_valid >= threshold).astype(int)

    # Metrics
    acc = float(accuracy_score(y_valid, y_pred))
    prec = float(precision_score(y_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_valid, y_pred, zero_division=0))
    f1 = float(f1_score(y_valid, y_pred, zero_division=0))

    try:
        roc = float(roc_auc_score(y_valid, y_proba_valid))
    except ValueError:
        roc = float("nan")

    try:
        pr = float(average_precision_score(y_valid, y_proba_valid))
    except ValueError:
        pr = float("nan")

    cm = confusion_matrix(y_valid, y_pred).tolist()

    results = LGBMResults(
        n_rows=n_rows,
        n_features=n_features,
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
        best_iteration=int(best_iter) if best_iter is not None else None,
    )

    artifacts: dict[str, Any] = {
        "model_table": str(model_table_path),
        "categorical_cols": cat_cols,
        "scale_pos_weight": float(scale_pos_weight),
    }

    return results, artifacts, clf, y_proba_valid, y_valid


def main() -> None:
    # Use PROJECT_ROOT-relative directories so running from anywhere is safe
    reports_dir = (PROJECT_ROOT / "reports").resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = (PROJECT_ROOT / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(str(reports_dir / "experiment_runs.jsonl"))

    random_state = 42
    test_size = 0.2
    threshold = 0.50

    def train_fn():
        results, extra_artifacts, model, y_proba_valid, _y_valid = train_lightgbm(
            random_state=random_state,
            test_size=test_size,
            threshold=threshold,
        )

        # Save model artifact
        model_path = artifacts_dir / "lgbm_model.txt"
        model.booster_.save_model(str(model_path))

        # Save metrics artifact
        metrics_path = reports_dir / "lightgbm_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        # ---------------------------------------------------------------------
        # ✅ FIX: Save VALID probabilities for ensemble soft-vote
        # This is what your ensemble expects:
        #   reports/preds_lightgbm_valid.npy
        # ---------------------------------------------------------------------
        preds_path = reports_dir / "preds_lightgbm_valid.npy"
        np.save(preds_path, y_proba_valid.astype(float))
        print(f"✅ Saved validation probabilities for ensemble: {preds_path}")

        print("\n=== LightGBM Results (sklearn API) ===")
        print(f"ROC-AUC: {results.roc_auc:.4f}")
        print(f"PR-AUC:  {results.pr_auc:.4f}")
        print(f"Precision/Recall/F1: {results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(results.confusion_matrix)
        if results.best_iteration is not None:
            print(f"Best iteration: {results.best_iteration}")

        # Return metrics dict for logging (JSON-serializable)
        return {
            "roc_auc": results.roc_auc,
            "pr_auc": results.pr_auc,
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "threshold": results.threshold,
            "n_rows": results.n_rows,
            "n_features": results.n_features,
            "best_iteration": results.best_iteration,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path),
            "artifact_metrics": str(metrics_path),
            "artifact_valid_probs": str(preds_path),  # nice to have in logs
            **{f"extra_{k}": v for k, v in extra_artifacts.items()},
        }

    record = logger.log_run(
        model_name="lightgbm",
        train_fn=train_fn,
        dataset={
            "name": "kkbox",
            "split": f"train/valid (test_size={test_size})",
        },
        params={
            "random_state": random_state,
            "test_size": test_size,
            "threshold": threshold,
            # Keep these aligned with params above for future comparison
            "n_estimators": 5000,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 200,
            "imbalance": "scale_pos_weight",
        },
        notes=(
            "LightGBM baseline comparable to XGBoost. "
            "Early stopping not enabled yet; uses fixed n_estimators. "
            "Also saves validation probabilities for ensemble: reports/preds_lightgbm_valid.npy"
        ),
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
