from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS
from src.utils.run_logger import RunLogger


@dataclass
class CatBoostResults:
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
    best_iteration: int


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to numeric seconds since epoch (no deprecated .view)."""
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = (df[c].astype("int64") // 10**9).astype("float32")
    return df


def train_catboost(
    random_state: int = 42,
    test_size: float = 0.2,
    threshold: float = 0.50,
    use_gpu: bool = True,
    gpu_device: str = "0",
) -> tuple[CatBoostResults, dict[str, Any], CatBoostClassifier]:

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

    # Identify categorical columns (CatBoost expects column indices)
    cat_cols = [
        i for i, c in enumerate(X.columns)
        if str(X[c].dtype) in ("object", "category", "bool")
    ]
    print(f"Rows: {n_rows:,}")
    print(f"Features: {X.shape[1]} | categorical={len(cat_cols)}")
    if cat_cols:
        print(f"Categorical feature indices: {cat_cols} | names: {[X.columns[i] for i in cat_cols]}")

    n_features = X.shape[1]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    # Imbalance handling
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # ---- SPEED + EARLY STOPPING ----
    # iterations is just an upper bound; od_wait makes it stop early.
    params = dict(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=random_state,
        scale_pos_weight=scale_pos_weight,

        # Early stopping / best model selection
        od_type="Iter",
        od_wait=200,
        use_best_model=True,

        # Logging & metric overhead
        metric_period=50,
        verbose=50,

        # Avoid writing temp files
        allow_writing_files=False,
    )

    # ---- GPU (optional) ----
    # If GPU training fails (driver/cuda issues), set use_gpu=False.
    if use_gpu:
        params.update(
            task_type="GPU",
            devices=gpu_device,
        )
        print(f"Using GPU: task_type=GPU, devices={gpu_device}")
    else:
        params.update(task_type="CPU")
        print("Using CPU (task_type=CPU)")

    model = CatBoostClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_cols,
    )

    # Best iteration chosen by OD (when use_best_model=True)
    best_iter = int(model.get_best_iteration() or 0)

    # Predict probabilities (model is already the best model if use_best_model=True)
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

    results = CatBoostResults(
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
        best_iteration=best_iter,
    )

    artifacts = {
        "model_table": str(model_table_path),
        "categorical_feature_indices": cat_cols,
        "categorical_feature_names": [X.columns[i] for i in cat_cols],
        "scale_pos_weight": float(scale_pos_weight),
        "task_type": "GPU" if use_gpu else "CPU",
        "devices": gpu_device if use_gpu else None,
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

    # Toggle this if GPU errors out on your setup.
    USE_GPU = True
    GPU_DEVICE = "0"

    def train_fn():
        results, extra_artifacts, model = train_catboost(
            random_state=random_state,
            test_size=test_size,
            threshold=threshold,
            use_gpu=USE_GPU,
            gpu_device=GPU_DEVICE,
        )

        model_path = artifacts_dir / "catboost_model.cbm"
        model.save_model(model_path)

        metrics_path = reports_dir / "catboost_metrics.json"
        metrics_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")

        print("\n=== CatBoost Results ===")
        print(f"ROC-AUC: {results.roc_auc:.4f}")
        print(f"PR-AUC:  {results.pr_auc:.4f}")
        print(
            f"Precision/Recall/F1: "
            f"{results.precision:.4f} / {results.recall:.4f} / {results.f1:.4f}"
        )
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(results.confusion_matrix)
        print(f"Best iteration: {results.best_iteration}")

        return {
            "roc_auc": results.roc_auc,
            "pr_auc": results.pr_auc,
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1": results.f1,
            "threshold": results.threshold,
            "best_iteration": results.best_iteration,
            "n_rows": results.n_rows,
            "n_features": results.n_features,
            "confusion_matrix": results.confusion_matrix,
            "artifact_model": str(model_path),
            "artifact_metrics": str(metrics_path),
            **{f"extra_{k}": v for k, v in extra_artifacts.items()},
        }

    record = logger.log_run(
        model_name="catboost",
        train_fn=train_fn,
        dataset={"name": "kkbox", "split": f"train/valid (test_size={test_size})"},
        params={
            "random_state": random_state,
            "test_size": test_size,
            "threshold": threshold,
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "od_wait": 200,
            "metric_period": 50,
            "task_type": "GPU" if USE_GPU else "CPU",
            "devices": GPU_DEVICE if USE_GPU else None,
            "imbalance": "scale_pos_weight",
        },
        notes="CatBoost with early stopping (od_wait=200) and GPU acceleration when available.",
    )

    print(f"\n🧾 Logged run: {record['run_id']} ({record['status']})")


if __name__ == "__main__":
    main()
