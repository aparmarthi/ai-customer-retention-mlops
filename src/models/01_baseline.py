from __future__ import annotations

import sys
from pathlib import Path

# Add project root to import path so `import config` works
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ai-customer-retention-mlops/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import PATHS

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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

@dataclass
class BaselineResults:
    n_rows: int
    n_features: int
    churn_rate_overall: float
    churn_rate_train: float
    churn_rate_valid: float
    majority_class: int
    majority_prob: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]


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

    # Basic sanity
    n_rows = len(df)
    dup_msno = df["msno"].duplicated().sum()
    if dup_msno > 0:
        print(f"⚠️ Warning: {dup_msno:,} duplicate msno rows found (model_table should ideally be 1 row per msno).")

    # Prepare X/y (we don't use msno as a feature)
    y = df["is_churn"].astype(int).values
    X = df.drop(columns=["is_churn", "msno"], errors="ignore")

    # If any non-numeric columns exist, that's fine for later models;
    # for baselines we don't need X at all. We still track n_features.
    n_features = X.shape[1]

    # Split (stratified)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    churn_rate_overall = float(np.mean(y))
    churn_rate_train = float(np.mean(y_train))
    churn_rate_valid = float(np.mean(y_valid))

    # Majority class baseline learned from TRAIN split
    majority_class = int(1 if np.mean(y_train) >= 0.5 else 0)
    majority_prob = float(np.mean(y_train))  # constant probability baseline = churn rate

    # Hard predictions (always majority class)
    y_pred = np.full_like(y_valid, fill_value=majority_class)

    # Probabilistic predictions (constant churn probability)
    # This is useful for ROC-AUC / PR-AUC sanity.
    y_proba = np.full_like(y_valid, fill_value=majority_prob, dtype=float)

    # Metrics
    acc = float(accuracy_score(y_valid, y_pred))
    prec = float(precision_score(y_valid, y_pred, zero_division=0))
    rec = float(recall_score(y_valid, y_pred, zero_division=0))
    f1 = float(f1_score(y_valid, y_pred, zero_division=0))

    # ROC-AUC/PR-AUC require probabilities (or scores)
    # If valid split has only one class (rare), roc_auc_score will error; guard for that.
    try:
        roc = float(roc_auc_score(y_valid, y_proba))
    except ValueError:
        roc = float("nan")

    try:
        pr = float(average_precision_score(y_valid, y_proba))
    except ValueError:
        pr = float("nan")

    cm = confusion_matrix(y_valid, y_pred).tolist()

    results = BaselineResults(
        n_rows=n_rows,
        n_features=n_features,
        churn_rate_overall=churn_rate_overall,
        churn_rate_train=churn_rate_train,
        churn_rate_valid=churn_rate_valid,
        majority_class=majority_class,
        majority_prob=majority_prob,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
    )

    print("\n=== Baseline: Majority Class (learned on train) ===")
    print(f"Rows: {results.n_rows:,}")
    print(f"Features (excluding msno/is_churn): {results.n_features:,}")
    print(f"Churn rate (overall/train/valid): {results.churn_rate_overall:.4f} / {results.churn_rate_train:.4f} / {results.churn_rate_valid:.4f}")
    print(f"Majority class: {results.majority_class} (constant prob={results.majority_prob:.4f})")
    print(f"Accuracy:   {results.accuracy:.4f}")
    print(f"Precision:  {results.precision:.4f}")
    print(f"Recall:     {results.recall:.4f}")
    print(f"F1:         {results.f1:.4f}")
    print(f"ROC-AUC:    {results.roc_auc:.4f}")
    print(f"PR-AUC:     {results.pr_auc:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(results.confusion_matrix)

    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "baseline_majority.json"
    out_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
