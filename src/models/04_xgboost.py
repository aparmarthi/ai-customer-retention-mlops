from __future__ import annotations

"""
Step 4: XGBoost (train API) with early stopping - version-proof

Why this version:
- Some XGBoost sklearn wrappers don't support early_stopping_rounds depending on version/build.
- xgboost.train() + DMatrix supports early stopping broadly and is stable.

Run:
    python src/models/04_xgboost.py

Optional sanity check (recommended once):
    python -c "import xgboost; print(xgboost.__version__); print(xgboost.__file__)"
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

from src.data.config import PATHS


def _convert_datetime_to_int_seconds(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dt_cols:
        return df, []
    df = df.copy()
    for c in dt_cols:
        s = df[c]
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        mask = s.notna()
        out.loc[mask] = (s.loc[mask].astype("int64") // 1_000_000_000).astype("float64")
        df[c] = out
    return df, dt_cols


def main() -> None:
    print("xgboost version:", xgb.__version__)
    print("xgboost path:", xgb.__file__)
    print("-" * 80)

    model_table_path = PATHS.PROCESSED_DIR / "model_table.parquet"
    if not model_table_path.exists():
        raise FileNotFoundError(f"Missing model table: {model_table_path}")

    print(f"Loading: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    if "is_churn" not in df.columns:
        raise ValueError("model_table is missing 'is_churn'")
    y = df["is_churn"].astype(int).values
    X = df.drop(columns=["is_churn", "msno"], errors="ignore")

    # Datetime -> numeric
    X, dt_cols = _convert_datetime_to_int_seconds(X)

    # Column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Rows: {len(df):,}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")
    if dt_cols:
        print("Datetime columns converted:", dt_cols)

    # Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing (one-hot for categorical; numeric median impute)
    # NOTE: OneHotEncoder sparse_output=False to produce dense array; XGBoost DMatrix accepts numpy.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    print("Preprocessing features...")
    X_train_p = preprocessor.fit_transform(X_train)
    X_valid_p = preprocessor.transform(X_valid)

    # Handle class imbalance
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    dtrain = xgb.DMatrix(X_train_p, label=y_train)
    dvalid = xgb.DMatrix(X_valid_p, label=y_valid)

    # XGBoost params (tune later)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",           # optimize PR-AUC for imbalanced churn
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": 42,
    }

    num_boost_round = 2000
    early_stopping_rounds = 50

    print("Training XGBoost with early stopping (train API)...")
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    # Predict probabilities using best iteration
    # Different versions expose best_iteration / best_ntree_limit differently; handle robustly.
    best_iter = getattr(bst, "best_iteration", None)
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)

    if best_iter is not None:
        # Many versions support iteration_range
        try:
            y_proba = bst.predict(dvalid, iteration_range=(0, best_iter + 1))
        except TypeError:
            # Fallback for older versions
            y_proba = bst.predict(dvalid, ntree_limit=best_ntree_limit or 0)
    else:
        y_proba = bst.predict(dvalid)

    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "model": "xgboost_train_api",
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "n_features_after_preprocess": int(X_train_p.shape[1]),
        "threshold": float(threshold),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "precision": float(precision_score(y_valid, y_pred, zero_division=0)),
        "recall": float(recall_score(y_valid, y_pred, zero_division=0)),
        "f1": float(f1_score(y_valid, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "pr_auc": float(average_precision_score(y_valid, y_proba)),
        "confusion_matrix": confusion_matrix(y_valid, y_pred).tolist(),
        "params": params,
    }

    print("\n=== XGBoost Results (train API) ===")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC:  {results['pr_auc']:.4f}")
    print(f"Precision/Recall/F1: {results['precision']:.4f} / {results['recall']:.4f} / {results['f1']:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(results["confusion_matrix"])
    if results["best_iteration"] is not None:
        print("Best iteration:", results["best_iteration"])

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "xgboost_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
