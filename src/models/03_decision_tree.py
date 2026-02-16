from __future__ import annotations

"""
Step 3 (Diagnostic): Decision Tree baseline

Purpose (high-value, low-effort):
- Quick non-linear model to sanity-check feature signal + spot leakage
- Get simple feature importance + confusion matrix
- NOT expected to beat XGBoost/LightGBM

Notes:
- Uses the same split (random_state=42, stratify=y) as previous scripts
- Handles datetime64 columns by converting to numeric seconds since epoch
- One-hot encodes categorical columns (e.g., gender)
- Keeps the tree shallow to prevent overfitting and keep interpretability

Run:
    python src/models/03_decision_tree.py
"""

import sys
from pathlib import Path

# Add project root to import path (requested)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier

from src.data.config import PATHS


@dataclass
class TreeResults:
    model: str
    run_ts: str
    n_rows: int
    n_features_raw: int
    n_numeric: int
    n_categorical: int
    n_datetime_converted: int
    datetime_cols: list[str]
    churn_rate_overall: float
    churn_rate_train: float
    churn_rate_valid: float
    params: dict
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    top_features: list[dict]


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

    # Datetime -> numeric seconds (for sklearn)
    X, dt_cols = _convert_datetime_to_int_seconds(X)

    # Categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Rows: {len(df):,}")
    print(f"Raw features: {X.shape[1]:,} | numeric={len(num_cols):,} categorical={len(cat_cols):,}")
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

    # Preprocess (simple; trees don't need scaling)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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

    # Diagnostic tree settings: shallow + regularized
    tree_params = dict(
        max_depth=5,
        min_samples_leaf=500,
        min_samples_split=2000,
        random_state=42,
        class_weight="balanced",
    )

    clf = DecisionTreeClassifier(**tree_params)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    print("Training Decision Tree (diagnostic)...")
    model.fit(X_train, y_train)

    # Predict proba + preds
    y_proba = model.predict_proba(X_valid)[:, 1]
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

    # Feature names + importances (top 20)
    fitted_pre = model.named_steps["preprocess"]
    fitted_tree = model.named_steps["clf"]

    feature_names = fitted_pre.get_feature_names_out()
    importances = fitted_tree.feature_importances_

    top_k = 20
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_features = [
        {"feature": str(feature_names[i]), "importance": float(importances[i])}
        for i in top_idx
        if importances[i] > 0
    ]

    results = TreeResults(
        model="decision_tree",
        run_ts=datetime.now().isoformat(timespec="seconds"),
        n_rows=len(df),
        n_features_raw=X.shape[1],
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        n_datetime_converted=len(dt_cols),
        datetime_cols=dt_cols,
        churn_rate_overall=churn_rate_overall,
        churn_rate_train=churn_rate_train,
        churn_rate_valid=churn_rate_valid,
        params=tree_params,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
        top_features=top_features,
    )

    print("\n=== Decision Tree (Diagnostic) ===")
    print(f"Churn rate (overall/train/valid): {results.churn_rate_overall:.4f} / {results.churn_rate_train:.4f} / {results.churn_rate_valid:.4f}")
    print("Params:", results.params)
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:   {results.accuracy:.4f}")
    print(f"Precision:  {results.precision:.4f}")
    print(f"Recall:     {results.recall:.4f}")
    print(f"F1:         {results.f1:.4f}")
    print(f"ROC-AUC:    {results.roc_auc:.4f}")
    print(f"PR-AUC:     {results.pr_auc:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(results.confusion_matrix)

    print("\nTop features (by importance):")
    for row in results.top_features[:15]:
        print(f"  {row['importance']:.6f}  {row['feature']}")

    # Save results
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "decision_tree_diagnostic.json"
    out_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
