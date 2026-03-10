from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import is_datetime64_any_dtype
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------
# CONFIG DEFAULTS
# ---------------------------

TARGET_COL = "is_churn"
TIME_COL = "txn_last_date"
ID_COL = "msno"

CUTOFF_DATE = pd.Timestamp("2017-01-31")
SPLIT_QUANTILE = 0.80
FEATURE_VERSION = "model_table_v1"
TOP_K_VALUES = [5_000, 10_000, 20_000]
DROP_DT_COLS = True

LGBM_PARAMS = {
    "colsample_bytree": 0.784575377162775,
    "learning_rate": 0.03583753342568752,
    "max_bin": 1023,
    "min_child_samples": 28,
    "n_estimators": 146,
    "n_jobs": -1,
    "num_leaves": 1212,
    "reg_alpha": 0.5616512686484578,
    "reg_lambda": 0.0009765625,
    "verbose": -1,
    "random_state": 42,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker-provided paths
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Folder containing training parquet file(s)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Folder where SageMaker expects model artifacts",
    )

    # User-controlled params
    parser.add_argument("--target-col", type=str, default=TARGET_COL)
    parser.add_argument("--time-col", type=str, default=TIME_COL)
    parser.add_argument("--id-col", type=str, default=ID_COL)
    parser.add_argument("--subset-fraction", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def find_parquet_file(folder: str) -> str:
    path = Path(folder)
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {folder}")
    if len(parquet_files) > 1:
        print(f"Found multiple parquet files; using first one: {parquet_files[0]}")
    return str(parquet_files[0])


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    id_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns:
      X: feature matrix with only numeric + category types
      feature_cols: list of feature columns used
      categorical_cols: list of categorical columns
    """
    drop_cols = {target_col, time_col, id_col}
    candidates = [c for c in df.columns if c not in drop_cols]

    if DROP_DT_COLS:
        candidates = [c for c in candidates if not is_datetime64_any_dtype(df[c])]

    X = df[candidates].copy()

    categorical_cols = []
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
            categorical_cols.append(c)

    for c in X.columns:
        if is_datetime64_any_dtype(X[c]):
            X[c] = (X[c].astype("int64") // 10**9).astype("int64")

    return X, list(X.columns), categorical_cols


def precision_recall_at_k(
    y_true: np.ndarray, y_proba: np.ndarray, k: int
) -> tuple[float, float]:
    k = int(min(k, len(y_true)))
    order = np.argsort(y_proba)[::-1]
    top_idx = order[:k]
    tp_at_k = int(y_true[top_idx].sum())
    precision_k = float(tp_at_k / k) if k > 0 else 0.0
    total_churners = int(y_true.sum())
    recall_k = float(tp_at_k / total_churners) if total_churners > 0 else 0.0
    return precision_k, recall_k


def main():
    args = parse_args()

    print("=== SageMaker champion training script starting ===")
    print(f"Train channel path: {args.train}")
    print(f"Model dir: {args.model_dir}")

    os.makedirs(args.model_dir, exist_ok=True)

    # 1) Read parquet from SageMaker input channel
    input_file = find_parquet_file(args.train)
    print(f"Reading data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded dataframe shape: {df.shape}")

    # 2) Optional stratified subset for cost control
    if args.subset_fraction < 1.0:
        df, _ = train_test_split(
            df,
            train_size=args.subset_fraction,
            random_state=args.random_state,
            stratify=df[args.target_col],
        )
        print(f"After subset_fraction={args.subset_fraction}, shape={df.shape}")

    # 3) Ensure time column is datetime
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col])

    # 4) Deterministic chronological split
    train_df = df[df[args.time_col] <= CUTOFF_DATE].copy()
    valid_df = df[df[args.time_col] > CUTOFF_DATE].copy()
    split_method = "fixed_date"
    actual_cutoff = CUTOFF_DATE

    if len(valid_df) == 0:
        print(f"WARNING: No rows found after CUTOFF_DATE {CUTOFF_DATE.date()}.")
        print(f"Max {args.time_col} in data: {df[args.time_col].max()}")
        print(f"Falling back to quantile split (q={SPLIT_QUANTILE})...")
        actual_cutoff = df[args.time_col].quantile(SPLIT_QUANTILE)
        train_df = df[df[args.time_col] <= actual_cutoff].copy()
        valid_df = df[df[args.time_col] > actual_cutoff].copy()
        split_method = f"quantile_{SPLIT_QUANTILE}"
        print(f"Quantile cutoff: {actual_cutoff}")

    print(f"Split method: {split_method}")
    print(f"Cutoff: {actual_cutoff}")
    print(f"Train size: {len(train_df):,}")
    print(f"Valid size: {len(valid_df):,}")
    print(f"Valid churn rate: {valid_df[args.target_col].mean():.4f}")

    # 5) Prepare features
    X_train, feature_cols, cat_cols = prepare_features(
        train_df,
        target_col=args.target_col,
        time_col=args.time_col,
        id_col=args.id_col,
    )
    y_train = train_df[args.target_col].astype(int)

    X_valid = valid_df[feature_cols].copy()
    for c in cat_cols:
        X_valid[c] = X_valid[c].astype("category")
    y_valid = valid_df[args.target_col].astype(int).to_numpy()

    print(f"Features used: {len(feature_cols)}")
    if cat_cols:
        print(f"Categorical features: {cat_cols}")

    # 6) Train champion model
    print("Training LightGBM champion model...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    # 7) Evaluate
    print("Evaluating...")
    y_proba = model.predict_proba(X_valid)[:, 1]

    roc_auc = float(roc_auc_score(y_valid, y_proba))
    pr_auc = float(average_precision_score(y_valid, y_proba))
    f1_05 = float(f1_score(y_valid, (y_proba >= 0.5).astype(int), zero_division=0))

    p_at_k = {}
    r_at_k = {}
    for k in TOP_K_VALUES:
        p, r = precision_recall_at_k(y_valid, y_proba, k)
        p_at_k[k] = p
        r_at_k[k] = r

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_at_0_5": f1_05,
        "time_col": args.time_col,
        "cutoff_policy": split_method,
        "cutoff_date": str(actual_cutoff),
        "valid_rows": int(len(valid_df)),
        "valid_churn_rate": float(y_valid.mean()),
        "rows_used": int(len(df)),
        "feature_version": FEATURE_VERSION,
        "subset_fraction": args.subset_fraction,
        "precision_at_k": {str(k): float(v) for k, v in p_at_k.items()},
        "recall_at_k": {str(k): float(v) for k, v in r_at_k.items()},
    }

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    # 8) Save artifacts to /opt/ml/model
    model_path = Path(args.model_dir) / "model.pkl"
    feature_list_path = Path(args.model_dir) / "feature_list.json"
    categorical_cols_path = Path(args.model_dir) / "categorical_cols.json"
    params_path = Path(args.model_dir) / "flaml_best_params.json"
    metrics_path = Path(args.model_dir) / "metrics.json"
    scored_path = Path(args.model_dir) / "valid_scored.parquet"

    joblib.dump(model, model_path)
    feature_list_path.write_text(json.dumps(feature_cols, indent=2))
    categorical_cols_path.write_text(json.dumps(cat_cols, indent=2))
    params_path.write_text(json.dumps(LGBM_PARAMS, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))

    scored_df = pd.DataFrame(
        {
            args.id_col: valid_df[args.id_col].values
            if args.id_col in valid_df.columns
            else np.arange(len(valid_df)),
            "y_true": y_valid.astype(int),
            "y_proba": y_proba.astype(float),
        }
    )
    scored_df.to_parquet(scored_path, index=False)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved scored validation to: {scored_path}")
    print("=== Training complete ===")


if __name__ == "__main__":
    main()
