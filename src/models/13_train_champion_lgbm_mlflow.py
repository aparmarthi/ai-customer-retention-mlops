from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pandas.api.types import is_datetime64_any_dtype

# MLflow wrapper (you created this in Phase 1.2)
from src.utils.mlflow_utils import (
    set_experiment,
    start_run,
    log_params_flat,
    log_metrics_safe,
    log_artifacts_safe,
)

# ---------------------------
# CONFIG
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "kkbox" / "processed" / "model_table.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "champion"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "is_churn"
TIME_COL = "txn_last_date"
ID_COL = "msno"

# Deterministic time split.
# Primary:  fixed calendar cutoff (preferred when data covers the holdout window).
# Fallback: quantile split, used automatically if the fixed date leaves valid empty.
CUTOFF_DATE    = pd.Timestamp("2017-01-31")
SPLIT_QUANTILE = 0.80   # fallback only

# Feature version string for tracking
FEATURE_VERSION = "model_table_v1"

# Precision@K evaluation
TOP_K_VALUES = [5_000, 10_000, 20_000]

# FLAML optimized params
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

# Columns to drop from features (raw ids, target, and datetimes)
DROP_DT_COLS = True  # keep it simple for capstone


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns:
      X: feature matrix with only numeric + category types
      feature_cols: list of feature columns used
      categorical_cols: list of categorical columns (dtype category)
    """
    drop_cols = {TARGET_COL, TIME_COL, ID_COL}
    candidates = [c for c in df.columns if c not in drop_cols]

    # Optionally drop datetime columns (like *_dt)
    if DROP_DT_COLS:
        candidates = [c for c in candidates if not is_datetime64_any_dtype(df[c])]

    X = df[candidates].copy()

    # Handle object columns (e.g., gender)
    categorical_cols = []
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")
            categorical_cols.append(c)

    # If any datetime columns remain (when DROP_DT_COLS=False), convert to int
    for c in X.columns:
        if is_datetime64_any_dtype(X[c]):
            X[c] = (X[c].astype("int64") // 10**9).astype("int64")

    return X, list(X.columns), categorical_cols


def precision_recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> tuple[float, float]:
    """
    Precision@K: fraction of top-k ranked users that are true churners
    Recall@K: fraction of all churners captured in top-k
    """
    k = int(min(k, len(y_true)))
    order = np.argsort(y_proba)[::-1]
    top_idx = order[:k]
    tp_at_k = int(y_true[top_idx].sum())
    precision_k = float(tp_at_k / k) if k > 0 else 0.0
    total_churners = int(y_true.sum())
    recall_k = float(tp_at_k / total_churners) if total_churners > 0 else 0.0
    return precision_k, recall_k


def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Ensure TIME_COL is datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    # Deterministic chronological split — fixed date, with quantile fallback
    train_df = df[df[TIME_COL] <= CUTOFF_DATE].copy()
    valid_df = df[df[TIME_COL] > CUTOFF_DATE].copy()
    split_method = "fixed_date"
    actual_cutoff = CUTOFF_DATE

    if len(valid_df) == 0:
        print(f"  WARNING: No rows found after CUTOFF_DATE {CUTOFF_DATE.date()}.")
        print(f"  Max {TIME_COL} in data: {df[TIME_COL].max()}")
        print(f"  Falling back to quantile split (q={SPLIT_QUANTILE})...")
        actual_cutoff = df[TIME_COL].quantile(SPLIT_QUANTILE)
        train_df = df[df[TIME_COL] <= actual_cutoff].copy()
        valid_df = df[df[TIME_COL] > actual_cutoff].copy()
        split_method = f"quantile_{SPLIT_QUANTILE}"
        print(f"  Quantile cutoff: {actual_cutoff}")

    print(f"Split method: {split_method}")
    print(f"Cutoff: {actual_cutoff}")
    print(f"Train size: {len(train_df):,}")
    print(f"Valid size: {len(valid_df):,}")
    print(f"Valid churn rate: {valid_df[TARGET_COL].mean():.4f}")

    X_train, feature_cols, cat_cols = prepare_features(train_df)
    y_train = train_df[TARGET_COL].astype(int)

    X_valid = valid_df[feature_cols].copy()
    # Ensure same categorical dtype in valid
    for c in cat_cols:
        X_valid[c] = X_valid[c].astype("category")
    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()

    print(f"Features used: {len(feature_cols)}")
    if cat_cols:
        print(f"Categorical features: {cat_cols}")

    # --- MLflow experiment ---
    set_experiment("kkbox_churn")

    with start_run(run_name="champion_lgbm_time_holdout", tags={"stage": "champion", "model": "lgbm"}):
        # Log params
        log_params_flat(
            {
                "model_name": "LightGBM",
                "time_col": TIME_COL,
                "cutoff_policy": split_method,
                "cutoff_date": str(actual_cutoff),
                "feature_version": FEATURE_VERSION,
                "drop_datetime_cols": DROP_DT_COLS,
                "lgbm_params": LGBM_PARAMS,
                "top_k_values": TOP_K_VALUES,
            }
        )

        print("Training LightGBM champion model...")
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )

        print("Evaluating...")
        y_proba = model.predict_proba(X_valid)[:, 1]

        roc_auc = float(roc_auc_score(y_valid, y_proba))
        pr_auc = float(average_precision_score(y_valid, y_proba))
        f1_05 = float(f1_score(y_valid, (y_proba >= 0.5).astype(int)))

        # Precision@K and Recall@K
        p_at_k: dict[int, float] = {}
        r_at_k: dict[int, float] = {}
        for k in TOP_K_VALUES:
            p, r = precision_recall_at_k(y_valid, y_proba, k)
            p_at_k[k] = p
            r_at_k[k] = r

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_at_0_5": f1_05,
            "time_col": TIME_COL,
            "cutoff_policy": split_method,
            "cutoff_date": str(actual_cutoff),
            "valid_rows": int(len(valid_df)),
            "valid_churn_rate": float(y_valid.mean()),
            "precision_at_k": {str(k): float(v) for k, v in p_at_k.items()},
            "recall_at_k": {str(k): float(v) for k, v in r_at_k.items()},
        }

        print(metrics)

        # Log MLflow metrics (flattened)
        log_metrics_safe(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1_at_0_5": f1_05,
                **{f"precision_at_{k}": v for k, v in p_at_k.items()},
                **{f"recall_at_{k}": v for k, v in r_at_k.items()},
            }
        )

        # ---------------------------
        # Save artifacts (champion bundle)
        # ---------------------------

        # Save model
        model_path = ARTIFACT_DIR / "model.pkl"
        joblib.dump(model, model_path)

        # Save feature list
        (ARTIFACT_DIR / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

        # Save categorical list
        (ARTIFACT_DIR / "categorical_cols.json").write_text(json.dumps(cat_cols, indent=2))

        # Save params
        (ARTIFACT_DIR / "flaml_best_params.json").write_text(json.dumps(LGBM_PARAMS, indent=2))

        # Save metrics
        metrics_path = ARTIFACT_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Save scored validation set (needed by threshold_optimization.py)
        scored_path = ARTIFACT_DIR / "valid_scored.parquet"
        scored_df = pd.DataFrame(
            {
                ID_COL: valid_df[ID_COL].values if ID_COL in valid_df.columns else np.arange(len(valid_df)),
                "y_true": y_valid.astype(int),
                "y_proba": y_proba.astype(float),
            }
        )
        scored_df.to_parquet(scored_path, index=False)

        # Log artifacts to MLflow
        log_artifacts_safe(
            [
                model_path,
                ARTIFACT_DIR / "feature_list.json",
                ARTIFACT_DIR / "categorical_cols.json",
                ARTIFACT_DIR / "flaml_best_params.json",
                metrics_path,
                scored_path,
                # If these exist from other steps, log them too
                REPORT_DIR / "threshold_sweep.csv",
                REPORT_DIR / "threshold_vs_precision_recall.png",
                REPORT_DIR / "threshold_vs_roi.png",
                REPORT_DIR / "shap_summary.png",
                REPORT_DIR / "top_features.csv",
            ],
            artifact_path="artifacts",
        )

        print(f"\n✅ Champion model saved to {model_path}")
        print(f"✅ Scored validation saved to {scored_path}")
        print("\n✅ MLflow run logged (experiment: kkbox_churn)")


if __name__ == "__main__":
    main()