from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pandas.api.types import is_datetime64_any_dtype


# ---------------------------
# CONFIG
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "kkbox" / "processed" / "model_table.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "champion"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "is_churn"
TIME_COL = "txn_last_date"
ID_COL = "msno"

# Deterministic 80/20 chronological split
SPLIT_QUANTILE = 0.80

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



def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Ensure TIME_COL is datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    # Chronological split
    cutoff = df[TIME_COL].quantile(SPLIT_QUANTILE)
    train_df = df[df[TIME_COL] <= cutoff].copy()
    valid_df = df[df[TIME_COL] > cutoff].copy()

    print(f"Train size: {len(train_df):,}")
    print(f"Valid size: {len(valid_df):,}")

    X_train, feature_cols, cat_cols = prepare_features(train_df)
    y_train = train_df[TARGET_COL].astype(int)

    X_valid = valid_df[feature_cols].copy()
    # Ensure same categorical dtype in valid
    for c in cat_cols:
        X_valid[c] = X_valid[c].astype("category")

    y_valid = valid_df[TARGET_COL].astype(int)

    print(f"Features used: {len(feature_cols)}")
    if cat_cols:
        print(f"Categorical features: {cat_cols}")

    print("Training LightGBM champion model...")
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    print("Evaluating...")
    y_proba = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
        "pr_auc": float(average_precision_score(y_valid, y_proba)),
        "f1_at_0_5": float(f1_score(y_valid, (y_proba >= 0.5).astype(int))),
        "cutoff_quantile": SPLIT_QUANTILE,
        "time_col": TIME_COL,
    }

    print(metrics)

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
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n✅ Champion model saved to {model_path}")


if __name__ == "__main__":
    main()
