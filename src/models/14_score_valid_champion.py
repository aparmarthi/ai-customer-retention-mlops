from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "kkbox" / "processed" / "model_table.parquet"
CHAMP_DIR = PROJECT_ROOT / "artifacts" / "champion"
OUT_PATH = CHAMP_DIR / "valid_scored.parquet"

TARGET_COL = "is_churn"
TIME_COL = "txn_last_date"
ID_COL = "msno"
SPLIT_QUANTILE = 0.80

def main():
    df = pd.read_parquet(DATA_PATH)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    cutoff = df[TIME_COL].quantile(SPLIT_QUANTILE)
    valid_df = df[df[TIME_COL] > cutoff].copy()

    feature_cols = json.loads((CHAMP_DIR / "feature_list.json").read_text())
    cat_cols = json.loads((CHAMP_DIR / "categorical_cols.json").read_text())

    X_valid = valid_df[feature_cols].copy()
    for c in cat_cols:
        X_valid[c] = X_valid[c].astype("category")

    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()

    model = joblib.load(CHAMP_DIR / "model.pkl")
    proba = model.predict_proba(X_valid)[:, 1]

    scored = pd.DataFrame({
        ID_COL: valid_df[ID_COL].values if ID_COL in valid_df.columns else np.arange(len(valid_df)),
        TIME_COL: valid_df[TIME_COL].values,
        "y_true": y_valid,
        "y_proba": proba
    })

    scored.to_parquet(OUT_PATH, index=False)
    print(f"✅ Saved scored validation set to: {OUT_PATH}")
    print(f"Rows: {len(scored):,} | Churn rate: {scored['y_true'].mean():.4f}")

if __name__ == "__main__":
    main()
