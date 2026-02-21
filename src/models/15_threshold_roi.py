from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAMP_DIR = PROJECT_ROOT / "artifacts" / "champion"
SCORED_PATH = CHAMP_DIR / "valid_scored.parquet"

OUT_CSV = PROJECT_ROOT / "reports" / "threshold_sweep.csv"
OUT_JSON = CHAMP_DIR / "threshold.json"

PROJECT_ROOT.joinpath("reports").mkdir(exist_ok=True)

# --- Business assumptions (editable) ---
# You can tune these later; keep it simple & transparent.
ASSUMPTIONS = {
    "base_churn_cost": 120.0,         # $ value lost if a churner leaves (proxy for LTV/gross margin)
    "intervention_cost": 5.0,         # $ cost to contact + offer
    "save_rate_if_targeted": 0.20,    # probability we prevent churn if we target a true churner
    "target_budget_k": 10000          # operational capacity per cycle
}

def precision_at_k(y_true, y_proba, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_proba)[:k]
    return float(y_true[idx].mean())

def recall_at_k(y_true, y_proba, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(-y_proba)[:k]
    tp = y_true[idx].sum()
    total_pos = y_true.sum()
    return float(tp / total_pos) if total_pos > 0 else 0.0

def roi_for_targeting(tp: int, fp: int, assumptions: dict) -> float:
    # Expected value gained from true positives targeted:
    # each targeted true churner has save_rate chance of being retained -> avoids base_churn_cost loss
    gain = tp * assumptions["save_rate_if_targeted"] * assumptions["base_churn_cost"]
    spend = (tp + fp) * assumptions["intervention_cost"]
    return float(gain - spend)

def main():
    df = pd.read_parquet(SCORED_PATH)
    y_true = df["y_true"].to_numpy().astype(int)
    y_proba = df["y_proba"].to_numpy().astype(float)

    # --- Top-K policy metrics ---
    k = int(ASSUMPTIONS["target_budget_k"])
    p_at_k = precision_at_k(y_true, y_proba, k)
    r_at_k = recall_at_k(y_true, y_proba, k)

    # --- Threshold sweep ---
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        roi = roi_for_targeting(tp, fp, ASSUMPTIONS)

        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roi": float(roi),
            "n_targeted": int(tp + fp)
        })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_CSV, index=False)

    # Best thresholds for different objectives
    best_roi = sweep.loc[sweep["roi"].idxmax()].to_dict()
    best_f1 = sweep.loc[sweep["f1"].idxmax()].to_dict()

    # Recommend policy (AI PM friendly): prefer top-K if you have fixed ops capacity
    policy = {
        "recommended_policy": "top_k",
        "top_k": k,
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "assumptions": ASSUMPTIONS,
        "best_threshold_by_roi": best_roi,
        "best_threshold_by_f1": best_f1
    }

    (CHAMP_DIR / "roi_policy.json").write_text(json.dumps(policy, indent=2))
    (CHAMP_DIR / "threshold.json").write_text(json.dumps({
        "default_threshold": float(best_roi["threshold"]),
        "selected_by": "roi_max",
        "assumptions": ASSUMPTIONS
    }, indent=2))

    print(f"✅ Wrote sweep to: {OUT_CSV}")
    print("✅ Wrote policy to: artifacts/champion/roi_policy.json")
    print("✅ Wrote threshold to: artifacts/champion/threshold.json")
    print(f"\nTop-K={k}: Precision@K={p_at_k:.4f} | Recall@K={r_at_k:.4f}")
    print(f"Best ROI threshold: {best_roi['threshold']:.2f} | ROI={best_roi['roi']:.2f} | targeted={best_roi['n_targeted']}")
    print(f"Best F1 threshold:  {best_f1['threshold']:.2f} | F1={best_f1['f1']:.4f} | targeted={best_f1['n_targeted']}")

if __name__ == "__main__":
    main()
