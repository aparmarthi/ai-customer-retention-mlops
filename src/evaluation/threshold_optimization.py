"""
threshold_optimization.py

Sweep decision thresholds 0.01 >> 0.99 over the champion model's validation
scores and select two serving policies:

  1. ROI-optimal threshold  - maximizes expected net ROI
  2. Ops-friendly top-K     - target the top-K highest-risk customers
                              (budget-constrained, no threshold tuning)

Reads:
  artifacts/champion/valid_scored.parquet   (msno, y_true, y_proba)

Outputs:
  reports/threshold_sweep.csv
  artifacts/champion/threshold.json
  reports/threshold_vs_precision_recall.png
  reports/threshold_vs_roi.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORED_PATH  = PROJECT_ROOT / "artifacts" / "champion" / "valid_scored.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "champion"
REPORT_DIR   = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── business assumptions ───────────────────────────────────────────────────────
# Two scenarios aligned to reports/business_assumptions.md
SCENARIOS = {
    # Retention offer scenario (incentive-based)
    "offer": {
        "cost_per_target": 10.0,   # C
        "save_rate": 0.20,         # S
        "value_per_save": 60.0,    # V
    },
    # Proactive outreach scenario (low-cost) — your headline example
    "outreach": {
        "cost_per_target": 0.50,   # C
        "save_rate": 0.12,         # S
        "value_per_save": 80.0,    # V
    },
}

# Choose which scenario defines "ROI-optimal threshold"
ROI_OPTIMIZE_SCENARIO = "outreach"

# Keep legacy ROI definition for backward compatibility with your current outputs
CHURN_COST     = 120.0
INTERVENE_COST = 5.0
SAVE_RATE      = 0.2

# ── business assumptions ───────────────────────────────────────────────────────
#CHURN_COST     = 120.0   # revenue lost per unretained churner ($)
#INTERVENE_COST = 5.0     # cost of one outreach / offer ($)
#SAVE_RATE      = 0.2     # fraction of targeted churners successfully retained

# Precision@K targets and ops budget
TOP_K_VALUES = [5_000, 10_000, 20_000]
OPS_K        = 10_000   # budget-constrained contact-list size
OPS_TOP_PCT = 0.01  # e.g., target top 1% highest risk

# ── helpers ────────────────────────────────────────────────────────────────────

def expected_roi(tp: float, n_targeted: float) -> float:
    """Net expected ROI = revenue saved − intervention cost."""
    return tp * SAVE_RATE * CHURN_COST - n_targeted * INTERVENE_COST


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Sort customers by score descending; return fraction in top-k that churn."""
    top_idx = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top_idx].mean())


def equiv_threshold_for_k(y_proba: np.ndarray, k: int) -> float:
    """Probability score at rank k — the implicit threshold for a top-k policy."""
    sorted_scores = np.sort(y_proba)[::-1]
    return float(sorted_scores[min(k, len(sorted_scores)) - 1])


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading scored validation set …")
    df = pd.read_parquet(SCORED_PATH)
    y_true  = df["y_true"].to_numpy(dtype=int)
    y_proba = df["y_proba"].to_numpy(dtype=float)

    total_churners = int(y_true.sum())
    print(f"  n={len(df):,}  churners={total_churners:,}  "
          f"churn_rate={y_true.mean():.4f}")

    # ── Precision@K (rank-based, score-ordered) ────────────────────────────────
    prec_at_k = {k: precision_at_k(y_true, y_proba, k) for k in TOP_K_VALUES}
    print("  Precision@K:", {f"{k//1000}k": round(v, 4) for k, v in prec_at_k.items()})

    # ── threshold sweep ────────────────────────────────────────────────────────
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    records = []

    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        tp = int((pred & y_true).sum())
        fp = int((pred & (1 - y_true)).sum())
        fn = int(((1 - pred) & y_true).sum())
        n_targeted = int(pred.sum())

        precision = tp / n_targeted        if n_targeted        else 0.0
        recall    = tp / total_churners   if total_churners   else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        row = dict(
            threshold  = round(float(thr), 2),
            tp=tp, fp=fp, fn=fn,
            precision  = round(precision, 6),
            recall     = round(recall,    6),
            f1         = round(f1,        6),
            roi        = round(expected_roi(tp, n_targeted), 2),
            n_targeted = n_targeted,
        )
        # Precision@K — constant score-ranked metrics; included for convenience
        for k in TOP_K_VALUES:
            row[f"precision_at_{k // 1000}k"] = round(prec_at_k[k], 6)

        records.append(row)

    sweep = pd.DataFrame(records)

    # ── policy selection ───────────────────────────────────────────────────────
    roi_idx = sweep["roi"].idxmax()
    roi_row = sweep.loc[roi_idx]

    # Ops top-K: threshold whose n_targeted is closest to OPS_K
    ops_idx = (sweep["n_targeted"] - OPS_K).abs().idxmin()
    ops_row = sweep.loc[ops_idx]

    ops_thr_exact = equiv_threshold_for_k(y_proba, OPS_K)

    print(f"\n  ROI-optimal  >> threshold={roi_row.threshold}  "
          f"ROI=${roi_row.roi:,.0f}  n_targeted={roi_row.n_targeted:.0f}")
    print(f"  Ops top-{OPS_K // 1000}k >> threshold~{ops_row.threshold}  "
          f"precision@{OPS_K // 1000}k={prec_at_k[OPS_K]:.4f}")

    # ── save sweep CSV ─────────────────────────────────────────────────────────
    csv_path = REPORT_DIR / "threshold_sweep.csv"
    sweep.to_csv(csv_path, index=False)
    print(f"\n  Saved >> {csv_path}")

    # ── save threshold.json ────────────────────────────────────────────────────
    threshold_doc = {
        "roi_optimal": {
            "threshold"   : float(roi_row.threshold),
            "selected_by" : "roi_max",
            "tp"          : int(roi_row.tp),
            "fp"          : int(roi_row.fp),
            "fn"          : int(roi_row.fn),
            "precision"   : round(float(roi_row.precision), 4),
            "recall"      : round(float(roi_row.recall),    4),
            "f1"          : round(float(roi_row.f1),        4),
            "roi"         : float(roi_row.roi),
            "n_targeted"  : int(roi_row.n_targeted),
        },
        "ops_top_k": {
            "policy"           : f"top_{OPS_K // 1000}k",
            "k"                : OPS_K,
            "equiv_threshold"  : round(ops_thr_exact, 4),
            "precision_at_k"   : round(prec_at_k[OPS_K], 4),
            "tp_at_k"          : int(round(prec_at_k[OPS_K] * OPS_K)),
            "recall_at_k"      : round(prec_at_k[OPS_K] * OPS_K / total_churners, 4),
            "roi_at_k"         : round(
                expected_roi(prec_at_k[OPS_K] * OPS_K, OPS_K), 2
            ),
            "description"      : (
                f"Target the top-{OPS_K:,} highest-risk customers by predicted "
                "churn probability. No threshold tuning required — operationally "
                "simple and budget-bounded."
            ),
        },
        "precision_at_k": {
            f"{k // 1000}k": round(v, 4) for k, v in prec_at_k.items()
        },
        "assumptions": {
            "churn_cost"           : CHURN_COST,
            "intervention_cost"    : INTERVENE_COST,
            "save_rate_if_targeted": SAVE_RATE,
            "ops_budget_k"         : OPS_K,
        },
    }

    thr_path = ARTIFACT_DIR / "threshold.json"
    thr_path.write_text(json.dumps(threshold_doc, indent=2))
    print(f"  Saved >> {thr_path}")

    # ── plots ──────────────────────────────────────────────────────────────────
    _plot_precision_recall(sweep, roi_row, ops_row)
    _plot_roi(sweep, roi_row, ops_row)

    print("\n  Done.")


# ── plot helpers ───────────────────────────────────────────────────────────────

def _plot_precision_recall(sweep: pd.DataFrame, roi_row, ops_row) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sweep["threshold"], sweep["precision"],
            label="Precision", color="#2196F3", lw=2)
    ax.plot(sweep["threshold"], sweep["recall"],
            label="Recall",    color="#FF5722", lw=2)
    ax.plot(sweep["threshold"], sweep["f1"],
            label="F1",        color="#4CAF50", lw=1.8, ls="--")

    ax.axvline(roi_row.threshold, color="#9C27B0", ls=":", lw=1.6,
               label=f"ROI-optimal  (t={roi_row.threshold})")
    ax.axvline(ops_row.threshold, color="#FF9800", ls=":", lw=1.6,
               label=f"Ops top-{OPS_K // 1000}k  (t≈{ops_row.threshold})")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold vs Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = REPORT_DIR / "threshold_vs_precision_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved >> {path}")


def _plot_roi(sweep: pd.DataFrame, roi_row, ops_row) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sweep["threshold"], sweep["roi"],
            color="#009688", lw=2, label="Expected ROI ($)")
    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.7)

    ax.axvline(roi_row.threshold, color="#9C27B0", ls=":", lw=1.6,
               label=f"ROI-optimal  (t={roi_row.threshold}, ROI=${roi_row.roi:,.0f})")
    ax.axvline(ops_row.threshold, color="#FF9800", ls=":", lw=1.6,
               label=f"Ops top-{OPS_K // 1000}k  (t≈{ops_row.threshold})")

    # Annotate peak
    ax.annotate(
        f"${roi_row.roi:,.0f}",
        xy=(roi_row.threshold, roi_row.roi),
        xytext=(min(roi_row.threshold + 0.08, 0.88), roi_row.roi * 0.85),
        arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.4),
        fontsize=10, color="#9C27B0", fontweight="bold",
    )

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Estimated ROI ($)", fontsize=12)
    ax.set_title("Threshold vs Expected ROI", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = REPORT_DIR / "threshold_vs_roi.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved >> {path}")


if __name__ == "__main__":
    main()
