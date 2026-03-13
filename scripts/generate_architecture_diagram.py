"""
Generate architecture diagram for the KKBox Churn MLOps project.
Output: docs/architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
FIG_W, FIG_H = 16, 20
BG_COLOR = "#FAFAFA"
FONT_FAMILY = "sans-serif"

# Color palette
C_DATA     = "#E3F2FD"  # light blue
C_MODEL    = "#FFF3E0"  # light orange
C_EVAL     = "#F3E5F5"  # light purple
C_SERVE    = "#E8F5E9"  # light green
C_CLOUD    = "#FFF8E1"  # light yellow
C_MONITOR  = "#FFEBEE"  # light red
C_BORDER   = "#37474F"  # dark gray
C_ARROW    = "#546E7A"  # medium gray
C_TEXT     = "#212121"  # near black
C_SUBTEXT  = "#616161"  # gray


def draw_box(ax, x, y, w, h, title, items, color, title_size=11, item_size=8.5):
    """Draw a rounded box with title and bullet items."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=C_BORDER,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(box)

    # Title
    ax.text(
        x + w / 2, y + h - 0.25,
        title,
        ha="center", va="top",
        fontsize=title_size, fontweight="bold", color=C_TEXT,
        fontfamily=FONT_FAMILY,
        zorder=3,
    )

    # Items
    for i, item in enumerate(items):
        ax.text(
            x + 0.3, y + h - 0.65 - i * 0.35,
            f"• {item}",
            ha="left", va="top",
            fontsize=item_size, color=C_SUBTEXT,
            fontfamily=FONT_FAMILY,
            zorder=3,
        )


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """Draw a downward arrow with optional label."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.3,head_length=0.15",
            color=C_ARROW,
            lw=2,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x + 0.15, mid_y,
            label,
            ha="left", va="center",
            fontsize=8, color=C_ARROW, fontstyle="italic",
            fontfamily=FONT_FAMILY,
            zorder=3,
        )


def draw_side_arrow(ax, x1, y1, x2, y2, label=""):
    """Draw a horizontal arrow with optional label."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.12",
            color=C_ARROW,
            lw=1.5,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.15,
            label,
            ha="center", va="bottom",
            fontsize=7.5, color=C_ARROW, fontstyle="italic",
            fontfamily=FONT_FAMILY,
            zorder=3,
        )


def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis("off")

    # Title
    ax.text(
        8, 19.5,
        "KKBox Churn Prediction — System Architecture",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_TEXT,
        fontfamily=FONT_FAMILY,
    )
    ax.text(
        8, 19.1,
        "End-to-end ML pipeline: raw data → model → decision policy → serving → monitoring",
        ha="center", va="center",
        fontsize=10, color=C_SUBTEXT,
        fontfamily=FONT_FAMILY,
    )

    # ── Phase labels (left column) ─────────────────────────────
    phases = [
        (0.15, 17.9, "PHASE 1"),
        (0.15, 15.6, "PHASE 2"),
        (0.15, 13.1, "PHASE 3"),
        (0.15, 10.3, "PHASE 4"),
        (0.15, 7.3,  "PHASE 5"),
        (0.15, 4.7,  "PHASE 6"),
        (0.15, 2.2,  "PHASE 7"),
    ]
    for px, py, plabel in phases:
        ax.text(
            px, py, plabel,
            ha="left", va="center",
            fontsize=7, fontweight="bold", color="#9E9E9E",
            fontfamily=FONT_FAMILY,
            rotation=90,
        )

    # ── Row 1: Data Pipeline ───────────────────────────────────
    draw_box(ax, 1.5, 17.0, 5.5, 1.8,
             "1  Data Pipeline  (src/data/)",
             [
                 "7 numbered ETL scripts (01→07)",
                 "Raw CSV → Parquet → Feature tables",
                 "DuckDB for large-scale SQL aggregation",
                 "~28 GB raw → ~1M row model table",
             ],
             C_DATA)

    draw_box(ax, 8.5, 17.0, 5.5, 1.8,
             "Time-Based Split",
             [
                 "Train: ≤ Jan 31, 2017",
                 "Valid: Feb 2017 holdout",
                 "Prevents temporal leakage",
                 "Churn rate: ~1.2% in holdout",
             ],
             C_DATA)

    # Arrow: data → split
    draw_side_arrow(ax, 7.0, 17.9, 8.5, 17.9, "model_table.parquet")

    # ── Row 2: Model Training ──────────────────────────────────
    draw_box(ax, 1.5, 14.5, 5.5, 2.0,
             "2  Model Training  (src/models/)",
             [
                 "12 architectures evaluated",
                 "LightGBM, XGBoost, CatBoost, RF",
                 "TabNet, FT-Transformer, NODE",
                 "FLAML AutoML → champion selection",
             ],
             C_MODEL)

    draw_box(ax, 8.5, 14.5, 5.5, 2.0,
             "MLflow Experiment Tracking",
             [
                 "Experiment: kkbox_churn",
                 "Params, metrics, artifacts per run",
                 "Safe logging (nested dict flattening)",
                 "Split audit trail (cutoff_policy)",
             ],
             C_MODEL)

    # Arrow: split → training
    draw_arrow(ax, 5.0, 17.0, 5.0, 16.5)

    # Arrow: training → mlflow
    draw_side_arrow(ax, 7.0, 15.5, 8.5, 15.5, "atomic run logging")

    # ── Row 3: Evaluation ──────────────────────────────────────
    draw_box(ax, 1.5, 12.0, 5.5, 2.0,
             "3  Champion Evaluation",
             [
                 "Champion: FLAML LightGBM",
                 "PR-AUC: 0.5392  |  ROC-AUC: 0.9660",
                 "P@10k: 18%  |  R@10k: 75%",
                 "SHAP explainability (TreeExplainer)",
             ],
             C_EVAL)

    draw_box(ax, 8.5, 12.0, 5.5, 2.0,
             "Threshold Optimization & ROI",
             [
                 "99-step threshold sweep",
                 "ROI-optimal: t=0.68, $17,666 net",
                 "Top-K: K=10,000, 3x lift over random",
                 "Business assumptions documented",
             ],
             C_EVAL)

    # Arrow: training → evaluation
    draw_arrow(ax, 4.25, 14.5, 4.25, 14.0)

    # Arrow: evaluation → ROI
    draw_side_arrow(ax, 7.0, 13.0, 8.5, 13.0, "valid_scored.parquet")

    # ── Row 4: Serving ─────────────────────────────────────────
    draw_box(ax, 1.5, 9.2, 5.5, 2.2,
             "4  Decision Policy Engine",
             [
                 "Primary: top-K (K=10,000)",
                 "Fallback: ROI threshold (t=0.68)",
                 "PolicyDecision objects",
                 "src/serving/policy.py",
             ],
             C_SERVE)

    draw_box(ax, 8.5, 9.2, 5.5, 2.2,
             "FastAPI Inference Service",
             [
                 "GET /health — model status",
                 "POST /predict — single record",
                 "POST /predict_batch — batch + CSV",
                 "Pydantic contracts, feature alignment",
             ],
             C_SERVE)

    # Arrow: evaluation → policy
    draw_arrow(ax, 4.25, 12.0, 4.25, 11.4)

    # Arrow: policy → API
    draw_side_arrow(ax, 7.0, 10.3, 8.5, 10.3, "policy logic")

    # ── Row 5: Cloud ───────────────────────────────────────────
    draw_box(ax, 1.5, 6.2, 5.5, 2.2,
             "5  AWS SageMaker Validation",
             [
                 "Single controlled training job",
                 "Script Mode (ml.m5.large)",
                 "Same champion hyperparameters",
                 "Artifact → model.tar.gz → S3",
             ],
             C_CLOUD)

    draw_box(ax, 8.5, 6.2, 5.5, 2.2,
             "SageMaker Model Registry",
             [
                 "Model Package Group: kkbox-churn-champion",
                 "Versioned + approval-gated",
                 "Artifact lineage: S3 URI + container",
                 "Metadata: metrics.json + configs",
             ],
             C_CLOUD)

    # Arrow: API → SageMaker
    draw_arrow(ax, 4.25, 9.2, 4.25, 8.4)

    # Arrow: SageMaker → Registry
    draw_side_arrow(ax, 7.0, 7.3, 8.5, 7.3, "register_model.py")

    # ── Row 6: Artifacts ───────────────────────────────────────
    draw_box(ax, 1.5, 3.6, 12.5, 2.0,
             "6  Champion Artifact Bundle  (artifacts/champion/)",
             [
                 "model.pkl    feature_list.json    flaml_best_params.json    metrics.json    threshold.json    valid_scored.parquet",
                 "Frozen, versioned, reproducible — every reported metric traces to this bundle",
                 "Deployed via FastAPI (local) and registered in SageMaker Model Registry (cloud)",
             ],
             "#ECEFF1")

    # Arrow: above → artifacts
    draw_arrow(ax, 8, 6.2, 8, 5.6)

    # ── Row 7: Monitoring ──────────────────────────────────────
    draw_box(ax, 1.5, 1.0, 12.5, 2.0,
             "7  Monitoring Plan  (designed, not deployed)",
             [
                 "Input drift: PSI on top SHAP features, null rate tracking, volume anomaly detection",
                 "Output drift: score distribution shift (KS test), predicted vs observed churn rate",
                 "Business: Precision@K decay tracking, ROI vs projection, retraining triggers (quarterly / performance / drift / emergency)",
             ],
             C_MONITOR)

    # Arrow: artifacts → monitoring
    draw_arrow(ax, 8, 3.6, 8, 3.0)

    # ── Save ───────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "docs" / "architecture.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
