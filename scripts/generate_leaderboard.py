from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

RUNS_PATH = Path("reports/experiment_runs.jsonl")
OUT_PATH = Path("leaderboard.md")


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def load_runs() -> List[Dict[str, Any]]:
    if not RUNS_PATH.exists():
        raise FileNotFoundError(f"Missing run log: {RUNS_PATH}")
    runs: List[Dict[str, Any]] = []
    with RUNS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs


def best_success_per_model(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Keep only the best SUCCESS run per model_name by PR-AUC (then ROC-AUC).
    Assumes your RunLogger stores something like:
      - model_name
      - status
      - result metrics inside: record["result"] or record["metrics"]
    We handle both.
    """
    best: Dict[str, Dict[str, Any]] = {}

    for r in runs:
        status = r.get("status")
        if status != "success":
            continue

        model_name = r.get("model_name", "unknown")
        # support both conventions
        metrics = r.get("result") or r.get("metrics") or {}

        pr = _safe_float(metrics.get("pr_auc"))
        roc = _safe_float(metrics.get("roc_auc"))

        # if metrics missing, skip
        if pr is None and roc is None:
            continue

        if model_name not in best:
            best[model_name] = r
            continue

        cur = best[model_name]
        cur_metrics = cur.get("result") or cur.get("metrics") or {}
        cur_pr = _safe_float(cur_metrics.get("pr_auc"))
        cur_roc = _safe_float(cur_metrics.get("roc_auc"))

        # compare (pr desc, roc desc)
        cur_key = (cur_pr if cur_pr is not None else -1.0, cur_roc if cur_roc is not None else -1.0)
        new_key = (pr if pr is not None else -1.0, roc if roc is not None else -1.0)

        if new_key > cur_key:
            best[model_name] = r

    return best


def make_markdown(best: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for model_name, r in best.items():
        metrics = r.get("result") or r.get("metrics") or {}
        params = r.get("params") or {}
        run_id = r.get("run_id", "—")

        rows.append(
            {
                "model": model_name,
                "run_id": run_id,
                "pr_auc": _safe_float(metrics.get("pr_auc")),
                "roc_auc": _safe_float(metrics.get("roc_auc")),
                "f1": _safe_float(metrics.get("f1")),
                "precision": _safe_float(metrics.get("precision")),
                "recall": _safe_float(metrics.get("recall")),
                "accuracy": _safe_float(metrics.get("accuracy")),
                "threshold": _safe_float(metrics.get("threshold")),
                "artifact_model": metrics.get("artifact_model") or params.get("artifact_model") or "",
            }
        )

    # sort by PR-AUC desc then ROC-AUC desc
    rows.sort(
        key=lambda x: (
            x["pr_auc"] if x["pr_auc"] is not None else -1.0,
            x["roc_auc"] if x["roc_auc"] is not None else -1.0,
        ),
        reverse=True,
    )

    lines = []
    lines.append("# KKBox Churn — Model Leaderboard\n")
    lines.append("Sorted by **PR-AUC (Average Precision)**, then ROC-AUC. Metrics are on the **valid split**.\n")

    lines.append("| Rank | Model | Run ID | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | Threshold |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for i, x in enumerate(rows, 1):
        lines.append(
            f"| {i} | {x['model']} | {x['run_id']} | {_fmt(x['pr_auc'])} | {_fmt(x['roc_auc'])} | {_fmt(x['f1'])} | {_fmt(x['precision'])} | {_fmt(x['recall'])} | {_fmt(x['accuracy'])} | {_fmt(x['threshold'])} |"
        )

    lines.append("\n## Notes\n")
    lines.append("- PR-AUC is the primary metric because churn is highly imbalanced (~6.4% positive).")
    lines.append("- Threshold-dependent metrics (Precision/Recall/F1/Accuracy) assume your default `threshold` (often 0.50).")
    lines.append("- Champion selection and threshold policy are documented in `artifacts/champion/`.\n")

    return "\n".join(lines)


def main() -> None:
    runs = load_runs()
    best = best_success_per_model(runs)
    md = make_markdown(best)
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"✅ Wrote {OUT_PATH} using {len(best)} best-per-model runs from {RUNS_PATH}")


if __name__ == "__main__":
    main()
