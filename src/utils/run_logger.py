#!/usr/bin/env python3
"""
Lightweight experiment logger (single-file, append-only).

Writes JSON Lines (JSONL): one run per line -> easy to parse into pandas later.
Example run_id: "xgboost-trainapi-0007"
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return out or None
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    return _safe_run(["git", "rev-parse", "HEAD"])


def _git_branch() -> Optional[str]:
    return _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_last_run_number(log_path: Path, model_name: str) -> int:
    """
    Reads the JSONL file (if present) and finds the max run_number for the given model_name.
    This is O(file size) but fine for a lightweight setup.
    """
    if not log_path.exists():
        return 0

    max_n = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("model_name") == model_name:
                n = int(rec.get("run_number", 0))
                if n > max_n:
                    max_n = n
    return max_n


@dataclass
class RunRecord:
    run_id: str
    model_name: str
    run_number: int
    status: str  # "success" | "failed"
    started_at_utc: str
    finished_at_utc: str
    duration_sec: float

    # Useful context
    host: str
    user: str
    git_commit: Optional[str]
    git_branch: Optional[str]

    # Your content
    dataset: Dict[str, Any]
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    notes: str


class RunLogger:
    def __init__(self, log_file: str = "reports/experiment_runs.jsonl") -> None:
        self.log_path = Path(log_file)
        _ensure_parent_dir(self.log_path)

    def start_run(self, model_name: str) -> Dict[str, Any]:
        last_n = _load_last_run_number(self.log_path, model_name)
        run_number = last_n + 1
        run_id = f"{model_name}-{run_number:04d}"

        ctx = {
            "run_id": run_id,
            "model_name": model_name,
            "run_number": run_number,
            "started_at_utc": _utc_now_iso(),
        }
        return ctx

    def end_run(
        self,
        run_ctx: Dict[str, Any],
        status: str,
        finished_at_utc: str,
        duration_sec: float,
        dataset: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> RunRecord:
        record = RunRecord(
            run_id=run_ctx["run_id"],
            model_name=run_ctx["model_name"],
            run_number=int(run_ctx["run_number"]),
            status=status,
            started_at_utc=run_ctx["started_at_utc"],
            finished_at_utc=finished_at_utc,
            duration_sec=float(duration_sec),
            host=socket.gethostname(),
            user=os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            git_commit=_git_commit(),
            git_branch=_git_branch(),
            dataset=dataset or {},
            params=params or {},
            metrics=metrics or {},
            artifacts=artifacts or {},
            notes=notes,
        )
        return record

    def append(self, record: RunRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def log_run(
        self,
        model_name: str,
        train_fn,
        *,
        dataset: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        notes: str = "",
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Wrap any training/eval function so it automatically logs to JSONL.

        train_fn MUST return a dict of metrics, e.g.:
            {"roc_auc": 0.9875, "pr_auc": 0.8771, "logloss": 0.12}
        """
        import time

        ctx = self.start_run(model_name)
        t0 = time.time()

        try:
            metrics = train_fn()
            status = "success"
            err = None
        except Exception as e:
            metrics = {}
            status = "failed"
            err = repr(e)

        t1 = time.time()
        finished = _utc_now_iso()

        final_artifacts = dict(artifacts or {})
        if err is not None:
            final_artifacts["error"] = err

        record = self.end_run(
            run_ctx=ctx,
            status=status,
            finished_at_utc=finished,
            duration_sec=(t1 - t0),
            dataset=dataset,
            params=params,
            metrics=metrics,
            artifacts=final_artifacts,
            notes=notes,
        )
        self.append(record)

        # Return record as dict (handy for printing)
        return asdict(record)


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    logger = RunLogger("reports/experiment_runs.jsonl")

    # Example: wrap your training code with a function that returns metrics
    def train_and_eval():
        # Replace with your real pipeline:
        # - load data
        # - train
        # - predict
        # - compute metrics
        return {
            "roc_auc": 0.9875,
            "pr_auc": 0.8771,
            "threshold": 0.5,
        }

    record = logger.log_run(
        model_name="xgboost-trainapi",
        train_fn=train_and_eval,
        dataset={"name": "kkbox", "split": "train/valid", "version": "v1"},
        params={"max_depth": 8, "eta": 0.05, "subsample": 0.8},
        notes="Baseline XGBoost train API run. No threshold tuning yet.",
        artifacts={"model_path": "artifacts/xgb.json"},
    )

    print("Logged run:", record["run_id"], "status:", record["status"])
