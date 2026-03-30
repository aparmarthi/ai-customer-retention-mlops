"""
FastAPI inference service for the churn champion model.

Endpoints:
  - GET  /health
  - POST /predict        (single record JSON)
  - POST /predict_batch  (either JSON list OR CSV upload)

Loads (on startup):
  artifacts/champion/model.pkl
  artifacts/champion/threshold.json
  artifacts/champion/feature_list.json
  artifacts/champion/categorical_cols.json (optional)

Returns:
  churn_probability
  churn_label  (threshold policy OR top-K policy for batch)
  (optional) rank, threshold_used, policy_used, metadata
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("churn_api")


# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = PROJECT_ROOT / "artifacts" / "champion"

MODEL_PATH = ART_DIR / "model.pkl"
THRESHOLD_PATH = ART_DIR / "threshold.json"
FEATURES_PATH = ART_DIR / "feature_list.json"
CATEGORICAL_PATH = ART_DIR / "categorical_cols.json"  # created by your 13_train_champion_lgbm.py

DEFAULT_TIME_COL = "txn_last_date"  # not used directly here, but kept for clarity


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Decision Intelligence API",
    version="1.0.0",
    description="Predict churn probability and return an action label using champion artifacts.",
)


# ── globals (loaded at startup) ────────────────────────────────────────────────
MODEL = None
THRESH_DOC: Dict[str, Any] = {}
FEATURE_COLS: List[str] = []
CATEGORICAL_COLS: List[str] = []


# ── pydantic models ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    record: Dict[str, Any] = Field(..., description="Single feature record as JSON object.")
    policy: str = Field("threshold", description="Policy to use: 'threshold' (default).")
    threshold: Optional[float] = Field(
        None,
        description="Override threshold for this request. If not provided, uses artifacts/champion/threshold.json",
    )


class PredictResponse(BaseModel):
    churn_probability: float
    churn_label: int
    policy_used: str
    threshold_used: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of JSON feature records.")
    policy: str = Field("threshold", description="Policy: 'threshold' (default) or 'top_k'.")
    threshold: Optional[float] = Field(None, description="Override threshold (threshold policy).")
    k: Optional[int] = Field(None, description="Top-K (required for 'top_k' policy).")


class BatchPredictItem(BaseModel):
    churn_probability: float
    churn_label: int
    rank: Optional[int] = None  # 1 = highest risk
    threshold_used: Optional[float] = None
    policy_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictResponse(BaseModel):
    n: int
    policy_used: str
    items: List[BatchPredictItem]


# ── loading ───────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _startup_load() -> None:
    global MODEL, THRESH_DOC, FEATURE_COLS, CATEGORICAL_COLS

    missing = [p for p in [MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        logger.error("Missing required artifact(s): %s", missing)
        raise RuntimeError(f"Missing required artifact(s): {missing}")

    MODEL = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)

    THRESH_DOC = _load_json(THRESHOLD_PATH)
    FEATURE_COLS = list(_load_json(FEATURES_PATH))
    logger.info("Loaded %d features, default threshold=%.4f",
                len(FEATURE_COLS), THRESH_DOC.get("roi_optimal", {}).get("threshold", 0.5))

    if CATEGORICAL_PATH.exists():
        CATEGORICAL_COLS = list(_load_json(CATEGORICAL_PATH))
        logger.info("Loaded %d categorical columns", len(CATEGORICAL_COLS))
    else:
        CATEGORICAL_COLS = []
        logger.warning("categorical_cols.json not found; falling back to dtype inference")

    if not FEATURE_COLS:
        raise RuntimeError("feature_list.json is empty; cannot serve without feature columns.")


@app.on_event("startup")
def on_startup():
    _startup_load()
    logger.info("Churn API ready — model loaded, serving on /predict and /predict_batch")


# ── feature shaping ───────────────────────────────────────────────────────────
def _to_feature_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of dicts into a DataFrame with exactly FEATURE_COLS (missing -> NaN).
    Ensures categorical columns are 'category' dtype.
    """
    df = pd.DataFrame(records)

    # Add missing cols
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Drop extra cols (keep only expected)
    df = df[FEATURE_COLS].copy()

    # Convert categorical columns to category dtype
    # If we have stored categorical_cols.json, use it.
    # Otherwise, fall back to object columns.
    cat_cols = CATEGORICAL_COLS if CATEGORICAL_COLS else [c for c in df.columns if df[c].dtype == "object"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def _predict_proba(dfX: pd.DataFrame) -> np.ndarray:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    proba = MODEL.predict_proba(dfX)[:, 1]
    return np.asarray(proba, dtype=float)


# ── policies ──────────────────────────────────────────────────────────────────
def _get_default_threshold() -> float:
    """
    Uses ROI-optimal threshold from threshold.json if present, else falls back to 0.5.
    """
    try:
        return float(THRESH_DOC["roi_optimal"]["threshold"])
    except Exception:
        return 0.5


def _apply_threshold_policy(probas: np.ndarray, threshold: float) -> np.ndarray:
    return (probas >= float(threshold)).astype(int)


def _apply_topk_policy(probas: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels: 1 for top-k highest probas else 0
      ranks: 1..n (1=highest risk), based on descending proba
    """
    n = len(probas)
    k = int(k)
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0 for top_k policy.")
    k = min(k, n)

    order = np.argsort(probas)[::-1]  # descending
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    labels = (ranks <= k).astype(int)
    return labels, ranks


# ── request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    return response


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Landing page — directs visitors to the interactive API docs."""
    return {
        "service": "Churn Decision Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/predict_batch"],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "artifacts": {
            "model": str(MODEL_PATH),
            "threshold": str(THRESHOLD_PATH),
            "feature_list": str(FEATURES_PATH),
            "categorical_cols": str(CATEGORICAL_PATH) if CATEGORICAL_PATH.exists() else None,
        },
        "n_features": len(FEATURE_COLS),
        "default_threshold": _get_default_threshold(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Single-record prediction.
    Note: top-K policy requires batch context, so this endpoint only supports threshold policy.
    """
    if req.policy not in {"threshold"}:
        raise HTTPException(
            status_code=400,
            detail="Single /predict supports only 'threshold' policy. Use /predict_batch for 'top_k'.",
        )

    threshold = float(req.threshold) if req.threshold is not None else _get_default_threshold()

    X = _to_feature_frame([req.record])
    t0 = time.time()
    proba = float(_predict_proba(X)[0])
    inference_ms = (time.time() - t0) * 1000
    label = int(_apply_threshold_policy(np.array([proba]), threshold)[0])

    logger.info(
        "predict n=1 proba=%.4f label=%d threshold=%.4f inference_ms=%.1f",
        proba, label, threshold, inference_ms,
    )

    return PredictResponse(
        churn_probability=proba,
        churn_label=label,
        policy_used="threshold",
        threshold_used=threshold,
        metadata={
            "note": "Single prediction uses threshold policy. Top-K requires batch context (ranking).",
        },
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: Optional[BatchPredictRequest] = None,
    file: Optional[UploadFile] = File(default=None),
) -> BatchPredictResponse:
    """
    Batch prediction via:
      A) JSON body: {"records": [...], "policy": "threshold"|"top_k", "threshold": ..., "k": ...}
      B) CSV upload: multipart/form-data with file=<csv>

    CSV must include feature columns (extras allowed).
    """
    if (req is None) and (file is None):
        raise HTTPException(status_code=400, detail="Provide either JSON body or CSV file upload.")

    # ── parse input records ───────────────────────────────────────────────────
    records: List[Dict[str, Any]]
    policy = "threshold"
    threshold: Optional[float] = None
    k: Optional[int] = None

    if file is not None:
        content = await file.read()
        try:
            df_in = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

        records = df_in.to_dict(orient="records")
        policy = "threshold"  # default for CSV unless you add query params later
        threshold = None
        k = None
    else:
        assert req is not None
        records = req.records
        policy = req.policy
        threshold = req.threshold
        k = req.k

    if not records:
        raise HTTPException(status_code=400, detail="No records provided.")

    # ── predict probabilities ────────────────────────────────────────────────
    X = _to_feature_frame(records)
    t0 = time.time()
    probas = _predict_proba(X)
    inference_ms = (time.time() - t0) * 1000

    logger.info(
        "predict_batch n=%d policy=%s mean_proba=%.4f inference_ms=%.1f",
        len(probas), policy, float(probas.mean()), inference_ms,
    )

    # ── apply policy ─────────────────────────────────────────────────────────
    items: List[BatchPredictItem] = []

    if policy == "threshold":
        thr = float(threshold) if threshold is not None else _get_default_threshold()
        labels = _apply_threshold_policy(probas, thr)

        for p, y in zip(probas, labels):
            items.append(
                BatchPredictItem(
                    churn_probability=float(p),
                    churn_label=int(y),
                    rank=None,
                    threshold_used=thr,
                    policy_used="threshold",
                    metadata={},
                )
            )
        return BatchPredictResponse(n=len(items), policy_used="threshold", items=items)

    if policy == "top_k":
        if k is None:
            raise HTTPException(status_code=400, detail="For policy='top_k', provide k (e.g., 10000).")
        labels, ranks = _apply_topk_policy(probas, k=int(k))

        # (Optional) include equivalent threshold from artifact doc if it matches the same k
        equiv_thr = None
        try:
            if int(THRESH_DOC.get("ops_top_k", {}).get("k", -1)) == int(k):
                equiv_thr = float(THRESH_DOC["ops_top_k"]["equiv_threshold"])
        except Exception:
            equiv_thr = None

        for p, y, r in zip(probas, labels, ranks):
            items.append(
                BatchPredictItem(
                    churn_probability=float(p),
                    churn_label=int(y),
                    rank=int(r),
                    threshold_used=equiv_thr,
                    policy_used=f"top_k({int(k)})",
                    metadata={},
                )
            )
        return BatchPredictResponse(n=len(items), policy_used=f"top_k({int(k)})", items=items)

    raise HTTPException(status_code=400, detail="policy must be 'threshold' or 'top_k'.")