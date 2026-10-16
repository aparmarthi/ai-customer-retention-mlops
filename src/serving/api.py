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
  artifacts/champion/deployment_policy_v2.json
  artifacts/champion/categorical_cols.json (optional)

Returns:
  churn_probability
  churn_label  (threshold policy OR top-K policy for batch)
  action       (contact_now | nudge | monitor, from deployment_policy_v2.json)
  /predict also returns the top TreeSHAP contributions and a missing-feature count;
  /predict_batch returns a score-mix drift check (PSI) for batches of 1,000+.

Set API_KEY to require an X-API-Key header on the POST endpoints (unset = public demo).
"""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.serving.policy import assign_action, load_policy, score_drift

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
POLICY_PATH = ART_DIR / "deployment_policy_v2.json"

MAX_BATCH_RECORDS = 50_000
TOP_N_EXPLANATIONS = 5

DEFAULT_TIME_COL = "txn_last_date"  # not used directly here, but kept for clarity


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Decision Intelligence API",
    version="1.1.0",
    description="Predict churn probability and return an action label using champion artifacts.",
)

# Browser front ends call this API cross-origin. Public read-only API with no cookies,
# so "*" is the default; lock down via CORS_ALLOW_ORIGINS="https://a.com,https://b.com".
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── globals (loaded at startup) ────────────────────────────────────────────────
MODEL = None
THRESH_DOC: Dict[str, Any] = {}
FEATURE_COLS: List[str] = []
CATEGORICAL_COLS: List[str] = []
POLICY_DOC: Dict[str, Any] = {}


# ── pydantic models ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    record: Dict[str, Any] = Field(..., description="Single feature record as JSON object.")
    policy: str = Field("threshold", description="Policy to use: 'threshold' (default).")
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override threshold for this request. If not provided, uses artifacts/champion/threshold.json",
    )
    explain: bool = Field(True, description="Return the top TreeSHAP contributions for this score.")


class Explanation(BaseModel):
    feature: str
    value: Optional[Union[float, str]] = None  # None = missing in the request
    contribution: float  # log-odds; positive raises churn risk


class PredictResponse(BaseModel):
    churn_probability: float
    churn_label: int
    action: str
    policy_used: str
    threshold_used: Optional[float] = None
    explanations: List[Explanation] = Field(default_factory=list)
    base_value: Optional[float] = None  # model log-odds before any feature contributes
    n_missing_features: int = 0
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of JSON feature records.")
    policy: str = Field("threshold", description="Policy: 'threshold' (default) or 'top_k'.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override threshold (threshold policy).")
    k: Optional[int] = Field(None, description="Top-K (required for 'top_k' policy).")


class BatchPredictItem(BaseModel):
    churn_probability: float
    churn_label: int
    action: str
    rank: Optional[int] = None  # 1 = highest risk
    threshold_used: Optional[float] = None
    policy_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictResponse(BaseModel):
    n: int
    policy_used: str
    items: List[BatchPredictItem]
    n_mostly_missing: int = 0  # records missing more than half of the features
    score_drift: Optional[Dict[str, Any]] = None  # PSI vs validation tier mix; None under 1,000 records


# ── loading ───────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _startup_load() -> None:
    global MODEL, THRESH_DOC, FEATURE_COLS, CATEGORICAL_COLS, POLICY_DOC

    missing = [p for p in [MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH, POLICY_PATH] if not p.exists()]
    if missing:
        logger.error("Missing required artifact(s): %s", missing)
        raise RuntimeError(f"Missing required artifact(s): {missing}")

    MODEL = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)

    THRESH_DOC = _load_json(THRESHOLD_PATH)
    POLICY_DOC = load_policy(POLICY_PATH)
    logger.info("Loaded policy %s %s", POLICY_DOC.get("policy_name"), POLICY_DOC.get("version"))
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


def _explain(X: pd.DataFrame, record: Dict[str, Any]) -> tuple[List[Explanation], float]:
    """Top TreeSHAP contributions (log-odds) for one row.

    LightGBM's pred_contrib returns one column per feature plus the bias; together they
    sum exactly to the row's log-odds, so these are the model's own attributions.
    """
    contrib = MODEL.predict(X, pred_contrib=True)[0]
    base, feats = float(contrib[-1]), contrib[:-1]
    top = np.argsort(-np.abs(feats))[:TOP_N_EXPLANATIONS]
    items = []
    for i in top:
        value = record.get(FEATURE_COLS[i])
        if not isinstance(value, (int, float, str)) or (isinstance(value, float) and np.isnan(value)):
            value = None
        items.append(Explanation(feature=FEATURE_COLS[i], value=value, contribution=round(float(feats[i]), 4)))
    return items, base


def _missing_warning(n_missing: int) -> List[str]:
    if n_missing <= len(FEATURE_COLS) // 2:
        return []
    return [
        f"{n_missing} of {len(FEATURE_COLS)} features are missing. The model reads missing activity "
        "as a churn signal, so check field names against the feature list at GET /."
    ]


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Enforce X-API-Key when the API_KEY env var is set; no-op for the public demo."""
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


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
    """Landing page: links to the docs and lists the feature contract."""
    return {
        "service": "Churn Decision Intelligence API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/predict_batch"],
        "features": FEATURE_COLS,
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
            "policy": str(POLICY_PATH),
        },
        "n_features": len(FEATURE_COLS),
        "default_threshold": _get_default_threshold(),
        "policy_version": POLICY_DOC.get("version"),
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Single-record prediction with the model's top TreeSHAP contributions.
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
    explanations, base_value = _explain(X, req.record) if req.explain else ([], None)
    inference_ms = (time.time() - t0) * 1000
    label = int(_apply_threshold_policy(np.array([proba]), threshold)[0])
    n_missing = int(X.iloc[0].isna().sum())

    logger.info(
        "predict n=1 proba=%.4f label=%d threshold=%.4f missing=%d inference_ms=%.1f",
        proba, label, threshold, n_missing, inference_ms,
    )

    return PredictResponse(
        churn_probability=proba,
        churn_label=label,
        action=assign_action(proba, POLICY_DOC),
        policy_used="threshold",
        threshold_used=threshold,
        explanations=explanations,
        base_value=base_value,
        n_missing_features=n_missing,
        warnings=_missing_warning(n_missing),
        metadata={
            "note": "Single prediction uses threshold policy. Top-K requires batch context (ranking).",
            "action_policy": POLICY_DOC.get("version"),
        },
    )


# Request is parsed manually (not via typed params) so one endpoint can accept JSON or multipart.
# Declaring both a pydantic body and File() makes FastAPI expect multipart only, silently
# dropping JSON bodies. openapi_extra keeps both content types documented in /docs.
_BATCH_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": BatchPredictRequest.model_json_schema()},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary"}},
                    "required": ["file"],
                }
            },
        },
    }
}


async def _read_batch(request: Request) -> tuple[List[Dict[str, Any]], Optional[BatchPredictRequest]]:
    """Return records from a CSV upload or JSON body (plus the parsed JSON request, if any)."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if not hasattr(file, "read"):
            raise HTTPException(status_code=400, detail="Multipart upload must include a CSV in field 'file'.")
        content = await file.read()
        try:
            df_in = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
        return df_in.to_dict(orient="records"), None

    if content_type.startswith("application/json"):
        try:
            req = BatchPredictRequest.model_validate(await request.json())
        except ValueError as e:  # covers malformed JSON and pydantic ValidationError
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {e}")
        return req.records, req

    raise HTTPException(status_code=400, detail="Provide either JSON body or CSV file upload.")


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    openapi_extra=_BATCH_OPENAPI,
    dependencies=[Depends(require_api_key)],
)
async def predict_batch(
    request: Request,
    policy: str = "threshold",
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    k: Optional[int] = None,
) -> BatchPredictResponse:
    """
    Batch prediction via:
      A) JSON body: {"records": [...], "policy": "threshold"|"top_k", "threshold": ..., "k": ...}
      B) CSV upload: multipart/form-data with file=<csv>; policy/threshold/k as query params,
         e.g. POST /predict_batch?policy=top_k&k=10000

    CSV must include feature columns (extras allowed). Up to 50,000 records per call.
    """
    records, req = await _read_batch(request)
    if req is not None:
        policy, threshold, k = req.policy, req.threshold, req.k

    if not records:
        raise HTTPException(status_code=400, detail="No records provided.")
    if len(records) > MAX_BATCH_RECORDS:
        raise HTTPException(status_code=413, detail=f"Batch limit is {MAX_BATCH_RECORDS:,} records per call.")

    # ── predict probabilities ────────────────────────────────────────────────
    X = _to_feature_frame(records)
    t0 = time.time()
    probas = _predict_proba(X)
    inference_ms = (time.time() - t0) * 1000
    actions = [assign_action(float(p), POLICY_DOC) for p in probas]
    n_mostly_missing = int((X.isna().sum(axis=1) > len(FEATURE_COLS) // 2).sum())
    drift = score_drift(probas, POLICY_DOC)

    logger.info(
        "predict_batch n=%d policy=%s mean_proba=%.4f mostly_missing=%d psi=%s inference_ms=%.1f",
        len(probas), policy, float(probas.mean()), n_mostly_missing,
        drift["psi"] if drift else None, inference_ms,
    )

    # ── apply policy ─────────────────────────────────────────────────────────
    if policy == "threshold":
        thr = float(threshold) if threshold is not None else _get_default_threshold()
        labels = _apply_threshold_policy(probas, thr)
        items = [
            BatchPredictItem(churn_probability=float(p), churn_label=int(y), action=a, threshold_used=thr, policy_used="threshold")
            for p, y, a in zip(probas, labels, actions)
        ]
        return BatchPredictResponse(
            n=len(items), policy_used="threshold", items=items, n_mostly_missing=n_mostly_missing, score_drift=drift
        )

    if policy == "top_k":
        if k is None:
            raise HTTPException(status_code=400, detail="For policy='top_k', provide k (e.g., 10000).")
        labels, ranks = _apply_topk_policy(probas, k=int(k))

        # Report the validation-window score cutoff when k matches the ops capacity in threshold.json
        equiv_thr = None
        if int(THRESH_DOC.get("ops_top_k", {}).get("k", -1)) == int(k):
            equiv_thr = float(THRESH_DOC["ops_top_k"]["equiv_threshold"])

        used = f"top_k({int(k)})"
        items = [
            BatchPredictItem(
                churn_probability=float(p), churn_label=int(y), action=a, rank=int(r), threshold_used=equiv_thr, policy_used=used
            )
            for p, y, a, r in zip(probas, labels, actions, ranks)
        ]
        return BatchPredictResponse(
            n=len(items), policy_used=used, items=items, n_mostly_missing=n_mostly_missing, score_drift=drift
        )

    raise HTTPException(status_code=400, detail="policy must be 'threshold' or 'top_k'.")
