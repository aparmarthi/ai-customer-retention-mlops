"""
Streamlit demo UI that consumes the FastAPI churn inference service.

Tabs:
  1) Single Prediction  - user provides one JSON record -> calls POST /predict
  2) Batch Scoring      - upload CSV -> calls POST /predict_batch (CSV) or fallback to JSON list
  3) ROI Simulator      - sliders -> computes expected ROI using your precision@K and assumptions

Run:
  # Terminal 1 (API)
  uvicorn src.serving.api:app --reload --port 8000

  # Terminal 2 (UI)
  streamlit run src/serving/app_streamlit.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAMPION_DIR = PROJECT_ROOT / "artifacts" / "champion"

# Reads from env var so cloud deployments auto-point to the right backend.
# Set CHURN_API_URL=https://your-api.onrender.com in Render / Streamlit Cloud secrets.
DEFAULT_API_BASE = os.environ.get("CHURN_API_URL", "http://127.0.0.1:8000")


def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def _api_urls(api_base: str) -> dict:
    api_base = api_base.rstrip("/")
    return {
        "health": f"{api_base}/health",
        "predict": f"{api_base}/predict",
        "predict_batch": f"{api_base}/predict_batch",
    }


def _check_health(url: str) -> Tuple[bool, str]:
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            return True, r.text
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


def _post_json(url: str, payload: Any, timeout: int = 60) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def _post_csv_multipart(url: str, csv_bytes: bytes, filename: str = "batch.csv", timeout: int = 120) -> requests.Response:
    files = {"file": (filename, csv_bytes, "text/csv")}
    return requests.post(url, files=files, timeout=timeout)


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, default=str)


def _compute_roi(
    n_targeted: int,
    precision_at_k: float,
    churn_cost: float,
    intervention_cost: float,
    save_rate: float,
) -> Dict[str, float]:
    """
    Expected true churners in targeted set = N * P@K
    Expected saves = N * P@K * save_rate
    Benefit = saves * churn_cost
    Cost = N * intervention_cost
    Net ROI = Benefit - Cost
    """
    expected_churners = n_targeted * precision_at_k
    expected_saves = expected_churners * save_rate
    benefit = expected_saves * churn_cost
    cost = n_targeted * intervention_cost
    net = benefit - cost
    return {
        "n_targeted": float(n_targeted),
        "precision_at_k": float(precision_at_k),
        "expected_true_churners_in_targeted": float(expected_churners),
        "expected_saves": float(expected_saves),
        "benefit": float(benefit),
        "cost": float(cost),
        "net_roi": float(net),
        "net_roi_per_targeted_user": float(net / n_targeted) if n_targeted > 0 else 0.0,
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Churn Decision Intelligence Demo", layout="wide")

st.title("📉 Churn Decision Intelligence Demo")
st.caption("Streamlit UI that calls FastAPI for predictions (clean separation of concerns).")

with st.sidebar:
    st.header("API Connection")
    api_base = st.text_input("FastAPI base URL", value=DEFAULT_API_BASE)
    urls = _api_urls(api_base)

    ok, msg = _check_health(urls["health"])
    if ok:
        st.success("API healthy")
        st.code(msg)
    else:
        st.error("API not reachable")
        st.code(msg)
        st.info("Start FastAPI with: uvicorn src.serving.api:app --reload --port 8000")

    st.divider()
    st.header("Champion Bundle (local)")
    threshold_doc = _load_json(CHAMPION_DIR / "threshold.json")
    feature_list = _load_json(CHAMPION_DIR / "feature_list.json")

    if threshold_doc:
        st.success("Loaded artifacts/champion/threshold.json")
    else:
        st.warning("Missing artifacts/champion/threshold.json (ROI tab will still work with manual inputs)")

    if feature_list and isinstance(feature_list, list):
        st.success(f"Loaded feature_list.json ({len(feature_list)} features)")
    else:
        st.warning("Missing artifacts/champion/feature_list.json (single-record template will be generic)")

tabs = st.tabs(["Single Prediction", "Batch Scoring", "ROI Simulator"])


# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tabs[0]:
    st.subheader("Single Prediction")
    st.write("Paste a single record as JSON. The UI calls `POST /predict` and displays the returned decision.")

    # Create a helpful template
    if feature_list and isinstance(feature_list, list) and len(feature_list) > 0:
        template = {k: 0 for k in feature_list[: min(20, len(feature_list))]}
        template["_note"] = f"Template shows first {min(20, len(feature_list))} features. Add the rest if needed."
    else:
        template = {"example_feature_1": 0, "example_feature_2": 0}

    default_payload = _pretty_json(template)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        payload_text = st.text_area("Input JSON", value=default_payload, height=320)
        send = st.button("🔮 Predict", type="primary")

    with col2:
        st.markdown("**Response**")
        if send:
            try:
                payload = json.loads(payload_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                payload = None

            if payload is not None:
                try:
                    # Auto-wrap flat feature dicts in the PredictRequest envelope.
                    # Accepts both formats:
                    #   { "city": 1, "bd": 29, ... }           <- flat features (template default)
                    #   { "record": {...}, "policy": "...", "threshold": 0.21 }  <- full envelope
                    if "record" not in payload:
                        api_payload = {"record": payload, "policy": "threshold"}
                    else:
                        api_payload = payload

                    r = _post_json(urls["predict"], api_payload, timeout=60)
                    if r.ok:
                        st.success("Prediction returned")
                        st.code(_pretty_json(r.json()))
                    else:
                        st.error(f"API error: HTTP {r.status_code}")
                        st.code(r.text)
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.info(
        "Note: top-K policies require *batch context* (ranking across many users). "
        "Single predictions typically use the tuned threshold fallback."
    )


# -----------------------------
# Tab 2: Batch Scoring
# -----------------------------
with tabs[1]:
    st.subheader("Batch Scoring")
    st.write("Upload a CSV of rows to score. The UI calls `POST /predict_batch`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(df.head(20), use_container_width=True)

            colA, colB = st.columns([1, 1])
            with colA:
                st.caption("Optionally limit rows (for quick demo)")
                max_rows = st.number_input("Max rows to score", min_value=1, value=min(5000, len(df)), step=100)
            with colB:
                st.caption("Download results after scoring")
                st.write("")

            score_btn = st.button("📦 Score Batch", type="primary")
            if score_btn:
                df_use = df.head(int(max_rows)).copy()

                # Try CSV upload first (if your API supports it)
                try:
                    csv_bytes = df_use.to_csv(index=False).encode("utf-8")
                    r = _post_csv_multipart(urls["predict_batch"], csv_bytes, filename="batch.csv", timeout=240)

                    if not r.ok:
                        # Fallback: send as list[dict] JSON
                        records = df_use.to_dict(orient="records")
                        r = _post_json(urls["predict_batch"], records, timeout=240)

                    if r.ok:
                        data = r.json()
                        st.success("Batch scored")

                        # Common response formats:
                        # 1) {"predictions":[...]}
                        # 2) [{"id":..., "churn_probability":...}, ...]
                        preds = data.get("predictions") if isinstance(data, dict) else data

                        if isinstance(preds, list):
                            out_df = pd.DataFrame(preds)
                            st.dataframe(out_df.head(50), use_container_width=True)

                            # Download
                            out_csv = out_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Download scored results (CSV)",
                                data=out_csv,
                                file_name="scored_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.code(_pretty_json(data))
                    else:
                        st.error(f"API error: HTTP {r.status_code}")
                        st.code(r.text)

                except Exception as e:
                    st.error(f"Batch request failed: {e}")

        except Exception as e:
            st.error(f"Could not read CSV file: {e}")


# -----------------------------
# Tab 3: ROI Simulator
# -----------------------------
with tabs[2]:
    st.subheader("ROI Simulator")
    st.write(
        "Use sliders to estimate expected ROI for a targeting policy. "
        "This is a business-facing layer built on top of model precision@K."
    )

    # Defaults from your threshold.json if available
    default_k = 10_000
    default_precision_at_k = 0.179

    if threshold_doc and isinstance(threshold_doc, dict):
        try:
            # Prefer precision@10k from saved doc
            pmap = threshold_doc.get("precision_at_k", {})
            if isinstance(pmap, dict):
                default_precision_at_k = float(pmap.get("10k", default_precision_at_k))
            # Prefer ops K from assumptions if present
            assump = threshold_doc.get("assumptions", {})
            if isinstance(assump, dict):
                default_k = int(assump.get("ops_budget_k", default_k))
        except Exception:
            pass

    col1, col2, col3 = st.columns(3)

    with col1:
        n_targeted = st.slider("Monthly contacts (N)", min_value=1000, max_value=50000, value=int(default_k), step=1000)
        precision_at_k = st.slider(
            "Precision@K (churn concentration)",
            min_value=0.01,
            max_value=0.60,
            value=float(default_precision_at_k),
            step=0.01,
        )

    with col2:
        churn_cost = st.slider("Value of retaining 1 churner ($)", min_value=10, max_value=500, value=120, step=10)
        intervention_cost = st.slider("Cost per contact ($)", min_value=0.0, max_value=20.0, value=0.50, step=0.25)

    with col3:
        save_rate = st.slider("Save rate among true churners", min_value=0.01, max_value=0.50, value=0.12, step=0.01)

    roi = _compute_roi(
        n_targeted=int(n_targeted),
        precision_at_k=float(precision_at_k),
        churn_cost=float(churn_cost),
        intervention_cost=float(intervention_cost),
        save_rate=float(save_rate),
    )

    st.markdown("### Results")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Expected true churners in targeted set", f"{roi['expected_true_churners_in_targeted']:.0f}")
    kpi2.metric("Expected saves", f"{roi['expected_saves']:.0f}")
    kpi3.metric("Expected benefit ($)", f"{roi['benefit']:.0f}")
    kpi4.metric("Net ROI ($)", f"{roi['net_roi']:.0f}")

    st.caption("Details")
    st.code(_pretty_json(roi))

    st.info(
        "This ROI calculator is intentionally simple: it converts model precision@K into expected saves, "
        "then applies business assumptions (save rate, churn value, contact cost). "
        "In production, you’d validate uplift via an A/B test."
    )