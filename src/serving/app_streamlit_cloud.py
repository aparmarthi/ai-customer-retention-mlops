"""
Self-contained Streamlit app for Streamlit Community Cloud deployment.

Loads the champion LightGBM model directly (no FastAPI dependency).
All inference runs in-process — single URL, zero backend setup.

Local development:
    streamlit run src/serving/app_streamlit_cloud.py

Production architecture note:
    The companion app_streamlit.py + api.py pair demonstrates clean
    separation of concerns (UI → API → Model). This cloud version
    trades that separation for deployment simplicity.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = PROJECT_ROOT / "artifacts" / "champion"

MODEL_PATH = ART_DIR / "model.pkl"
THRESHOLD_PATH = ART_DIR / "threshold.json"
FEATURES_PATH = ART_DIR / "feature_list.json"
CATEGORICAL_PATH = ART_DIR / "categorical_cols.json"
METRICS_PATH = ART_DIR / "metrics.json"
ROI_POLICY_PATH = ART_DIR / "roi_policy.json"


# ─── Load artifacts (cached so they load only once) ──────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: str) -> Optional[dict | list]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None


# ─── Inference helpers ────────────────────────────────────────────────────────
def prepare_features(records: List[Dict[str, Any]], feature_cols: List[str],
                     cat_cols: List[str]) -> pd.DataFrame:
    """Convert list of dicts to model-ready DataFrame."""
    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_cols].copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df)[:, 1].astype(float)


def get_default_threshold(threshold_doc: dict) -> float:
    try:
        return float(threshold_doc["roi_optimal"]["threshold"])
    except Exception:
        return 0.5


# ─── ROI calculator ──────────────────────────────────────────────────────────
def compute_roi(n_targeted: int, precision_at_k: float, churn_cost: float,
                intervention_cost: float, save_rate: float) -> Dict[str, float]:
    expected_churners = n_targeted * precision_at_k
    expected_saves = expected_churners * save_rate
    benefit = expected_saves * churn_cost
    cost = n_targeted * intervention_cost
    net = benefit - cost
    return {
        "n_targeted": n_targeted,
        "precision_at_k": precision_at_k,
        "expected_true_churners": round(expected_churners),
        "expected_saves": round(expected_saves),
        "benefit": round(benefit),
        "cost": round(cost),
        "net_roi": round(net),
        "roi_per_user": round(net / n_targeted, 2) if n_targeted > 0 else 0.0,
    }


# ─── Sample data for "Try it" button ─────────────────────────────────────────
SAMPLE_RECORDS = [
    {"city": 22, "bd": 25, "gender": "female", "registered_via": 9,
     "registration_init_time": 20111121, "txn_cnt": 20, "cancel_cnt": 1,
     "cancel_rate": 0.05, "auto_renew_cnt": 20, "auto_renew_rate": 1.0,
     "plan_list_price_mean": 141.55, "plan_list_price_max": 149,
     "plan_list_price_min": 0, "actual_paid_mean": 149, "actual_paid_max": 149,
     "actual_paid_min": 149, "payment_method_nunique": 1,
     "txn_first_date": 20150131, "txn_tenure_days_approx": 20083,
     "membership_expire_date_max": 20170307, "log_row_cnt": 657,
     "log_first_date": 20150101, "log_last_date": 20170228,
     "log_active_days": 657, "log_span_days_approx": 20127,
     "total_secs_sum": 5842310, "total_secs_mean_per_active_day": 8893.2,
     "total_secs_max_per_day": 47041.8, "num_unq_sum": 17687,
     "num_unq_mean_per_active_day": 26.9, "num_unq_max_per_day": 162,
     "num_25_sum": 1818, "num_50_sum": 621, "num_75_sum": 328,
     "num_985_sum": 372, "num_100_sum": 18475,
     "listen_full_share": 0.855, "listens_per_unique_track": 1.222},
    {"city": 5, "bd": 21, "gender": "female", "registered_via": 3,
     "registration_init_time": 20160611, "txn_cnt": 1, "cancel_cnt": 0,
     "cancel_rate": 0.0, "auto_renew_cnt": 0, "auto_renew_rate": 0.0,
     "plan_list_price_mean": 180, "plan_list_price_max": 180,
     "plan_list_price_min": 180, "actual_paid_mean": 180, "actual_paid_max": 180,
     "actual_paid_min": 180, "payment_method_nunique": 1,
     "txn_first_date": 20170102, "txn_tenure_days_approx": 0,
     "membership_expire_date_max": 20170201, "log_row_cnt": 6,
     "log_first_date": 20160611, "log_last_date": 20170102,
     "log_active_days": 6, "log_span_days_approx": 9491,
     "total_secs_sum": 103142, "total_secs_mean_per_active_day": 17190.4,
     "total_secs_max_per_day": 32982.2, "num_unq_sum": 342,
     "num_unq_mean_per_active_day": 57, "num_unq_max_per_day": 120,
     "num_25_sum": 439, "num_50_sum": 98, "num_75_sum": 30,
     "num_985_sum": 30, "num_100_sum": 379,
     "listen_full_share": 0.388, "listens_per_unique_track": 2.854},
    {"city": 1, "bd": 0, "gender": "unknown", "registered_via": 7,
     "registration_init_time": 20130611, "txn_cnt": 14, "cancel_cnt": 1,
     "cancel_rate": 0.071, "auto_renew_cnt": 14, "auto_renew_rate": 1.0,
     "plan_list_price_mean": 99, "plan_list_price_max": 99,
     "plan_list_price_min": 99, "actual_paid_mean": 99, "actual_paid_max": 99,
     "actual_paid_min": 99, "payment_method_nunique": 1,
     "txn_first_date": 20160215, "txn_tenure_days_approx": 10000,
     "membership_expire_date_max": 20170314, "log_row_cnt": 242,
     "log_first_date": 20160215, "log_last_date": 20170215,
     "log_active_days": 242, "log_span_days_approx": 10000,
     "total_secs_sum": 1460238, "total_secs_mean_per_active_day": 6034,
     "total_secs_max_per_day": 30731.3, "num_unq_sum": 4740,
     "num_unq_mean_per_active_day": 19.6, "num_unq_max_per_day": 118,
     "num_25_sum": 1282, "num_50_sum": 414, "num_75_sum": 283,
     "num_985_sum": 293, "num_100_sum": 5527,
     "listen_full_share": 0.709, "listens_per_unique_track": 1.645},
    {"city": 13, "bd": 33, "gender": "male", "registered_via": 9,
     "registration_init_time": 20150920, "txn_cnt": 5, "cancel_cnt": 0,
     "cancel_rate": 0.0, "auto_renew_cnt": 5, "auto_renew_rate": 1.0,
     "plan_list_price_mean": 149, "plan_list_price_max": 149,
     "plan_list_price_min": 149, "actual_paid_mean": 149, "actual_paid_max": 149,
     "actual_paid_min": 149, "payment_method_nunique": 1,
     "txn_first_date": 20160801, "txn_tenure_days_approx": 5000,
     "membership_expire_date_max": 20170301, "log_row_cnt": 180,
     "log_first_date": 20160801, "log_last_date": 20170201,
     "log_active_days": 150, "log_span_days_approx": 5500,
     "total_secs_sum": 900000, "total_secs_mean_per_active_day": 6000,
     "total_secs_max_per_day": 25000, "num_unq_sum": 3000,
     "num_unq_mean_per_active_day": 20, "num_unq_max_per_day": 80,
     "num_25_sum": 800, "num_50_sum": 300, "num_75_sum": 200,
     "num_985_sum": 250, "num_100_sum": 3500,
     "listen_full_share": 0.72, "listens_per_unique_track": 1.5},
    {"city": 6, "bd": 45, "gender": "male", "registered_via": 4,
     "registration_init_time": 20170101, "txn_cnt": 1, "cancel_cnt": 1,
     "cancel_rate": 1.0, "auto_renew_cnt": 0, "auto_renew_rate": 0.0,
     "plan_list_price_mean": 100, "plan_list_price_max": 100,
     "plan_list_price_min": 100, "actual_paid_mean": 100, "actual_paid_max": 100,
     "actual_paid_min": 100, "payment_method_nunique": 1,
     "txn_first_date": 20170115, "txn_tenure_days_approx": 0,
     "membership_expire_date_max": 20170215, "log_row_cnt": 3,
     "log_first_date": 20170115, "log_last_date": 20170118,
     "log_active_days": 3, "log_span_days_approx": 3,
     "total_secs_sum": 5000, "total_secs_mean_per_active_day": 1667,
     "total_secs_max_per_day": 3000, "num_unq_sum": 50,
     "num_unq_mean_per_active_day": 17, "num_unq_max_per_day": 25,
     "num_25_sum": 20, "num_50_sum": 5, "num_75_sum": 3,
     "num_985_sum": 2, "num_100_sum": 30,
     "listen_full_share": 0.30, "listens_per_unique_track": 2.0},
]


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor — Amey Parmarth",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hero gradient banner */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 { color: white; margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero-banner p  { color: rgba(255,255,255,0.9); margin: 0.2rem 0; font-size: 1rem; }
    .hero-links a   { color: white; text-decoration: none; margin-right: 1.5rem;
                      font-weight: 600; border-bottom: 2px solid rgba(255,255,255,0.5); }
    .hero-links a:hover { border-bottom-color: white; }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .metric-card.green  { border-left-color: #28a745; }
    .metric-card.red    { border-left-color: #dc3545; }
    .metric-card.blue   { border-left-color: #007bff; }
    .metric-card.purple { border-left-color: #764ba2; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 0.2rem; }

    /* Risk badge */
    .risk-high   { background: #dc3545; color: white; padding: 2px 10px;
                   border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .risk-low    { background: #28a745; color: white; padding: 2px 10px;
                   border-radius: 12px; font-weight: 600; font-size: 0.85rem; }

    /* Subtle section dividers */
    .section-header {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
        color: #6c757d; margin: 1.5rem 0 0.5rem 0; font-weight: 600;
    }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>📉 Churn Decision Intelligence Engine</h1>
    <p>End-to-end ML system: feature engineering → LightGBM → ROI-optimized targeting</p>
    <p style="font-size: 0.9rem; margin-top: 0.8rem;">
        Built by <strong>Amey Parmarth</strong> &nbsp;|&nbsp; KKBox Music Streaming Dataset &nbsp;|&nbsp; ~1.2% Churn Rate
    </p>
    <div class="hero-links" style="margin-top: 0.8rem;">
        <a href="https://github.com/ameyp-parmarth/ai-customer-retention-mlops" target="_blank">📂 GitHub Repo</a>
        <a href="https://www.linkedin.com/in/ameyparmarth/" target="_blank">💼 LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load everything ─────────────────────────────────────────────────────────
model = load_model()
feature_cols = load_json(str(FEATURES_PATH)) or []
cat_cols = load_json(str(CATEGORICAL_PATH)) or []
metrics = load_json(str(METRICS_PATH)) or {}
threshold_doc = load_json(str(THRESHOLD_PATH)) or {}
roi_policy = load_json(str(ROI_POLICY_PATH)) or {}

model_loaded = model is not None and len(feature_cols) > 0


# ─── Sidebar: Model info ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Info")

    if model_loaded:
        st.success("Model loaded & ready")
    else:
        st.error("Model failed to load")

    st.markdown(f"""
    | Detail | Value |
    |---|---|
    | **Algorithm** | LightGBM (FLAML AutoML) |
    | **Estimators** | 146 |
    | **Features** | {len(feature_cols)} |
    | **Validation** | Time-based holdout |
    """)

    st.markdown("### 📊 Champion Metrics")
    pr_auc = metrics.get("pr_auc", 0)
    roc_auc = metrics.get("roc_auc", 0)
    st.metric("PR-AUC", f"{pr_auc:.4f}")
    st.metric("ROC-AUC", f"{roc_auc:.4f}")

    p_at_k = metrics.get("precision_at_k", {})
    r_at_k = metrics.get("recall_at_k", {})
    st.markdown("**Precision / Recall @ K**")
    for k_val in ["5000", "10000", "20000"]:
        p = p_at_k.get(k_val, "–")
        r = r_at_k.get(k_val, "–")
        k_label = f"{int(k_val)//1000}k"
        p_str = f"{p:.4f}" if isinstance(p, float) else p
        r_str = f"{r:.4f}" if isinstance(r, float) else r
        st.caption(f"@{k_label}:  P={p_str}  R={r_str}")

    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.caption(
        "This cloud version loads the model directly for demo simplicity. "
        "The production version uses a FastAPI backend with clean separation of concerns."
    )
    st.markdown(
        "[View production API code →]"
        "(https://github.com/ameyp-parmarth/ai-customer-retention-mlops/blob/main/src/serving/api.py)"
    )


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Prediction",
    "📦 Batch Scoring",
    "💰 ROI Simulator",
    "📊 Model Explainability",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1: Single Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("Single Customer Prediction")
    st.caption("Paste a customer feature record as JSON, or click **Try Sample** to load an example.")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        # Try Sample button
        if st.button("🧪 Try Sample Customer", type="secondary"):
            st.session_state["single_payload"] = json.dumps(SAMPLE_RECORDS[4], indent=2)

        default_val = st.session_state.get(
            "single_payload",
            json.dumps({k: 0 for k in feature_cols[:15]}, indent=2)
            if feature_cols else '{"feature_1": 0}'
        )
        payload_text = st.text_area("Input JSON", value=default_val, height=380)
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>',
                    unsafe_allow_html=True)

        if predict_btn and model_loaded:
            try:
                payload = json.loads(payload_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                payload = None

            if payload is not None:
                try:
                    df_x = prepare_features([payload], feature_cols, cat_cols)
                    proba = float(predict_proba(model, df_x)[0])
                    threshold = get_default_threshold(threshold_doc)
                    label = int(proba >= threshold)

                    # Visual result
                    if label == 1:
                        st.markdown(
                            f'<div style="text-align:center; padding: 1.5rem;">'
                            f'<span class="risk-high">⚠️ HIGH CHURN RISK</span>'
                            f'<div class="metric-value" style="margin-top: 1rem;">'
                            f'{proba:.1%}</div>'
                            f'<div class="metric-label">churn probability</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div style="text-align:center; padding: 1.5rem;">'
                            f'<span class="risk-low">✅ LOW RISK</span>'
                            f'<div class="metric-value" style="margin-top: 1rem;">'
                            f'{proba:.1%}</div>'
                            f'<div class="metric-label">churn probability</div>'
                            f'</div>',
                            unsafe_allow_html=True)

                    st.divider()
                    st.caption("Details")
                    st.json({
                        "churn_probability": round(proba, 6),
                        "churn_label": label,
                        "threshold_used": threshold,
                        "policy": "ROI-optimal threshold",
                    })
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        elif predict_btn and not model_loaded:
            st.error("Model not loaded. Check artifacts/champion/ directory.")

    st.info(
        "💡 **Interview note:** Single predictions use the ROI-optimal threshold "
        f"({get_default_threshold(threshold_doc):.2f}). "
        "Top-K ranking requires batch context — see the Batch Scoring tab."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2: Batch Scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("Batch Scoring")
    st.caption("Upload a CSV or use the built-in sample to score multiple customers at once.")

    col_upload, col_options = st.columns([2, 1])

    with col_options:
        st.markdown('<div class="section-header">Scoring Policy</div>',
                    unsafe_allow_html=True)
        policy = st.radio(
            "Decision policy",
            ["ROI-Optimal Threshold", "Top-K Targeting"],
            help="Threshold uses the ROI-tuned cutoff. Top-K selects the K highest-risk customers."
        )
        if policy == "Top-K Targeting":
            k_val = st.number_input("K (customers to target)", min_value=1,
                                    value=100, step=10)
        else:
            k_val = None

    with col_upload:
        use_sample = st.button("📋 Load Sample Data (5 customers)", type="secondary")
        uploaded = st.file_uploader("Or upload your own CSV", type=["csv"])

    # Determine data source
    df_score = None
    if use_sample:
        df_score = pd.DataFrame(SAMPLE_RECORDS)
        st.session_state["batch_source"] = "sample"
    elif uploaded is not None:
        try:
            df_score = pd.read_csv(uploaded)
            st.session_state["batch_source"] = "upload"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if df_score is not None:
        st.markdown(f"**Preview** ({len(df_score)} rows)")
        st.dataframe(df_score.head(20), use_container_width=True, height=200)

        score_btn = st.button("📦 Score All", type="primary", use_container_width=True)

        if score_btn and model_loaded:
            with st.spinner("Scoring..."):
                df_x = prepare_features(
                    df_score.to_dict(orient="records"), feature_cols, cat_cols)
                probas = predict_proba(model, df_x)

                results = df_score.copy()
                results["churn_probability"] = probas.round(6)

                threshold = get_default_threshold(threshold_doc)

                if policy == "Top-K Targeting" and k_val is not None:
                    k = min(int(k_val), len(results))
                    # Rank: 1 = highest risk
                    results["rank"] = results["churn_probability"].rank(
                        ascending=False, method="first").astype(int)
                    results["action"] = np.where(results["rank"] <= k,
                                                 "🎯 TARGET", "—")
                    results = results.sort_values("rank")
                    policy_label = f"Top-{k}"
                else:
                    results["churn_label"] = (probas >= threshold).astype(int)
                    results["action"] = np.where(results["churn_label"] == 1,
                                                 "⚠️ CHURN RISK", "✅ Retain")
                    results = results.sort_values("churn_probability",
                                                  ascending=False)
                    policy_label = f"Threshold ({threshold:.2f})"

            # Summary metrics
            n_flagged = (results["action"].str.contains("TARGET|CHURN", na=False)).sum()
            avg_proba = probas.mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Customers Scored", len(results))
            m2.metric("Flagged for Action", n_flagged)
            m3.metric("Avg Churn Probability", f"{avg_proba:.1%}")

            st.markdown(f"**Results** — Policy: *{policy_label}*")
            st.dataframe(results, use_container_width=True, height=300)

            # Download button
            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 3: ROI Simulator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("ROI Simulator")
    st.caption(
        "Translate model precision into dollar impact. "
        "Adjust assumptions to see how targeting strategy affects ROI."
    )

    # Load defaults from threshold doc
    assumptions = threshold_doc.get("assumptions", {}) if threshold_doc else {}
    default_k = int(assumptions.get("ops_budget_k", 10000))
    default_p_at_k = float(
        threshold_doc.get("precision_at_k", {}).get("10k", 0.1801)
        if threshold_doc else 0.1801
    )
    default_churn_cost = float(assumptions.get("churn_cost", 120))
    default_intervention = float(assumptions.get("intervention_cost", 5))
    default_save_rate = float(assumptions.get("save_rate_if_targeted", 0.2))

    st.markdown('<div class="section-header">Business Assumptions</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        n_targeted = st.slider("Monthly contacts (N)", 1000, 50000,
                               default_k, step=1000)
        precision_at_k = st.slider("Precision@K", 0.01, 0.60,
                                   default_p_at_k, step=0.01,
                                   help="Fraction of targeted customers who are true churners")
    with c2:
        churn_cost = st.slider("Value of retaining 1 churner ($)", 10, 500,
                               int(default_churn_cost), step=10)
        intervention_cost = st.slider("Cost per contact ($)", 0.0, 20.0,
                                      default_intervention, step=0.25)
    with c3:
        save_rate = st.slider("Save rate", 0.01, 0.50,
                              default_save_rate, step=0.01,
                              help="Probability that a contacted churner is retained")

    roi = compute_roi(n_targeted, precision_at_k, churn_cost,
                      intervention_cost, save_rate)

    st.markdown('<div class="section-header">Projected Impact</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    net_roi = roi["net_roi"]
    roi_color = "green" if net_roi > 0 else "red"

    with k1:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-value">{roi["expected_true_churners"]:,}</div>
            <div class="metric-label">True Churners in Target</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card purple">
            <div class="metric-value">{roi["expected_saves"]:,}</div>
            <div class="metric-label">Expected Saves</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="metric-value">${roi["benefit"]:,}</div>
            <div class="metric-label">Gross Benefit</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card {roi_color}">
            <div class="metric-value">${net_roi:,}</div>
            <div class="metric-label">Net ROI</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Cost: ${roi['cost']:,}  |  ROI per targeted user: ${roi['roi_per_user']}")

    # Formula explanation
    with st.expander("📐 ROI Formula"):
        st.markdown("""
        ```
        Expected Churners = N_targeted × Precision@K
        Expected Saves    = Expected Churners × Save Rate
        Benefit           = Expected Saves × Churn Cost
        Cost              = N_targeted × Intervention Cost
        Net ROI           = Benefit − Cost
        ```
        """)

    st.info(
        "💡 **Interview note:** The ROI-optimal threshold (0.68) yields $17,666/month "
        "by targeting only 1,478 high-confidence users. The Top-10K policy maximizes "
        "recall (75%) but has negative ROI at current assumptions — a real tradeoff "
        "the business must decide."
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 4: Model Explainability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.subheader("Model Explainability")
    st.caption("Feature importance from the champion LightGBM model (split-based).")

    if model_loaded:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        # Top-15 horizontal bar chart
        top_n = st.slider("Show top N features", 5, len(feature_cols),
                          min(15, len(feature_cols)))
        top = imp_df.head(top_n).sort_values("importance", ascending=True)

        st.bar_chart(top.set_index("feature")["importance"],
                     horizontal=True, height=max(300, top_n * 28))

        # Human-readable feature descriptions
        FEATURE_DESCRIPTIONS = {
            "txn_tenure_days_approx": "How long the customer has been subscribing",
            "txn_first_date": "Date of first transaction (proxy for account age)",
            "membership_expire_date_max": "Latest membership expiry date",
            "registration_init_time": "When the account was originally registered",
            "listens_per_unique_track": "Average plays per unique song (engagement depth)",
            "total_secs_max_per_day": "Peak daily listening time",
            "bd": "Customer age (birthday)",
            "listen_full_share": "Fraction of songs played to completion",
            "num_25_sum": "Songs stopped at 25% (disengagement signal)",
            "cancel_rate": "Historical cancellation frequency",
            "auto_renew_rate": "Fraction of renewals that were automatic",
            "log_row_cnt": "Total listening log entries (usage volume)",
        }

        with st.expander("📖 Feature Descriptions (Top Drivers)"):
            for _, row in imp_df.head(12).iterrows():
                feat = row["feature"]
                desc = FEATURE_DESCRIPTIONS.get(feat, "—")
                st.markdown(f"- **{feat}**: {desc}")

        st.info(
            "💡 **Interview note:** Tenure and recency features dominate — "
            "customers with shorter tenure and recent expiry dates are highest risk. "
            "Behavioral signals (listening patterns, cancel rate) provide additional "
            "discriminative power beyond demographics."
        )
    else:
        st.warning("Model not loaded — cannot display feature importances.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align: center; color: #6c757d; font-size: 0.8rem; padding: 1rem 0;">'
    'Built with Streamlit · LightGBM · FLAML AutoML · MLflow · FastAPI<br>'
    '<a href="https://github.com/ameyp-parmarth/ai-customer-retention-mlops" '
    'style="color: #667eea;">View full project on GitHub</a>'
    '</div>',
    unsafe_allow_html=True,
)
