# Bolt.new prompt: KKBox Retention Decision Engine UI

Paste everything below the line into bolt.new.

---

Build a polished, production-quality web app called **"Retention Decision Engine"**. It's the front end for a churn-prediction platform I built on KKBox's 31 GB subscription dataset. The audience is enterprise buyers and technical interviewers. It must feel like a real B2B SaaS analytics product (think Linear, Vercel or Stripe dashboards), not a data-science notebook.

## Stack
- React + Vite + TypeScript, Tailwind CSS, shadcn/ui components, Recharts for charts, lucide-react icons, Framer Motion for subtle transitions.
- Dark theme by default (deep navy #0B1220 background, slate cards, teal #14B8A6 accent for positive $, rose #F43F5E for negative $), with a light-mode toggle.
- Fully responsive. Left sidebar navigation on desktop, bottom tabs on mobile.
- All API calls go through one `api.ts` module that reads `import.meta.env.VITE_API_URL`. If it's unset or a request fails, fall back to "Demo mode" (clearly badged in the header) using the embedded data below. The app must never show a broken state.

## Core idea (put this on the Overview page)
"The model was never the product. The policy was." The flow is: **churn score → targeting policy → net dollars**. The same model justifies contacting 1,478 people or 10,000. Which is right depends on what an intervention costs.

## Honesty rule (important)
Every dollar figure shows a small badge: "Simulated under stated assumptions". Add a footer note: "ROI is simulated. True incremental impact requires a randomized treatment/control test (see Experiment Plan)." Never call it measured revenue.

## Embedded data (use exactly these numbers)

Holdout (chronological, later time window): N = 193,205 subscribers, churn rate 1.24%, total churners = 2,402.

Measured operating points (K = number of top-ranked users contacted, TP = true churners among them):
| K | TP | Source |
|---|----|--------|
| 0 | 0 | — |
| 1,262 | 992 | threshold 0.71 |
| 1,478 | 1,044 | threshold 0.68 (ROI-optimal) |
| 5,000 | 1,345 | P@5K 26.9% |
| 10,000 | 1,801 | P@10K 18.0% |
| 20,000 | 2,179 | P@20K 10.9% |
Interpolate TP linearly between points for any K in 0–20,000. Random targeting: TP = K × 0.0124.

Model metrics (holdout): ROC-AUC 0.966, PR-AUC 0.539 (43.5× the base rate), Recall@10K 75.0%, Recall@5K 56.0%, Recall@20K 90.7%.

Leaderboard (PR-AUC, random split used for fast benchmarking; label this caveat): LightGBM 0.889 (~3 min CPU), XGBoost 0.877, CatBoost 0.874, FT-Transformer 0.821 (~23 min GPU), Random Forest 0.794, NODE 0.772, TabNet 0.523. A callout: "On a time-based holdout the champion scores 0.539. Random splits leak temporal patterns; the lower number is the honest one."

SHAP top drivers (mean |SHAP|): auto_renew_rate 0.593, cancel_rate 0.499, plan_list_price_max 0.329, log_last_date 0.281, membership_expire_date_max 0.247. Map each to an intervention: renewal reminder, payment and cancellation-friction fix, price-tier offer, re-engagement campaign, pre-expiry outreach.

Default assumptions: value per saved subscriber V = $120, contact cost C = $5, save rate S = 20%.

## Pages

1. **Overview (executive)**
   - Hero KPI cards, scaled to a user-adjustable subscriber base (default 1,000,000; the scale factor is base ÷ 193,205):
     - Revenue at risk per monthly cycle = 2,402 × V × scale (≈ $1.49M per 1M subscribers)
     - Net value of the ROI-optimal policy per cycle (≈ $91K per 1M) and annualized ×12 (≈ $1.1M per 1M)
     - Return on retention spend = gross saved ÷ spend (≈ 3.4×)
     - Break-even save rate = (K × C) ÷ (TP × V) (≈ 5.9%, vs. 20% assumed)
   - An animated 3-step flow diagram: Score → Policy → Dollars.
   - A "Model vs. random targeting" comparison for the same 1,478 contacts: model +$17,666 vs. random ≈ −$6,950.

2. **Policy Simulator (the star page)**
   - Sliders: contact cost ($0.25–$20), save rate (2–50%), value per save ($20–$500), subscriber base (100K–20M).
   - Main chart: net value vs. K (0–20,000) for the model (teal) and random targeting (gray), with the optimal K marked and annotated.
   - Policy comparison table: Threshold 0.68 (K=1,478), Top-5K, Top-10K, and **Tiered**: top 1,478 get the incentive (C, S), ranks 1,479–10,000 get a cheap nudge with its own cost and save rate sliders (default $0.50, 12%). Show contacts, expected saves, spend, net value and ROI multiple, and highlight the winner.
   - A preset toggle: "Paid incentive ($5 / 20%)" vs. "Low-cost outreach ($0.50 / 12% / $80)". Show that the winning policy flips between them, with a callout explaining why.
   - Heatmap: net value of the best policy across save rate (x) × contact cost (y), with the break-even contour.
   - Break-even panel: minimum save rate and maximum contact cost at which the policy stays profitable.

3. **Model Evidence**
   - Holdout metric cards; precision@K and recall@K line chart; the leaderboard table with a training-time column and a "champion" badge; SHAP horizontal bar chart with the intervention mapping; a "Why trees beat the transformer here" explainer card.

4. **Score Customers** (calls the API; demo fallback)
   - Single customer: form fields auto_renew_rate (0–1), cancel_rate (0–1), plan_list_price_max, actual_paid_mean, txn_cnt, log_active_days, total_secs_mean_per_active_day, listen_full_share (0–1). POST `{VITE_API_URL}/predict` with body `{"record": {...fields}, "policy": "threshold"}`. Missing features are allowed (the API fills them with NaN). Response: `{churn_probability, churn_label, policy_used, threshold_used, metadata}`. Show a risk gauge, a risk tier (Low / Medium / High / Critical), and the recommended intervention based on the highest-risk inputs.
   - Batch: CSV upload. POST multipart to `{VITE_API_URL}/predict_batch` with policy `top_k` and a K input. Response: `{n, policy_used, items:[{churn_probability, churn_label, rank, threshold_used, policy_used}]}`. Show a sortable ranked table, a risk distribution histogram and "Export targeted list (CSV)".
   - `GET /health` drives a green or amber API-status dot in the header.

5. **MLOps & Monitoring**
   - Pipeline diagram: 31 GB raw CSV → DuckDB streaming → Parquet ZSTD (10 GB, 3.4×) → two-stage aggregation → 38-feature model table → 12-model benchmark → LightGBM champion → MLflow registry → FastAPI → SageMaker training job + Model Registry.
   - Monitoring plan card (label it "Plan, not yet live"): PSI > 0.20 on top-10 SHAP features = alert; null rate > 2× baseline; volume ±20%; ~30-day label delay; retrain if P@K < 1.5× base rate for 2 months or PSI > 0.25 on 3+ features; quarterly retrain; rollback to the previous registry version.

6. **Experiment Plan**
   - Randomized 50/50 treatment vs. control within the targeted group, measured after the label window. Include a small sample-size calculator (baseline churn among targeted, minimum detectable effect, power 80%, alpha 0.05, giving n per arm). Explain uplift modeling as the next step: target persuadables, not just high-risk users.

## Polish requirements
- Currency is formatted compactly ($1.49M, $91.4K). Numbers count up on load.
- Tooltips on every metric explain it in plain business English.
- An "Assumptions" drawer, reachable from any page, lists every assumption with its source.
- Header: product name, Demo/Live badge, API status dot, GitHub link (https://github.com/<my-username>/ai-customer-retention-mlops; leave a placeholder).
- Loading skeletons, empty states, and friendly error toasts.
- Accessibility: keyboard navigable, sufficient contrast, aria labels on charts.
