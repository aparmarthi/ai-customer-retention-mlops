# bolt.new Prompt — Churn Decision Intelligence (v3, live API)

> Paste everything below the line into bolt.new as a single prompt.
> Every number, endpoint, field name and URL was verified against the repo on 2026-10-08
> (`src/serving/api.py`, `artifacts/champion/*.json`, `leaderboard.md`) and smoke-tested against the
> live Cloud Run API: https://churn-api-995576020018.us-central1.run.app (CORS open, ~200 ms warm,
> scale-to-zero so the first call after idle takes a few seconds).
> v3 changes vs v2: live default URL; `/predict_batch` JSON + server-side top-K now work (API fix).

---

Build a production-quality, single-page React + Vite + TypeScript + Tailwind app called **"Churn Decision Intelligence"**. It's the executive and operator front end for an existing FastAPI churn-prediction service, trained on the KKBox subscription-music dataset (Kaggle WSDM 2018). It must look like a Series-B SaaS product, not a data-science demo. Use Recharts for charts, lucide-react for icons, and no other UI framework. Dark theme by default, in a slate/zinc palette, with a single accent: emerald for "value" and rose for "risk". Use Inter, and tabular-nums for every number.

## 0. Global architecture

- `src/lib/api.ts` reads `import.meta.env.VITE_API_URL`, defaulting to `https://churn-api-995576020018.us-central1.run.app`. If `GET {base}/health` fails or times out, switch to **Demo Mode**. The service scales to zero, so use a 12 s timeout on the first /health call and show a "Waking API (cold start)…" state with a spinner. Use 6 s after that.
- The top bar shows a status pill:
  - green "Live API · model_loaded · 38 features · default threshold 0.68" when /health succeeds;
  - amber "Demo Mode — embedded validation data" otherwise.
- Add a Settings drawer (gear icon) with a text field to override the API base URL at runtime. Store it in localStorage and re-ping /health on save.
- All data shown on Overview, Simulator, Model Evidence and MLOps pages is **static validation data embedded in `src/data/*.ts`** (below). Only the "Score Customers" page calls the API.
- Every page has a small footer chip, "Simulated on Feb-2017 validation window (193,205 subscribers, 1 cycle = 1 month). Not realized revenue." Business-value numbers must never appear without that caveat nearby.
- The header carries the links: GitHub `https://github.com/aparmarthi/ai-customer-retention-mlops` · Original Streamlit app `https://amey-churn-predictor.streamlit.app/` · LinkedIn `https://www.linkedin.com/in/ameyparmarthi/` · Dataset `https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge`.
- Use a left sidebar nav with 6 pages, collapsible on mobile, and URL hash routing (`#/overview` etc).

## 1. Exact API contract (do not invent fields)

```
GET  /            -> { service: "Churn Decision Intelligence API", version: "1.0.0", docs: "/docs", health: "/health", endpoints: ["/predict","/predict_batch"] }
GET  /health      -> { status: "ok", model_loaded: boolean, artifacts: {model, threshold, feature_list, categorical_cols}, n_features: number, default_threshold: number }

POST /predict     (application/json)
  body:  { record: Record<string, number|string>, policy?: "threshold", threshold?: number }
  200:   { churn_probability: number, churn_label: 0|1, policy_used: "threshold", threshold_used: number, metadata: { note: string } }
  400 if policy !== "threshold". 503 { detail: "Model not loaded." } if model not ready.

POST /predict_batch   (application/json)
  body:  { records: Array<Record<string, number|string>>, policy: "threshold"|"top_k", threshold?: number, k?: number }
  200:   { n: number, policy_used: "threshold" | "top_k(<k>)", items: Array<{ churn_probability: number, churn_label: 0|1, rank: number|null, threshold_used: number|null, policy_used: string, metadata: {} }> }
         items are in INPUT order; with top_k, rank 1 = highest risk and churn_label=1 for rank <= k.
         threshold_used is 0.2058 when policy=top_k and k=10000 (the equivalent cutoff), else null for top_k.
  400 if policy="top_k" without k, or unknown policy. 422 { detail } on invalid body.

POST /predict_batch?policy=top_k&k=10000   (multipart/form-data, field "file" = CSV with header row)
  Same response. Query params policy/threshold/k are optional (default threshold policy @ 0.68).
```

**Important constraints (verified):**
- For an uploaded CSV, send it as-is via multipart with `?policy=top_k&k=<K>`. Don't re-parse it into JSON (that keeps big files fast). For in-app rows such as the samples, use the JSON body.
- **Top-K ranking is done server-side** (`rank` in the response). Items come back in input order, so join to input rows by index.
- Missing feature columns are treated as NaN by the server; extra columns are ignored. `gender` is categorical ("male" | "female" | "unknown"). All dates are **YYYYMMDD integers** (e.g., 20170228), not ISO strings.
- Surface API errors as a toast with the `detail` string. Show request latency (ms) measured client-side on every call.

The 38 model features, in order (put them in `src/data/features.ts`, grouped for the form UI):
- **Profile:** city, bd (age), gender, registered_via, registration_init_time
- **Billing:** txn_cnt, cancel_cnt, cancel_rate, auto_renew_cnt, auto_renew_rate, plan_list_price_mean, plan_list_price_max, plan_list_price_min, actual_paid_mean, actual_paid_max, actual_paid_min, payment_method_nunique, txn_first_date, txn_tenure_days_approx, membership_expire_date_max
- **Engagement:** log_row_cnt, log_first_date, log_last_date, log_active_days, log_span_days_approx, total_secs_sum, total_secs_mean_per_active_day, total_secs_max_per_day, num_unq_sum, num_unq_mean_per_active_day, num_unq_max_per_day, num_25_sum, num_50_sum, num_75_sum, num_985_sum, num_100_sum, listen_full_share, listens_per_unique_track

## 2. Embedded data (`src/data/`)

**validation.ts**
```ts
export const VALID = { n: 193205, churners: 2402, churnRate: 0.01243, window: "Feb 2017 (1 month)", champion: "LightGBM (FLAML-tuned)", rocAuc: 0.9660, prAuc: 0.5392, f1At05: 0.3678 };
// Operating curve: contacts K -> true churners caught (TP), from champion scores ranked desc.
export const CURVE = [
  { k: 0, tp: 0 }, { k: 1262, tp: 992, threshold: 0.71 }, { k: 1478, tp: 1044, threshold: 0.68 },
  { k: 5000, tp: 1345 }, { k: 10000, tp: 1801, threshold: 0.2058 }, { k: 20000, tp: 2179 }, { k: 193205, tp: 2402 },
];
export const POLICY = { name: "hybrid_topk_primary_roi_threshold_fallback", version: "v1",
  primary: { type: "top_k", k: 10000, note: "Capacity-driven: retention team works the top 10K each cycle" },
  fallback: { type: "threshold", threshold: 0.68, note: "ROI-max cutoff when capacity is unconstrained" } };
export const DEFAULTS = { valuePerSave: 120, costPerContact: 5, saveRate: 0.20 };
```

**leaderboard.ts.** This is an earlier random 80/20 benchmark at ~6.2% prevalence, so label it "Model-family benchmark (random split)" and keep it separate from the time-based validation numbers above:
```ts
export const LEADERBOARD = [
  { model: "LightGBM", prAuc: 0.8887, rocAuc: 0.9894, f1: 0.7845, precision: 0.6805, recall: 0.9260, train: "~3 min CPU", champion: true },
  { model: "XGBoost", prAuc: 0.8771, rocAuc: 0.9875, train: "~2 min CPU" },
  { model: "CatBoost", prAuc: 0.8737, rocAuc: 0.9865, train: "3.7 min GPU" },
  { model: "FT-Transformer", prAuc: 0.8214, rocAuc: 0.9824, train: "23 min GPU/AMP" },
  { model: "Random Forest", prAuc: 0.7935, rocAuc: 0.9782, train: "5 min CPU" },
  { model: "NODE", prAuc: 0.7719, rocAuc: 0.9737, train: "11 min GPU" },
  { model: "TabNet", prAuc: 0.5233, rocAuc: 0.9085, train: "32 min GPU (5 failed runs)" },
];
// Ensemble of top models tied LightGBM, so it was rejected (no lift for added complexity). Lift: 14.3x at PR-optimal; champion P@5K lift 21.7x, P@10K 14.5x.
export const SAGEMAKER = { rocAuc: 0.9484, prAuc: 0.4707, f1: 0.4658, instance: "ml.m5.large", registry: "SageMaker Model Registry" };
```

**samples.ts.** Five real demo customers. Scores are from the champion model via a local run of the API, so use them in Demo Mode:
```ts
export const SAMPLES = [
  { label: "Loyal auto-renewer, 2-yr heavy listener", expected: 0.786, record: {"city":22,"bd":25,"gender":"female","registered_via":9,"registration_init_time":20111121,"txn_cnt":20,"cancel_cnt":1,"cancel_rate":0.05,"auto_renew_cnt":20,"auto_renew_rate":1.0,"plan_list_price_mean":141.55,"plan_list_price_max":149,"plan_list_price_min":0,"actual_paid_mean":149,"actual_paid_max":149,"actual_paid_min":149,"payment_method_nunique":1,"txn_first_date":20150131,"txn_tenure_days_approx":20083,"membership_expire_date_max":20170307,"log_row_cnt":657,"log_first_date":20150101,"log_last_date":20170228,"log_active_days":657,"log_span_days_approx":20127,"total_secs_sum":5842310,"total_secs_mean_per_active_day":8893.2,"total_secs_max_per_day":47041.8,"num_unq_sum":17687,"num_unq_mean_per_active_day":26.9,"num_unq_max_per_day":162,"num_25_sum":1818,"num_50_sum":621,"num_75_sum":328,"num_985_sum":372,"num_100_sum":18475,"listen_full_share":0.855,"listens_per_unique_track":1.222} },
  { label: "New single-txn, no auto-renew, expired", expected: 0.978, record: {"city":5,"bd":21,"gender":"female","registered_via":3,"registration_init_time":20160611,"txn_cnt":1,"cancel_cnt":0,"cancel_rate":0.0,"auto_renew_cnt":0,"auto_renew_rate":0.0,"plan_list_price_mean":180,"plan_list_price_max":180,"plan_list_price_min":180,"actual_paid_mean":180,"actual_paid_max":180,"actual_paid_min":180,"payment_method_nunique":1,"txn_first_date":20170102,"txn_tenure_days_approx":0,"membership_expire_date_max":20170201,"log_row_cnt":6,"log_first_date":20160611,"log_last_date":20170102,"log_active_days":6,"log_span_days_approx":9491,"total_secs_sum":103142,"total_secs_mean_per_active_day":17190.4,"total_secs_max_per_day":32982.2,"num_unq_sum":342,"num_unq_mean_per_active_day":57,"num_unq_max_per_day":120,"num_25_sum":439,"num_50_sum":98,"num_75_sum":30,"num_985_sum":30,"num_100_sum":379,"listen_full_share":0.388,"listens_per_unique_track":2.854} },
  { label: "Mid-tenure, low price plan, prior cancel", expected: 0.964, record: {"city":1,"bd":0,"gender":"unknown","registered_via":7,"registration_init_time":20130611,"txn_cnt":14,"cancel_cnt":1,"cancel_rate":0.071,"auto_renew_cnt":14,"auto_renew_rate":1.0,"plan_list_price_mean":99,"plan_list_price_max":99,"plan_list_price_min":99,"actual_paid_mean":99,"actual_paid_max":99,"actual_paid_min":99,"payment_method_nunique":1,"txn_first_date":20160215,"txn_tenure_days_approx":10000,"membership_expire_date_max":20170314,"log_row_cnt":242,"log_first_date":20160215,"log_last_date":20170215,"log_active_days":242,"log_span_days_approx":10000,"total_secs_sum":1460238,"total_secs_mean_per_active_day":6034,"total_secs_max_per_day":30731.3,"num_unq_sum":4740,"num_unq_mean_per_active_day":19.6,"num_unq_max_per_day":118,"num_25_sum":1282,"num_50_sum":414,"num_75_sum":283,"num_985_sum":293,"num_100_sum":5527,"listen_full_share":0.709,"listens_per_unique_track":1.645} },
  { label: "Steady auto-renewer, active through Feb", expected: 0.004, record: {"city":13,"bd":33,"gender":"male","registered_via":9,"registration_init_time":20150920,"txn_cnt":5,"cancel_cnt":0,"cancel_rate":0.0,"auto_renew_cnt":5,"auto_renew_rate":1.0,"plan_list_price_mean":149,"plan_list_price_max":149,"plan_list_price_min":149,"actual_paid_mean":149,"actual_paid_max":149,"actual_paid_min":149,"payment_method_nunique":1,"txn_first_date":20160801,"txn_tenure_days_approx":5000,"membership_expire_date_max":20170301,"log_row_cnt":180,"log_first_date":20160801,"log_last_date":20170201,"log_active_days":150,"log_span_days_approx":5500,"total_secs_sum":900000,"total_secs_mean_per_active_day":6000,"total_secs_max_per_day":25000,"num_unq_sum":3000,"num_unq_mean_per_active_day":20,"num_unq_max_per_day":80,"num_25_sum":800,"num_50_sum":300,"num_75_sum":200,"num_985_sum":250,"num_100_sum":3500,"listen_full_share":0.72,"listens_per_unique_track":1.5} },
  { label: "Brand-new, already cancelled, 3 active days", expected: 0.957, record: {"city":6,"bd":45,"gender":"male","registered_via":4,"registration_init_time":20170101,"txn_cnt":1,"cancel_cnt":1,"cancel_rate":1.0,"auto_renew_cnt":0,"auto_renew_rate":0.0,"plan_list_price_mean":100,"plan_list_price_max":100,"plan_list_price_min":100,"actual_paid_mean":100,"actual_paid_max":100,"actual_paid_min":100,"payment_method_nunique":1,"txn_first_date":20170115,"txn_tenure_days_approx":0,"membership_expire_date_max":20170215,"log_row_cnt":3,"log_first_date":20170115,"log_last_date":20170118,"log_active_days":3,"log_span_days_approx":3,"total_secs_sum":5000,"total_secs_mean_per_active_day":1667,"total_secs_max_per_day":3000,"num_unq_sum":50,"num_unq_mean_per_active_day":17,"num_unq_max_per_day":25,"num_25_sum":20,"num_50_sum":5,"num_75_sum":3,"num_985_sum":2,"num_100_sum":30,"listen_full_share":0.72,"listens_per_unique_track":1.5} },
];
```

**Images.** These are public and load directly:
- Architecture: `https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main/docs/architecture.png`
- SHAP summary: `https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main/reports/shap_summary.png`
- Threshold vs precision/recall: `https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main/reports/threshold_vs_precision_recall.png`
- Threshold vs ROI: `https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main/reports/threshold_vs_roi.png`

## 3. Core math (`src/lib/economics.ts`, pure functions + unit-testable)

- `tpAtK(k)`: linear interpolation over `CURVE`. **Precision is always derived (`tpAtK(k)/k`), never a user input.** This is the credibility fix vs the old Streamlit simulator.
- `net(k, V, C, S) = tpAtK(k) * S * V - k * C` (per cycle, at validation scale).
- `randomNet(k, V, C, S) = k * churnRate * S * V - k * C`.
- `scale(x, subs) = x * subs / 193205`. The default "subscriber base" is 1,000,000 and is editable, from 100K to 50M.
- `breakEvenSaveRate(k, V, C) = (k*C) / (tpAtK(k)*V)`. `breakEvenCost(k, V, S) = tpAtK(k)*S*V / k`.
- `roiMultiple = (tpAtK(k)*S*V) / (k*C)`.
- Sanity values at defaults (V=$120, C=$5, S=20%), which must reproduce exactly:
  - K=1,478 → TP 1,044, precision 70.6%, recall 43.5%, value saved $25,056, spend $7,390, **net $17,666/cycle at 193K subs**, ROI 3.4×, break-even save rate 5.9%, break-even contact cost $16.95.
  - Random targeting of the same 1,478 → net **−$6,950**.
  - Scaled to 1M subs: net ≈ $91.4K/cycle ≈ **$1.10M/yr**; revenue at risk (churners × V) ≈ **$1.49M/cycle**.

## 4. Pages

### 4.1 Overview (`#/overview`), the executive one-pager
- Hero: "Find the 0.8% of subscribers who drive 43% of churn, before they leave." Subhead: "LightGBM churn model + capacity-aware decision policy on 31 GB of KKBox subscription and listening logs."
- Four KPI cards, scaled to the subscriber-base input in the header (default 1M):
  1. Annualized net retention value, $1.10M/yr
  2. Return on retention spend, 3.4×
  3. Churners caught by contacting 0.76% of the base, 43.5%
  4. Lift vs random at top 10K, 14.5×
- Each card has a tooltip with the formula and "simulated, validation window" caveat.
- "Model vs random" bar: +$17.7K vs −$7.0K at validation scale, with a one-line takeaway: "Same budget, same 1,478 contacts — the model is the difference between profit and loss."
- Pipeline strip: 31 GB raw CSV → DuckDB feature engineering (38 features) → 12-model benchmark + FLAML tuning → MLflow tracking → FastAPI (/predict, /predict_batch) → Docker → Cloud Run → Streamlit/this UI. Clicking it opens the architecture image in a lightbox.
- "What I'd tell the VP of Retention" box, with three bullets:
  - Start with top-1,478 at the 0.68 threshold for max ROI.
  - Scale to top-10K only when the team has capacity and a cheaper nudge channel.
  - Re-validate the save-rate assumption with a holdout A/B before trusting the dollar figures.

### 4.2 Policy Simulator (`#/simulator`), the centerpiece
- Inputs on the left:
  - K contacts: slider from 100 to 50,000, log scale, with snap buttons for 1,262 / 1,478 / 5,000 / 10,000 / 20,000.
  - Value per save V ($10–$500, default $120).
  - Cost per contact C ($0.10–$50, default $5).
  - Save rate S (1–60%, default 20%).
  - Subscriber base (default 1M).
- Show derived precision, recall and equivalent threshold (if K matches a known point) as **read-only** chips with a lock icon and the tooltip "Derived from the model's ranking — not adjustable."
- Main chart: net value vs K for Model (emerald) and Random (rose dashed), with vertical markers for "ROI-max (K=1,478)" and "Capacity policy (K=10,000)". There's a draggable/hover crosshair.
- Results panel: TP caught, precision, recall, value saved, spend, net (at validation scale and scaled base), ROI multiple, break-even save rate, break-even contact cost.
  - If net < 0, turn the panel rose and say why: "At $C/contact, each save must be worth > $X."
- Sensitivity tornado: ±50% on V, C and S around the current point, showing the effect on net. This makes the save-rate assumption the visible #1 risk.
- **Tiered policy toggle (illustrative):**
  - Tier 1: top 1,478 get a $5 incentive at a 20% save rate.
  - Tier 2: ranks 1,479–10,000 get a $0.50 email/push nudge at a 12% save rate.
  - Show combined net (~$24.3K/cycle at validation scale, +37% vs single-tier). Badge it "Illustrative — tier-2 save rate is an assumption."
- "Copy scenario" button copies a short text summary to the clipboard for a Slack/exec update.

### 4.3 Model Evidence (`#/evidence`)
- Validation card: ROC-AUC 0.966, PR-AUC 0.539 (base rate 1.24% → PR-AUC is **43× better than random**), with F1@0.5 0.368 shown and explained ("0.5 is the wrong cutoff for a 1.2%-prevalence problem — that's why the policy picks K, not 0.5").
- Leaderboard table from `LEADERBOARD`, sortable. Champion row highlighted. Add a horizontal bar of PR-AUC by model, and a "Why LightGBM" callout covering:
  - best PR-AUC;
  - roughly 3-minute CPU training vs 23–32 minutes on GPU for the deep tabular models;
  - native categorical handling;
  - cheap serving (an 18.7 MB artifact on 1 vCPU / 1 GiB).
  - Also: "Ensemble tied — rejected for complexity."
- Banner explaining the split difference: "Benchmark used a random split (6.2% prevalence); champion was re-validated on a time-based Feb-2017 window (1.24%). Numbers are not directly comparable — the time-based ones are what we'd ship on."
- SHAP image + threshold vs precision/recall image + threshold vs ROI image, in cards with captions. Use a lightbox on click.
- SageMaker parity card: the same pipeline retrained on SageMaker (ml.m5.large) gives ROC 0.948, PR-AUC 0.471, F1 0.466, registered in SageMaker Model Registry. Caption: "portability check, not the champion."

### 4.4 Score Customers (`#/score`), live API
Use three tabs.
- **Single customer:**
  - A sample picker (the 5 SAMPLES as cards with label + expected risk) and a JSON editor. Also offer a grouped form for the 38 features (Profile / Billing / Engagement accordions, with date inputs that convert to YYYYMMDD ints and a gender select).
  - Threshold input (default 0.68).
  - Call `POST /predict`.
  - Result: a large probability gauge, risk band (≥0.68 "Contact now", 0.2058–0.68 "Capacity tier", <0.2058 "Monitor"), the label, threshold used and latency.
  - Live mode: show a small "matches validation run ✓" chip when the returned score is within 0.001 of the sample's `expected` value. In Demo Mode, return `expected` with a "demo" badge. Custom edits show "Connect API to score custom records."
- **Batch (CSV):**
  - Drag-and-drop CSV, with "Download template CSV" (38 headers + the 5 samples) and "Use 5 samples" buttons.
  - POST it as multipart `file` to `/predict_batch?policy=top_k&k=<K>`, with K input defaulting to 10,000 and capped at n. "Use 5 samples" sends JSON `{records, policy:"top_k", k}` instead.
  - Show a table (rank, probability, label @0.68, in-top-K flag, plus the first 4 input columns), a probability histogram with the threshold and K cutoffs, and a summary (n scored, # ≥0.68, # in top-K, expected saves = Σ probability × S over the top-K, expected net using V/C/S from the Simulator, shared state).
  - Export the results to CSV.
- **API console:**
  - Shows the live base URL, the `/health` JSON and the `/` JSON, with copyable curl examples for both endpoints built from the current base URL.
  - Link to `{base}/docs`.

### 4.5 MLOps & Monitoring (`#/mlops`)
- Deployment card: FastAPI on Cloud Run (us-central1, 1 vCPU / 1 GiB, min 0 / max 3 instances, scale-to-zero). CI is GitHub Actions: pytest → build image → deploy. There's a Docker image, `/health` checks, request-latency logging middleware, and MLflow experiment tracking + model registry.
- Monitoring plan table (signal → threshold → action):

  | Signal | Threshold | Action |
  |---|---|---|
  | Feature drift (PSI) | warn > 0.20, alert > 0.25 | |
  | Null rate | > 2× baseline | |
  | Scoring volume | ±20% vs trailing | |
  | Label feedback | 30-day delay before precision can be measured | |
  | Precision@K | < 1.5× base rate for 2 consecutive months | Retrain |
  | Retrain cadence | Quarterly | Registry rollback to previous champion on regression |

- Make it a mock "Monitoring" panel with sparkline placeholders labeled "Mock — wiring planned" (be honest; there's no live telemetry).
- Deployment policy JSON viewer showing `POLICY` with an explanation: primary top-K = capacity, fallback threshold = ROI.

### 4.6 Experiment Plan (`#/experiment`), the AI-PM page
- "From simulated ROI to measured ROI" plan:
  - randomized holdout, 50/50 within the top 10K;
  - primary metric: 30-day retention rate delta;
  - guardrails: complaint/unsubscribe rate and incentive cost per retained user;
  - duration: 2 billing cycles.
- Sample-size calculator: inputs are baseline churn within the top-K (default 18% = 1,801/10,000), minimum detectable effect (default 3 pts absolute), α=0.05 and power=0.8. Output n per arm, with the two-proportion z-test formula shown.
- Decision log (static cards):
  - LightGBM over the ensemble (tie, simpler);
  - K-based over threshold-based primary policy (ops capacity is the real constraint);
  - PR-AUC as the selection metric (1.24% prevalence makes ROC flattering);
  - time-based validation (no leakage from future months);
  - CPU serving (no GPU needed at 18.7 MB).
- Risks and next steps:
  - the save rate is assumed, not measured;
  - uplift modeling to target persuadables rather than just high-risk users;
  - cost-sensitive tiers per channel.

## 5. Quality bar
- Fully responsive. Skeleton loaders on API calls. Empty and error states on every API-backed panel.
- Accessible: keyboard nav, focus rings, aria-labels on icon buttons, chart colors that are colorblind-safe (emerald/rose plus dash patterns).
- Format numbers with Intl: `$1.10M`, `3.4×`, `70.6%`, `1,478`.
- No lorem ipsum, no invented metrics. If a value isn't in this prompt, don't display it.
- Include a README covering `VITE_API_URL` (default is the live Cloud Run URL), demo-mode fallback, the cold-start note, and both `/predict_batch` input modes.
