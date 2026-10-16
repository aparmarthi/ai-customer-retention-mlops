# Customer Deployment Playbook

How to take the churn decision service from this repo into a customer's environment: what data it needs, how it plugs into their systems, how it's secured, and how a 90-day pilot proves (or disproves) the value.

Related: [PRD](PRD.md) · [decision log](decisions.md) · [Cloud Run deploy guide](deploy-cloud-run.md) · [evals](../evals/README.md)

---

## 1. What gets deployed

| Component | What it does | Where |
|---|---|---|
| Scoring API (FastAPI) | Score, policy `action`, top-5 attributions, guardrails, drift | One container; Cloud Run here, any container host in the customer's cloud |
| Champion bundle | `model.pkl`, `feature_list.json`, `threshold.json`, `deployment_policy_v2.json` | Baked into the image; versioned in git |
| Decision UI (optional) | Policy simulator, single and batch scoring, executive summary | Static React build; any CDN or internal host |
| Eval harness | Parity, contract, latency, economics | Runs in CI and against each environment |

**Ownership.** Customer data engineering owns the feature extract. The CRM admin owns write-back. The retention lead owns offers per tier. The delivery team owns the model, the API and the evals.

## 2. Data contract

**Grain:** one row per active subscriber per scoring cycle, built from data available at scoring time only. **Format:** JSON records or CSV with a header. **Key:** the API takes no subscriber ID; batch results come back in input order, so the caller keeps the join key.

The 38 features, also published at `GET /`:

| Group | Features | Notes |
|---|---|---|
| Profile (5) | `city`, `bd`, `gender`, `registered_via`, `registration_init_time` | `gender` is `male` / `female` / unknown; `bd` is age |
| Billing (15) | `txn_cnt`, `cancel_cnt`, `cancel_rate`, `auto_renew_cnt`, `auto_renew_rate`, `plan_list_price_{mean,max,min}`, `actual_paid_{mean,max,min}`, `payment_method_nunique`, `txn_first_date`, `txn_tenure_days_approx`, `membership_expire_date_max` | Rates in [0, 1] |
| Engagement (18) | `log_row_cnt`, `log_first_date`, `log_last_date`, `log_active_days`, `log_span_days_approx`, `total_secs_{sum,mean_per_active_day,max_per_day}`, `num_unq_{sum,mean_per_active_day,max_per_day}`, `num_{25,50,75,985,100}_sum`, `listen_full_share`, `listens_per_unique_track` | Aggregated over the subscriber's listening history |

Dates are integers in `YYYYMMDD` form (e.g. `20170315`). Unknown columns are dropped; missing ones become NaN.

**The rule that matters most:** the model reads missing activity as churn. An empty record scores 0.99. A renamed column upstream silently turns healthy subscribers into `contact_now`. So:

- `/predict` returns `n_missing_features` and a warning when more than half are missing; batches return `n_mostly_missing`.
- **Go-live gate:** `n_mostly_missing` is 0 on a full production extract, and the score parity check below passes.

**Mapping a non-KKBox customer.** The features are generic subscription signals: billing history, auto-renew, plan price, expiry, usage intensity. Map the customer's equivalents field by field, then **retrain on their history** before trusting the scores. This champion learned KKBox behaviour from 2015–2017.

## 3. Integration patterns

| Pattern | Flow | Use when |
|---|---|---|
| **Cycle batch** (default) | Warehouse extract → `/predict_batch` in chunks of ≤ 50,000 → write `score`, `action` and `rank` back to the CRM (a field plus campaign membership) | Monthly or weekly retention campaigns |
| **Real-time lookup** | CRM or agent console → `/predict` on case open → show score, action and top-5 attributions | Save desk, care conversations |
| **Fixed call list** | `/predict_batch?policy=top_k&k=<capacity>` → ranked list | Capacity is fixed and the offer is cheap ([D4](decisions.md#d4-roi-threshold-as-the-default-top-k-only-with-a-cheap-channel)) |

Routing on `action`: `contact_now` → incentive journey (personal outreach, $5 offer); `nudge` → low-cost email or in-app journey; `monitor` → no contact, re-score next cycle. Every routed subscriber also gets a randomized holdout flag (§5).

**Sizing.** 1,000 records score in about 27 ms in-process. Batches over 50,000 records return 413, so the caller chunks. Interactive `/predict` with attributions runs at p50 51 ms in-process; pass `"explain": false` when only the score is needed (3 ms).

## 4. Security and privacy

| Control | How |
|---|---|
| Authentication | Set `API_KEY` from the customer's secret manager. POST endpoints then require `X-API-Key` (401 otherwise). Callers are servers, never browsers. For stronger isolation, use Cloud Run IAM invoker auth or internal-only ingress |
| CORS | Set `CORS_ALLOW_ORIGINS` to the UI's origin; the public demo allows `*` |
| Deploy credentials | Workload Identity Federation: no long-lived cloud keys in CI |
| Data minimization | No names, emails or IDs needed. Logs carry counts, scores, PSI and latency, never feature values |
| Personal attributes | `bd` (age), `gender` and `city` are model inputs. Before go-live, compare contact rates by age band and gender; drop or constrain the feature if the disparity isn't justified by churn rates |
| Change control | Cutoffs and costs live in a versioned policy artifact; a change is a reviewed PR that must pass the evals |

## 5. 30/60/90 pilot plan

The model predicts who will churn, not who will respond. The pilot measures the response.

**Days 0–30: shadow and data contract.**
- Deploy to the customer's environment, wire the extract, and score a full cycle with **no contact**.
- Run `python -m evals.run --api-url <their URL>`: contract and latency must pass.
- Data-contract report: `n_mostly_missing`, null rate per feature, and the tier mix vs validation (`score_drift`).
- When labels land about 30 days later, check precision of `contact_now` against the 70.6% seen on validation.
- **Exit:** `n_mostly_missing` = 0, PSI < 0.25, `contact_now` precision ≥ 60%.

**Days 31–60: randomized holdout.**
- Within the top 10,000, stratified by tier: half get their tier's offer, half get nothing.
- Power: from 18% baseline churn in the top 10K, detecting 18% → 15% at α = 0.05 and 80% power needs **2,402 per arm** (`#/experiment` in the UI).
- Primary metric: churn rate, treatment vs holdout, pooled across tiers. Per-tier save rates are secondary, with wider intervals.
- Guardrails: complaint and unsubscribe rate, cost per incremental save.

**Days 61–90: decide and scale.**
- Replace the assumed save rates (20% incentive, 12% nudge) with the measured ones and re-run the simulator.
- **Go:** significant churn reduction (p < 0.05), positive net value at the *measured* save rates, and no guardrail breach. Then scale to the full tiered policy, keep a 5–10% standing holdout, and schedule retraining.
- **No-go or adjust:** if the incentive doesn't move churn, keep only `nudge` or test a different offer. If the lift sits only in some segments, that's the case for uplift modelling.

## 6. Runbook

| Situation | Action |
|---|---|
| Health check | `GET /health`: `model_loaded`, artifact paths, `policy_version` |
| `score_drift.status` = `moderate_shift` | Investigate before the campaign: compare the extract's null rates and volume with the last cycle |
| `score_drift.status` = `significant_shift` or `n_mostly_missing` > 0 | Hold the campaign; fix the extract or escalate for retraining |
| Bad deploy | `gcloud run services update-traffic churn-api --region us-central1 --to-revisions=<previous>=100` |
| Change a cutoff or cost | New `deployment_policy_vN.json`, point `POLICY_PATH` in `src/serving/api.py` at it, update tests, pass the evals, deploy |
| Retraining | Triggers in README §11.3; a new champion must beat the old one on PR-AUC on the latest window and pass parity on its own reference records |
