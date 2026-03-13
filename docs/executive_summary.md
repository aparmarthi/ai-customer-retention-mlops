# Executive Summary — KKBox Churn Prediction Platform

## TL;DR (One-Slide Pitch)

**Problem:** KKBox loses subscribers every month with no systematic way to predict who will leave or prioritize retention spend.

**Solution:** ML-driven churn prediction platform that ranks every subscriber by exit risk, targets the top 10,000 highest-risk users monthly, and projects $12,200–$17,666 net ROI per scoring cycle under conservative assumptions.

**Key result:** The model concentrates 75% of actual churners into the top 10,000 contacts — a **3x lift** over random targeting — while keeping outreach volume predictable and budget-bounded.

---

## 1. The Problem

Subscription businesses face a universal challenge: churn is expensive, but retention budgets are finite. Without a targeting system, companies either:

- **Blast everyone** — wasting budget on users who were never going to leave
- **Target no one** — losing subscribers who could have been saved with timely intervention
- **Rely on rules** (e.g., "inactive for 30 days") — missing nuanced behavioral patterns

KKBox, a music streaming service with millions of subscribers, provided real transaction and behavioral data spanning 2+ years. The dataset exhibits severe class imbalance (~1.2% churn rate in the evaluation window), making naive accuracy metrics misleading and precision-focused evaluation essential.

---

## 2. The Solution

An end-to-end ML platform — not just a model — that covers the full path from raw data to business decision:

| Layer | What It Does |
|---|---|
| **Data pipeline** | 7 ETL scripts process ~28 GB of raw logs into ML-ready features |
| **Model training** | 12 architectures evaluated; FLAML AutoML LightGBM selected as champion |
| **Decision policy** | Hybrid engine: top-K for operations, ROI-optimal threshold for cost-sensitive contexts |
| **ROI simulation** | Connects model precision to dollar outcomes under configurable assumptions |
| **Serving** | FastAPI REST API exposing both policies with Pydantic contracts |
| **Explainability** | SHAP feature contributions — every prediction is auditable |
| **Cloud validation** | AWS SageMaker training job + Model Registry — proves cloud readiness |

---

## 3. What Impact Does This Deliver?

### Quantified outcomes (per monthly scoring cycle)

| Policy | Users Targeted | Expected Saves | Estimated Net ROI |
|---|---:|---:|---:|
| **Outreach** (email/SMS, $0.50/contact) | 10,000 | ~215 | **~$12,200** |
| **ROI-optimal threshold** ($5/contact, 20% save rate) | 1,478 | ~209 | **$17,666** |

### Model performance on time-based holdout

| Metric | Value | Why It Matters |
|---|---:|---|
| PR-AUC | 0.5392 | Ranking quality under 1.2% churn — the honest metric |
| Recall @ top-10k | 75.0% | 3 out of 4 actual churners are in the contact list |
| Precision @ top-10k | 18.0% | 3x concentration over random — every $1 of outreach works 3x harder |

---

## 4. Key Tradeoffs Made

| Decision | What was chosen | What was sacrificed | Why |
|---|---|---|---|
| **PR-AUC over ROC-AUC** | Optimize for minority-class ranking | ROC-AUC looks better (0.966 vs 0.539) | ROC-AUC flatters under imbalance; PR-AUC reflects real targeting quality |
| **Time-based split over random** | Chronological holdout (Feb 2017) | Metrics drop ~40% vs. random split | Random splits leak temporal patterns; time-based simulates production |
| **Top-K over threshold (primary)** | Fixed 10,000 contacts/month | Misses some churners below the cutoff | Predictable budget, no threshold recalibration as distributions drift |
| **Single SageMaker job** | One controlled cloud validation run | No cloud-based hyperparameter tuning | Cost control; local experiments already optimized the model |
| **LightGBM over deep learning** | 3 min train time, best PR-AUC | No neural network novelty | TabNet/FT-Transformer took 10-30x longer and scored worse on every metric |

---

## 5. How Would You Roll This Out?

### Phase 1 — Shadow Mode (Weeks 1–4)

- Deploy the FastAPI service alongside the existing system (if any)
- Score all subscribers monthly; log predictions but **take no action**
- Compare model-predicted churn to actual churn after 30-day label delay
- Validate Precision@K matches offline evaluation (expect ~18% at K=10,000)
- Monitor input feature distributions for drift vs. training baseline

**Exit criteria:** Precision@K within 2pp of offline evaluation for 2 consecutive months.

### Phase 2 — Limited Pilot (Weeks 5–12)

- Target the top 5,000 (not 10,000) highest-risk users with low-cost outreach (email/SMS)
- Split into **Treatment (50%)** and **Control (50%)** — 2,500 users each
- Measure:
  - Churn rate in Treatment vs. Control (incremental lift)
  - Outreach cost vs. retained revenue (actual ROI)
  - False positive rate (user complaints from unnecessary outreach)

**Exit criteria:** Statistically significant churn reduction in Treatment group (p < 0.05); positive net ROI.

### Phase 3 — Scale (Months 4+)

- Expand to full top-10,000 targeting
- Add incentive-based intervention for highest-confidence / highest-LTV users
- Activate the ROI-optimal threshold policy for automated channels (push notifications, in-app messages)
- Implement quarterly retraining with automated Model Registry promotion
- Stand up monitoring dashboards (feature drift, Precision@K decay, ROI tracking)

---

## 6. A/B Test Design

The model predicts *who will churn* — not *who will respond to intervention*. Without a controlled experiment, any measured retention improvement could be natural fluctuation, not model-driven.

### Test structure

```
Monthly scoring cycle
    │
    ├── Score all subscribers → ranked by churn probability
    │
    ├── Select top-K (K = 5,000 for pilot, 10,000 at scale)
    │
    ├── Random assignment:
    │     ├── Treatment (50%): receive retention intervention
    │     └── Control (50%): no contact (business as usual)
    │
    └── After 30 days:
          ├── Measure churn rate in Treatment vs. Control
          ├── Compute incremental lift = churn_control − churn_treatment
          └── Compute true incremental ROI
```

### Success metric

```
Incremental ROI = (churn_rate_control − churn_rate_treatment)
                  × N_treatment
                  × average_LTV
                  − total_intervention_cost
```

### Sample size consideration

With a baseline churn rate of ~18% in the top-10k (Precision@K), detecting a 3pp absolute reduction (18% → 15%) at 80% power and α=0.05 requires ~2,800 users per arm. A pilot K of 5,000 (2,500 per arm) is close to sufficient; scaling to K=10,000 (5,000 per arm) provides comfortable power.

---

## 7. How Do You Measure Success?

### Leading indicators (available immediately)

| Metric | Target | Frequency |
|---|---|---|
| Prediction volume | 10,000 scores per cycle (stable) | Every scoring run |
| API latency (p95) | < 500ms | Continuous |
| Feature null rate | < 2x historical baseline | Every scoring run |
| Score distribution stability | PSI < 0.20 vs. training baseline | Monthly |

### Lagging indicators (available after ~30-day label delay)

| Metric | Target | Frequency |
|---|---|---|
| Precision @ top-10k | > 15% (> 2.5x base rate) | Monthly |
| Recall @ top-10k | > 65% | Monthly |
| Incremental churn reduction (A/B) | > 2pp vs. control | Monthly |
| Net ROI per cycle | > $5,000 | Monthly |

### Retraining triggers

| Trigger | Condition |
|---|---|
| Scheduled | Quarterly, or after major product/pricing changes |
| Performance-based | Precision@K drops below 1.5x base rate for 2 consecutive months |
| Drift-based | PSI > 0.25 on 3+ top SHAP features in a single month |
| Emergency | Model serving errors or complete prediction failure — rollback to previous Model Registry version |

---

## 8. What Makes This Project Different

Most ML portfolio projects stop at "I trained a model and got 0.96 AUC." This project demonstrates:

1. **Honest evaluation** — time-based split drops metrics by ~40% vs. random split; both are reported transparently
2. **Business translation** — every model metric maps to a dollar outcome via configurable ROI assumptions
3. **Operational realism** — top-K policy exists because real retention teams have fixed monthly capacity, not infinite threshold-based budgets
4. **Two-policy architecture** — not one decision boundary, but a hybrid engine with documented tradeoffs
5. **Cloud proof** — SageMaker training + Model Registry demonstrates the local pipeline works in managed infrastructure
6. **Explainability** — SHAP contributions make every prediction auditable and actionable for product teams
7. **Deployment readiness** — FastAPI service with Pydantic contracts, not a notebook with `model.predict()`
