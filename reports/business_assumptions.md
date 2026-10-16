# Business Assumptions for Churn Targeting (Capstone)

This document translates model performance into **business impact** using simple, defensible assumptions.
All values below are **assumptions** (not KKBox ground truth). The goal is to show how a churn model
supports a targeting policy and expected ROI.

---

## Model input used in this doc

From the champion evaluation:

- **Precision@K** where **K = 10,000**: **P@10000 = 0.1801**

Interpretation:
- If we target the **top 10,000** highest-risk users, about **18.0%** of them (1,801 users)
  are expected to be true churners (without intervention), on average.

---

# Core ROI Framework

Let:

- **N** = number of users targeted
- **P@K** = precision at K
- **C** = cost per targeted user ($)
- **S** = retention success rate among true churners when targeted
- **V** = value of retaining a subscriber ($)

Then:

- Expected true churners in targeted set = `N × P@K`
- Expected saves = `N × P@K × S`
- Expected benefit = `N × P@K × S × V`
- Expected cost = `N × C`
- **Expected net value = N × (P@K × S × V − C)**
- Net ROI = net value ÷ (N × C)

Break-even condition:
P@K × S × V ≥ C


Using **P@10000 = 0.1801** and **N = 10,000**:
- Expected churners in top 10k ≈ 1,801
- Expected saves ≈ 1,801 × S

---

# Headline Example (Low-Cost Outreach Policy)

**Policy:** Proactive outreach (email/SMS/in-app nudges + light support)

Chosen for simplicity and defensibility.

Assumptions:
- **N = 10,000**
- **P@10000 = 0.1801**
- **C = $0.50**
- **S = 12%**
- **V = $80**

Calculation:
- Expected saves ≈ 1,801 × 0.12 = 216.1
- Benefit ≈ 216.1 × $80 = $17,290
- Cost = 10,000 × $0.50 = $5,000
- **Net value ≈ $12,290**

### Headline:
Under a low-cost outreach policy, targeting the top 10,000 users yields approximately **$12.3K net value** per cycle on the validation window (simulated, not realised).

This provides a concrete business interpretation of model precision.

---

# ROI Model Used in the Threshold Sweep Code (Source of Truth)

The current `threshold_optimization.py` selects thresholds using a simplified ROI model:
ROI = (TP × SAVE_RATE × CHURN_COST) − (N_targeted × INTERVENE_COST)

Where:

- **CHURN_COST = $120**
- **SAVE_RATE = 20%**
- **INTERVENE_COST = $5**

This simplifies the general formula by approximating:
- `V = CHURN_COST`
- `C = INTERVENE_COST`

This ROI definition is used to determine:

- ROI-optimal decision threshold
- Ops-friendly top-10k targeting policy

Under these assumptions, top-10k targeting **loses $6,776** (1,801 × 0.20 × $120 − 10,000 × $5 = $43,224 − $50,000).
ROI peaks at threshold 0.68: 1,478 contacts, 1,044 churners reached, **$17,666 net** (net ROI 239%).
This is why the API defaults to the threshold policy, and why a 10,000-contact capacity plan needs a
cheap channel below the top 1,478. A tiered plan ($5 incentive for the top 1,478, $0.50 nudge at a
12% save rate for ranks 1,479–10,000) nets $24,306 under the same tier-1 assumptions.

---

# Scenario 1: Retention Offer (Incentive-Based)

### Story
Target high-risk users with discounts or incentives.

Typical Assumptions:
- Cost per targeted user: $5–$15
- Success rate: 10–30%
- Value per saved subscriber: $30–$120

### Insight
High cost means profitability requires:
- High precision (smaller K),
- Higher-value users,
- Or strong retention lift.

Incentives should be reserved for the highest-confidence or highest-value segment.

---

# Scenario 2: Proactive Outreach (Low-Cost)

### Story
Target high-risk users with reminders, nudges, or light support.

Typical Assumptions:
- Cost per targeted user: $0.25–$2
- Success rate: 5–15%
- Value per saved subscriber: $30–$120

### Insight
Low cost makes ROI feasible even at moderate precision.
This is a strong default churn-model deployment policy.

---

# Practical Policy Takeaways

Given **P@10000 = 0.1801**:

1. Use **low-cost outreach** as the default intervention.
2. Use **incentives selectively**:
   - For highest-risk users only
   - For high-value users only
   - After outreach fails
3. Increasing precision (higher threshold or smaller K) improves ROI per user.

---

# Why This Matters

Machine learning metrics (PR-AUC, ROC-AUC) measure ranking quality.

Business decisions require:

- A targeting size (K or threshold)
- A cost model
- A retention impact assumption

This document bridges model performance to actionable economic policy.

---

# Optional Future Enhancement

Future improvements could:

- Compute ROI curves separately for offer vs outreach scenarios
- Simulate dynamic targeting (top 1% policy)
- Incorporate customer-level value segmentation (different V per user)

Current implementation demonstrates a complete ML → policy → ROI pipeline.