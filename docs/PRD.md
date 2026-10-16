# PRD — Churn Decision Intelligence

**Status:** v2 live (API on Cloud Run, UI on Vercel). **Owner:** Amey Parmarthi. **Last updated:** 2026-10-15.

Related: [decision log](decisions.md) · [customer deployment playbook](customer-deployment-playbook.md) · [evals](../evals/README.md) · [executive summary](executive_summary.md)

---

## 1. Problem

A subscription business loses about 1.2% of active subscribers per cycle (KKBox validation window: 2,402 of 193,205). Retention teams have a fixed budget and a fixed number of contacts. Without a ranking they either contact everyone, contact no one, or use rules like "inactive 30 days" that miss most churners.

The real question is **who to contact, through which channel, at what cost** — and how to know it paid off.

**Why now:** the model is good enough to act on (PR-AUC 0.539, 43× the base rate). The open risk is no longer accuracy; it is spending incentive budget on subscribers who would have stayed anyway.

## 2. Users

| User | Job to be done | Surface |
|---|---|---|
| Retention operations lead | Get this cycle's contact list, split by channel, within capacity | Batch scoring, `action` per subscriber, CSV export |
| CRM / care agent | See why a subscriber is flagged before a conversation | `/predict` with top-5 model attributions |
| Finance / executive sponsor | Decide whether to fund the program and at what scale | Policy simulator, executive summary |
| Data / ML owner | Keep the model honest in production | Evals, drift, missing-data guardrail |

## 3. Success metrics

**North Star:** incremental subscribers retained per cycle, measured against a randomized holdout, net of contact cost.

The model cannot measure this on its own: it predicts who will churn, not who will respond. The simulated values below assume save rates; the pilot measures them ([playbook §5](customer-deployment-playbook.md#5-306090-pilot-plan)).

| Guardrail | Target | Today (validation, simulated) |
|---|---|---|
| Precision of `contact_now` | ≥ 60% | 70.6% (1,044 of 1,478) |
| Net value per cycle at measured save rates | > $0 | $17,666 threshold-only, $24,306 tiered (assumed save rates) |
| Complaint / unsubscribe rate, treatment vs holdout | No significant increase | Not yet measured |
| Output drift (PSI of tier mix) | < 0.25 every cycle | Measured per batch of ≥ 1,000 |
| `/predict` p95 latency with explanations | < 500 ms | 52.7 ms in-process ([evals](../evals/README.md)) |

## 4. Scope

**In scope (v2):**
- Score a subscriber (`/predict`) or a batch of up to 50,000 (`/predict_batch`, JSON or CSV).
- Return a policy `action` per score: `contact_now` (≥ 0.68), `nudge` (0.2058–0.68), `monitor`. Tiers live in `artifacts/champion/deployment_policy_v2.json`.
- Return the top-5 TreeSHAP attributions per single score.
- Warn on mostly-missing input; report output drift on large batches.
- Policy simulator and one-page executive summary in the UI.

**Out of scope (and why):**
- **Uplift modelling.** Needs treatment/holdout outcomes, which the pilot produces.
- **Calibrated probabilities.** Scores rank well but overstate risk in the nudge tier (8.9% observed churn for scores of 0.21–0.68). The economics use observed precision, not raw scores, so calibration isn't needed yet. See [decisions D7](decisions.md#d7-scores-are-rankings-not-calibrated-probabilities).
- **Writing to the CRM.** Integration is customer-specific ([playbook §3](customer-deployment-playbook.md#3-integration-patterns)).
- **Automated retraining.** Designed (README §11.3), not built.

## 5. User experience

**Happy path (operations):** export this cycle's features → upload to `/predict_batch` → receive score, action and rank per row in input order → load `contact_now` into the incentive campaign and `nudge` into the email journey → check `score_drift.status` is `stable`.

**Happy path (agent):** open a subscriber → `/predict` → see the score, the tier action, and the five features that moved the score most, in probability points (e.g. "Auto-renew rate: 0" pushing the score up).

**Error states:**

| Situation | Behaviour |
|---|---|
| More than half the features missing | Score is still returned, with a warning. The model reads missing activity as churn: an empty record scores 0.99 |
| Threshold outside [0, 1] | 422 |
| Batch over 50,000 records | 413; the caller chunks |
| Batch of ≥ 1,000 with a shifted tier mix | `score_drift.status` = `moderate_shift` or `significant_shift` |
| `API_KEY` set, header missing or wrong | 401 |
| API cold start (UI) | UI shows a labelled Demo Mode until the API responds |

## 6. Trade-offs

| Optimized for | Gave up | Why |
|---|---|---|
| Net value per cycle (ROI threshold 0.68) | Recall: 43% vs 75% at top-10K | Top-10K with a $5 incentive for everyone loses $6,776 per cycle |
| Explanations on every interactive score | Latency: p50 3 ms → 51 ms | Agents need a reason, and 51 ms is well inside budget; batch skips it |
| Warning on missing data | Hard rejection | Partial records are common in real CRMs; a silent 0.99 is the real failure |
| Output drift (no labels needed) | Per-feature drift | Labels arrive ~30 days late; the tier mix is the cheapest early signal |

## 7. Scaling path

- **10× (≈ 2M subscribers per cycle):** chunk batch scoring into 50,000-record calls (1,000 records take 27 ms in-process); still one Cloud Run service.
- **100× (≈ 20M):** move batch scoring next to the warehouse (a scheduled job loading `model.pkl`) and keep the API for interactive calls. Feature pipeline stays on DuckDB until it exceeds one machine (README §10).

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| Save rates are assumed; every dollar figure scales with them | Randomized holdout in the pilot; replace assumptions before committing budget |
| Nudge save rate (12%) is illustrative | Measure it in the pilot's nudge arm |
| Model trained on 2015–2017 KKBox behaviour | Parity and drift checks on the customer's own data in the first 30 days |
| Base rate shift (6.4% train vs 1.24% holdout) | Policy cutoffs were chosen on the holdout window, not on training |
| Cutoffs were tuned on the same window that reports their economics, so those figures are optimistic | The pilot re-measures precision and net value on new cycles |
