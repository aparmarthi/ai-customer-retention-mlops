# Front-End Architecture: Risk Drivers and Executive Summary

How the Decision Intelligence UI (`frontend/`) explains a score and turns a policy scenario into a one-page executive report. Both features run entirely in the browser: no new dependencies, no backend or API changes, no database. Setup, routes and scripts are in [frontend/README.md](../frontend/README.md).

Only the Score page calls the API (`/health`, `/predict`, `/predict_batch`). Every other page reads static validation data from `frontend/src/data/*.ts` (Feb-2017 window: 193,205 subscribers, 2,402 churners, 1.24% churn).

---

## 1. Explanation Flow ("Why this score")

```
CustomerRecord (form, sample, profile, CSV or JSON row)
        |
        v
explainRecord(record, limit)                 src/lib/demoScore.ts
  terms(record) -> named log-odds terms
  -> { estimate, factors[{ id, text, impact }] }
        |
        +--> RiskFactors panel (limit 5)     src/pages/score/RiskFactors.tsx, used by SingleScore.tsx
        +--> Batch "Top reasons" chips (2)   src/pages/score/BatchResults.tsx, first 100 rows shown
        `--> Batch CSV top_reasons (3)       summarizeFactors() -> "text (+N pts); ..."

score -> riskBand(p) -> { label, action }    src/pages/score/risk.ts
```

### 1.1 What the drivers are, and are not

The drivers come from the transparent rule-of-thumb estimator (`demoEstimate`) that Demo Mode uses when the API is unreachable. They are **indicative, rule-of-thumb drivers, not the trained LightGBM model's attributions**, and the UI labels them that way in four places: the panel footer, the panel's info tooltip, the batch column header ("Top reasons (indicative)") and a caveat in the executive report. When the headline score comes from the model (live or recorded), the panel also shows the rule of thumb's own estimate next to the model's score, so any disagreement is visible.

### 1.2 Terms

`demoEstimate` was refactored into named additive terms on a base logit of −3. Its output is numerically unchanged, and a test asserts `explainRecord(r).estimate === demoEstimate(r)` for every sample and profile.

| Term (input) | Raises risk when | Included |
|---|---|---|
| `auto_renew_rate` | Fewer payments on auto-renew | Always (defaults to 0.5 if missing) |
| `cancel_rate` | More cancelled transactions | Always (defaults to 0) |
| `membership_expire_date_max` | Lapsed before 2017-02-28 (capped at 30 days); runs > 45 days lowers risk | Valid `YYYYMMDD` only |
| `log_last_date` | More than 7 days since last listen (capped at 60) | Valid `YYYYMMDD` only |
| `txn_cnt` | Few transactions; extra bump at ≤ 2 | Always (defaults to 3) |
| `log_active_days` | Few active days; extra bump below 15 | Always (defaults to 30) |
| `actual_paid_mean` vs `plan_list_price_mean` | Paying < 75% of list price | Both present |
| `listen_full_share` | Under 70% of plays heard to the end | Present |
| `total_secs_mean_per_active_day` | Little listening per active day | Present |

Defaulted inputs still count toward the estimate but are never shown as a driver, because the panel only explains with inputs the record actually provided.

### 1.3 Attribution method

```
z        = −3 + Σ logit_i            p = sigmoid(z)
baseline = sigmoid(−3) ≈ 4.7%        (BASELINE_RISK)

impact_i = logit_i × (p − baseline) / (z − (−3))
```

Each term's logit contribution is multiplied by the sigmoid's **average slope** between the baseline and this customer. Across all terms, impacts therefore sum exactly to `p − baseline`, and the attribution stays well-behaved at extreme probabilities. When `z` equals the baseline, the local slope `p(1 − p)` is used instead.

**Why not leave-one-out?** It was tried first. Removing one term and measuring the change in `p` saturates near 0 and 1. For a customer at 99.7% risk (z = 6), dropping a 1.6-logit factor only moves the score to 98.8%, about 1 pt, even though that factor is a large part of a 95-pt gap above baseline. The deltas understate every driver and sum to far less than the gap. The average-slope method avoids both problems.

**Display rules:** only provided inputs, drop factors under 0.2 pts (`MIN_IMPACT = 0.002`), sort by absolute impact, keep the top `limit`. `fmtImpact` shows signed whole points and "<1 pt" for values that round to zero. Because of these filters, the visible bars can sum to less than the full gap.

### 1.4 Risk bands and suggested actions

Thresholds come from `src/data/validation.ts` (`DEFAULT_THRESHOLD`, `CAPACITY_THRESHOLD`), matching `artifacts/champion/threshold.json`.

| Band | Score | Action |
|---|---|---|
| Contact now | ≥ 0.68 | Personal outreach with retention incentive |
| Capacity tier | 0.2058–0.68 | Low-cost email or in-app nudge |
| Monitor | < 0.2058 | No action; re-score next cycle |

### 1.5 Batch CSV export

`churn_scores_top<K>.csv`, all rows, columns in this order:

```
row_index, rank, churn_probability, label_at_0_68, in_top_k, risk_band, suggested_action, top_reasons, <original input columns...>
```

`risk_band`, `suggested_action` and `top_reasons` were added after `in_top_k`, so consumers that parse by position must update (see [CHANGELOG](../CHANGELOG.md)). No code in this repo consumes the export.

---

## 2. Executive Report Flow

```
ScenarioState { k, valuePerSave, costPerContact, saveRate, subscribers }   (mirrored in the URL hash)
        |
        v
buildExecReport(state, link)                 src/lib/report.ts   (pure, unit-tested)
  scenario(k), sensitivity(k), tpAtK(), scale(x, subs) from src/lib/economics.ts
  -> ExecReport { headline, metrics, bands, assumptions, model, risks, caveats, link }
        |
        v
ExecutiveSummary modal                       src/pages/simulator/ExecutiveSummary.tsx  (opened from SimulatorPage)
  +--> Print or save PDF   window.print() + @media print rules in src/index.css hide the app shell
  +--> Download            reportToMarkdown() -> churn-retention-summary-k<K>.md
  `--> Copy text           same Markdown to the clipboard
```

- **Headline:** if net ≥ 0, it gives contacts per cycle at the user's scale, net per cycle and per year (× 12, 1 cycle = 1 month), ROI multiple and uplift vs random targeting. Otherwise it states the loss and the break-even value per save (`k·C / (TP·S)`) and save rate.
- **Scaling:** every count and dollar figure goes through `scale(x, subs) = x × subs / 193,205`, which assumes the user's base has the validation window's churn rate and risk mix. The report states this assumption.
- **Contact plan:** the bands are fixed validation-curve segments, independent of the chosen K: Contact now = top 1,478 (score ≥ 0.68), Capacity tier = ranks 1,479–10,000, Monitor = the rest. Expected churners per band = `tpAtK(to) − tpAtK(from)`, scaled.
- **Risks:** the assumption with the widest ±50% swing (`sensitivity(...).bars[0]`), the break-even save rate and contact cost, and a prompt to validate the assumed save rate with a holdout experiment.
- **Caveats:** a validation-window simulation, not realised revenue; scaling assumes the same churn mix; customer-level reasons are indicative.

---

## 3. Tests

`npm test` in `frontend/` runs 54 vitest tests and is part of CI. The new files:

- `src/lib/demoScore.test.ts`: parity with `demoEstimate`, top-N ordering, sign of key drivers, sum-to-gap when all inputs are provided, no drivers from missing inputs, impact formatting.
- `src/lib/report.test.ts`: positive recommendation at defaults, break-even guidance for a loss-making scenario, the three bands covering the whole subscriber base, and every section plus the scenario link in the Markdown export.

---

## 4. Next Step: Model Attributions

The rule of thumb stands in until the API can explain its own scores. The proposal ([README §7.5](../README.md#75-proposed-next-step-model-explanations-in-the-api)) is an optional `explanations` field on `/predict` and `/predict_batch`, with top-N TreeSHAP contributions from `MODEL.predict(X, pred_contrib=True)`. When the field is present, the panel, batch chips and CSV `top_reasons` would show those contributions, and the rule of thumb would remain only for Demo Mode.
