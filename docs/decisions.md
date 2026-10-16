# Decision Log

Key decisions and their trade-offs, newest last. Each entry: context, decision, consequences. Figures are on the 193,205-subscriber validation window; dollar values are simulated, not realised revenue.

| # | Decision | Status |
|---|---|---|
| D1 | [Chronological split; keep the mislabelled cutoff, don't retrain](#d1-chronological-split-keep-the-mislabelled-cutoff-dont-retrain) | Accepted |
| D2 | [PR-AUC as the selection metric](#d2-pr-auc-as-the-selection-metric) | Accepted |
| D3 | [Single LightGBM over the ensemble and deep learning](#d3-single-lightgbm-over-the-ensemble-and-deep-learning) | Accepted |
| D4 | [ROI threshold as the default; top-K only with a cheap channel](#d4-roi-threshold-as-the-default-top-k-only-with-a-cheap-channel) | Accepted |
| D5 | [Tiered policy v2 as a versioned artifact the API serves](#d5-tiered-policy-v2-as-a-versioned-artifact-the-api-serves) | Accepted, supersedes v1 |
| D6 | [TreeSHAP attributions in `/predict`, not in batch](#d6-treeshap-attributions-in-predict-not-in-batch) | Accepted |
| D7 | [Scores are rankings, not calibrated probabilities](#d7-scores-are-rankings-not-calibrated-probabilities) | Accepted; calibration is next |
| D8 | [Warn on missing data, don't reject](#d8-warn-on-missing-data-dont-reject) | Accepted |
| D9 | [Output drift on the tier mix before per-feature drift](#d9-output-drift-on-the-tier-mix-before-per-feature-drift) | Accepted |
| D10 | [Public demo without a key; optional `API_KEY` for real deployments](#d10-public-demo-without-a-key-optional-api_key-for-real-deployments) | Accepted |
| D11 | [Cloud Run with keyless deploys; scale to zero](#d11-cloud-run-with-keyless-deploys-scale-to-zero) | Accepted |
| D12 | [React for the customer-facing UI; Streamlit stays internal](#d12-react-for-the-customer-facing-ui-streamlit-stays-internal) | Accepted |
| D13 | [Evals gate CI and every deploy](#d13-evals-gate-ci-and-every-deploy) | Accepted |

---

## D1. Chronological split; keep the mislabelled cutoff, don't retrain

**Context.** Churn is temporal, so a random split leaks future behaviour (random-split lift is flattering). The trainer splits at the 80th percentile of `txn_last_date`, which is 2017-02-24. It parses `YYYYMMDD` integers without a format string, so `metrics.json` logs the cutoff as `1970-01-01 00:00:00.020170224`.

**Decision.** Keep the champion. The parse bug changes the label, not the ordering, so the split is still chronological and every metric stands. Document it in `reports/champion_metadata.json` and the CHANGELOG instead of retraining.

**Consequences.** No artifact churn and no metric drift across docs. The fix (`format="%Y%m%d"`) belongs in the next retrain, which must also re-run the downstream validation.

## D2. PR-AUC as the selection metric

**Context.** 1.24% prevalence on the holdout. ROC-AUC (0.966) looks excellent for almost any decent ranker.

**Decision.** Select on PR-AUC (0.539, 43× the base rate) and report precision and recall at the K the business will actually contact.

**Consequences.** Model comparisons track the positives we act on. ROC-AUC is still reported, labelled as the flattering one.

## D3. Single LightGBM over the ensemble and deep learning

**Context.** 12 experiments across baselines, 7 model families, a soft-vote ensemble and FLAML AutoML.

**Decision.** Ship the FLAML-tuned LightGBM. The ensemble put weight 1.0 on LightGBM, and TabNet / FT-Transformer trained 8–11× slower with lower PR-AUC.

**Consequences.** One 18.7 MB artifact, CPU serving, exact TreeSHAP for free (D6).

## D4. ROI threshold as the default; top-K only with a cheap channel

**Context.** The original ops framing was "contact the top 10,000 every cycle". At the documented assumptions ($120 per save, $5 per contact, 20% save rate) that loses **$6,776** per cycle: below the top 1,478, precision is too low to cover a $5 contact.

**Decision.** Default `churn_label` to the ROI-max threshold (0.68: 1,478 contacts, 70.6% precision, +$17,666). Keep `/predict_batch?policy=top_k` for teams with a fixed call list.

**Consequences.** Recall drops from 75% to 43%. The remaining churners are reached only through a cheaper channel (D5).

## D5. Tiered policy v2 as a versioned artifact the API serves

**Context.** v1 (`deployment_policy.json`) said "top-K primary, threshold fallback", which contradicted D4, and the API didn't read it.

**Decision.** `deployment_policy_v2.json` defines three tiers: `contact_now` (≥ 0.68, $5 incentive), `nudge` (0.2058–0.68, $0.50 email or in-app, 12% illustrative save rate) and `monitor`. The API loads it at startup and returns `action` with every score; `/health` reports the version. v1 stays in the repo, superseded. Tests and evals recompute the tier counts and net values (+$24,306 tiered) from the artifact.

**Consequences.** Changing a cutoff or cost is a reviewed artifact change, not a code change. The 12% nudge save rate is a placeholder until the pilot measures it.

## D6. TreeSHAP attributions in `/predict`, not in batch

**Context.** Agents and reviewers need to know *why the model* flagged someone. The UI originally showed rule-of-thumb drivers, labelled as indicative.

**Decision.** `/predict` returns the top 5 contributions from `MODEL.predict(X, pred_contrib=True)`: exact TreeSHAP in log-odds plus `base_value`, with no new dependency. It is on by default and can be turned off with `"explain": false`. Batch scoring never explains. The UI converts log-odds to probability points and labels them "model attributions"; Demo Mode keeps the rule of thumb, labelled "indicative drivers".

**Consequences.** p50 latency rises from 3.0 ms to 51 ms in-process (146 trees × 1,212 leaves). That's inside the 500 ms budget for interactive use but would be wasted on a 50,000-row batch. Attributions explain the score, not what causes churn.

## D7. Scores are rankings, not calibrated probabilities

**Context.** Trained at 6.4% prevalence, evaluated at 1.24%. On validation, the nudge tier (scores 0.2058–0.68) churned at **8.9%**, well below its scores; `contact_now` churned at 70.6% and `monitor` at 0.33%. The eval harness reports this as an info-level check.

**Decision.** Treat scores as a ranking. All economics use observed precision per tier (churners ÷ contacts), never the raw score. Don't calibrate yet.

**Consequences.** Nobody should read 0.45 as "45% likely to churn"; the docs and UI avoid that. The next step is isotonic calibration on a later window, which is needed before any expected-value-per-subscriber logic.

## D8. Warn on missing data, don't reject

**Context.** Missing features become NaN, and the model reads missing activity as churn: an empty record scores 0.99. A wrong field name upstream would silently put healthy subscribers into `contact_now`.

**Decision.** Keep scoring, but `/predict` returns `n_missing_features` and a warning when more than half are missing, and batches return `n_mostly_missing`. `GET /` publishes the 38-feature contract.

**Consequences.** Partial CRM records still score. Integrations must alert on the warning count; the playbook's data-contract check makes that a go-live gate.

## D9. Output drift on the tier mix before per-feature drift

**Context.** Labels arrive about 30 days after scoring, so performance metrics lag a full cycle.

**Decision.** Every batch of ≥ 1,000 records returns `score_drift`: the PSI of its tier mix against validation (< 0.1 stable, < 0.25 moderate, else significant). Smaller batches return `null` because their shares are too noisy.

**Consequences.** A label-free early warning with three bins and no stored state. Per-feature PSI (README §11.1) is still designed, not built, and needs a feature baseline store.

## D10. Public demo without a key; optional `API_KEY` for real deployments

**Context.** The portfolio UI calls the API from the browser, so any key it held would be public.

**Decision.** If `API_KEY` is set, both POST endpoints require a matching `X-API-Key` (constant-time compare, 401 otherwise); `/health` stays open for probes. The public demo leaves it unset.

**Consequences.** In a customer deployment the key goes in Secret Manager and the caller is a server (CRM or batch job), never a browser. Stronger options there: Cloud Run IAM invoker auth or internal ingress, plus `CORS_ALLOW_ORIGINS` restricted.

## D11. Cloud Run with keyless deploys; scale to zero

**Context.** One small CPU service, a solo maintainer, no budget for idle instances.

**Decision.** Cloud Run (1 vCPU, 1 GiB, 0–3 instances), deployed from GitHub Actions via Workload Identity Federation, so no service-account key exists anywhere.

**Consequences.** The first request after idle is a cold start of a few seconds; the UI covers it with a labelled Demo Mode, and `--min-instances=1` is a one-line change for a demo or a customer SLA ([deploy guide](deploy-cloud-run.md#cold-starts)).

## D12. React for the customer-facing UI; Streamlit stays internal

**Context.** Two audiences: executives and operators (customer-facing), and the data team (internal).

**Decision.** The decision UI is React + TypeScript on Vercel, with typed API contracts, unit-tested economics and a print-ready executive summary. The Streamlit apps stay as internal prototypes. An unused Supabase dependency from the Bolt scaffold was removed: app data goes through FastAPI.

**Consequences.** Two front ends to keep in sync with the same figures. The eval harness and the README results table are the source of truth.

## D13. Evals gate CI and every deploy

**Context.** Docs, UI and artifacts once disagreed on headline figures (e.g. a stale $17.6K). Tests checked code paths, not business numbers.

**Decision.** `python -m evals.run` recomputes the economics and lifts from the committed artifacts, checks score parity on five reference records, the input contract and latency, and writes timestamped results to `evals/results/`. CI runs it against the in-process API on every push and PR, and again against Cloud Run after each deploy.

**Consequences.** The headline economics and lifts can't drift from the artifacts without failing CI (prose docs are still checked by hand). A deploy that changes scores fails the post-deploy eval right after traffic shifts; rollback is one command ([playbook §6](customer-deployment-playbook.md#6-runbook)). A staging revision gated by the same eval is the next step.
