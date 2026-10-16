# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project is not versioned, so entries are dated.

## [2026-10-15]

### Added

- **API 1.1.0: model attributions.** `/predict` returns `explanations`, the top-5 TreeSHAP contributions in log-odds from `MODEL.predict(X, pred_contrib=True)`, plus `base_value`. On by default; `"explain": false` skips it (p50 51 ms vs 3.0 ms in-process). Batch scoring never explains. The "Why this score" panel shows these as model attributions when the live API returns them (`frontend/src/lib/attributions.ts`), and keeps the rule of thumb, labelled indicative, for Demo Mode and batch.
- **Policy v2** (`artifacts/champion/deployment_policy_v2.json`): tiers `contact_now` (≥ 0.68, $5 incentive), `nudge` (≥ 0.2058, $0.50, 12% illustrative save rate) and `monitor`. The API loads it at startup and returns `action` on every single and batch score; `/health` reports `policy_version`. v1 is kept, superseded.
- **Guardrails:** `n_missing_features` and a warning when more than half the features are missing (an empty record scores 0.99); `n_mostly_missing` on batches; `threshold` validated to [0, 1] (422); batches over 50,000 records rejected (413); optional `API_KEY` env var requiring `X-API-Key` on POST endpoints. `GET /` lists the 38-feature contract.
- **Output drift:** batches of ≥ 1,000 records return `score_drift`, the PSI of the tier mix against validation (`src/serving/policy.py`).
- **Eval harness** (`evals/`): economics, lift, calibration (info), score parity on 5 reference records, input contract and latency, with timestamped results in `evals/results/`. CI runs it against the in-process API and again against Cloud Run after each deploy. Finding: the nudge tier (scores 0.21–0.68) churned at 8.9%, so scores rank well but aren't calibrated.
- Docs: `docs/PRD.md`, `docs/decisions.md` (13 decisions), `docs/customer-deployment-playbook.md` (data contract, integration, security, 30/60/90 pilot, runbook), `evals/README.md`; API key setup in `docs/deploy-cloud-run.md`.
- Tests: `src/serving/test_policy.py` rewritten as pytest (tiers and economics vs artifacts, drift); 12 new API tests (41 total); `frontend/src/lib/attributions.test.ts` (62 front-end tests).

- **Front end source in the repo:** the React + Vite + TypeScript Decision Intelligence UI now lives in `frontend/`, deployed on Vercel at https://churn.ameyparmarthi.com/ (auto-deploys from `main`; replaces the Bolt-hosted copy).
- **"Why this score" panel** on single-customer scores: up to 5 indicative risk drivers with signed impact in probability points, plus a suggested action under the risk band. When the score comes from the model, the panel also shows the rule of thumb's own estimate. The drivers come from the transparent rule-of-thumb estimator, **not** from the LightGBM model.
- **"Top reasons (indicative)" column** in batch results (top 2 drivers per row).
- **Executive summary report** on the Policy Simulator (`#/simulator`): a print-ready one-pager covering the recommendation, key numbers, contact plan by risk band at the user's subscriber count, assumptions, model, risks and caveats. Exports: print / save as PDF, Markdown download (`churn-retention-summary-k<K>.md`) and copy text.
- `explainRecord`, `fmtImpact`, `summarizeFactors` and `BASELINE_RISK` in `frontend/src/lib/demoScore.ts`; `buildExecReport` and `reportToMarkdown` in `frontend/src/lib/report.ts`.
- Tests: `frontend/src/lib/demoScore.test.ts` and `frontend/src/lib/report.test.ts` (59 tests at the time).
- CI: a `frontend` job in `.github/workflows/deploy.yml` runs `npm ci`, typecheck, tests and build.
- Docs: `docs/frontend-architecture.md` and `docs/demo-script.md`.

### Changed

- `src/serving/policy.py`: the unused v1 `decide_single_with_fallback` / `decide_batch` replaced by `assign_action` and `score_drift`, which the API now calls.
- Risk band "Capacity tier" renamed "Nudge" to match the policy v2 action. The MLOps page's policy card and the Experiment page's decision log now describe v2.
- CI: `python -m pytest src/serving` (was `test_api.py` only), eval gate, post-deploy eval, `npm run lint`.
- `reports/champion_metadata.json`: `n_features` 38 (was 40), `feature_list_path` set, cutoff recorded as 2017-02-24 with the logged label kept; unknown fields say "not recorded" instead of "TBD".

- **Potentially breaking: batch CSV export columns.** `risk_band`, `suggested_action` and `top_reasons` are inserted after `in_top_k`. New order: `row_index, rank, churn_probability, label_at_0_68, in_top_k, risk_band, suggested_action, top_reasons, <original input columns...>`. Anything that reads the export by column position must be updated; readers that use header names are unaffected.
- `demoEstimate` refactored into named additive log-odds terms. Its output is numerically unchanged.
- Risk bands (`frontend/src/pages/score/risk.ts`) now carry a suggested action: Contact now, ≥ 0.68: personal outreach with retention incentive. Capacity tier, 0.2058–0.68: low-cost email or in-app nudge. Monitor, < 0.2058: no action; re-score next cycle.
- Print styles added to `frontend/src/index.css` for the report.
- `.dockerignore` excludes `frontend/` from the API image build context.
- Root README: live UI link, §7.4 front end, §7.5 API explanations, policy v2, evals, status and next steps updated; the stale "Streamlit *(planned)*" and "executive dashboard *(planned)*" entries were corrected.
- **Policy Simulator (`#/simulator`) reworked around the subscriber base.** Every count and dollar figure is now per cycle at the base in the header (default 1M), with K shown as a share of the base and the measured validation figures (e.g. net $17,666 at K = 1,478) in a footnote. Previously the results panel showed validation-window dollars next to scaled counts.
- **New "Which policy should we run?" card** (`PolicyOptions.tsx`, replaces `TieredPolicy.tsx`): ROI-max threshold, capacity top-10K with one channel, and tiered, side by side at the user's assumptions. At defaults and 1M subscribers: +$91K, −$35K and +$126K per cycle.
- `roiMultiple` split into `returnOnSpend` (gross, 3.39× at defaults) and `netRoi` (net ÷ spend, 239%). The results panel, Overview KPI and executive report show both, labelled.
- Sensitivity callout and report risks name every assumption tied for the largest swing (`sensitivity().leaders`); save rate and value per save always tie because net is linear in S·V.
- Docs aligned with the code and artifacts: README, `leaderboard.md`, `docs/executive_summary.md`, `reports/business_assumptions.md`, `artifacts/champion/notes.md`, `cloud/sagemaker/README.md`, `reports/sagemaker_mini_project.md`; `docs/deploy-cloud-run.md` rewritten for the keyless Workload Identity Federation deploy.

### Removed

- `render.yaml` (Render was superseded by Cloud Run) and Render mentions in the `Dockerfile` and Streamlit comments.
- Unused `@supabase/supabase-js` dependency from the Bolt scaffold (nothing imported it); the front-end package is renamed `churn-decision-intelligence`.
- `frontend/CHANGES-explanations-and-exec-summary.md` (a patch-handoff note, superseded by this changelog).

### Fixed

- **Validation split description.** The champion split is the 80th percentile of `txn_last_date` (2017-02-24); validation is the 193,205 subscribers whose last transaction is later. Docs previously said "train ≤ 2017-01-31, validate Feb 2017". The training script parses `YYYYMMDD` without a format string, so `metrics.json` stores the cutoff as `1970-01-01 00:00:00.020170224`; ordering is preserved, so the split itself is unaffected. The SageMaker job uses a fixed 2017-01-31 cutoff on a 10% random sample (not stratified), so its metrics are a workflow check, not parity.
- **Policy framing.** Docs called top-10K the primary policy, but at the documented assumptions it loses $6,776 per cycle (recorded in `threshold.json`); the API defaults to the 0.68 threshold. Docs now present the threshold as the default and top-K as a capacity option that pays only with a cheap lower-tier channel.
- **Lifts and figures.** Random-split lifts recomputed against the measured 6.39% prevalence (LightGBM 13.9×, not 14.3×); top-10K precision lift is 14.5× (was "3x" against the wrong base); PR-AUC lift 43.4×, P@5K 21.6×; outreach scenario $12,290 (was ~$12,200); top-10K at defaults loses $6,776 (was "close to break-even"); A/B sample size 2,402 per arm (was ~2,800); data sizes from notebook 08 (32.7 GB raw, 10.7 GB Parquet, 123 MB model table, 992,931 rows); pytest count 22.
- Dollar values labelled "Net ROI" in the Streamlit apps and docs are now "net value"; net ROI means net ÷ spend.
- UI: Evidence page prevalence (6.4%), ensemble wording (soft vote put weight 1.0 on LightGBM), time-based lifts, SageMaker card; MLflow is tracking only (the registry is SageMaker); sample records described as "recorded champion scores", not real customers.
- Generated artifacts (`valid_scored.parquet`, `flaml_best_params.json`, `reports/threshold_sweep.csv`) are now described as gitignored outputs, not committed files.
