# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project is not versioned, so entries are dated.

## [2026-10-15]

### Added

- **Front end source in the repo:** the React + Vite + TypeScript Decision Intelligence UI now lives in `frontend/`, the same app that is live at https://churn-decision-intel-3o63.bolt.host/.
- **"Why this score" panel** on single-customer scores: up to 5 indicative risk drivers with signed impact in probability points, plus a suggested action under the risk band. When the score comes from the model, the panel also shows the rule of thumb's own estimate. The drivers come from the transparent rule-of-thumb estimator, **not** from the LightGBM model.
- **"Top reasons (indicative)" column** in batch results (top 2 drivers per row).
- **Executive summary report** on the Policy Simulator (`#/simulator`): a print-ready one-pager covering the recommendation, key numbers, contact plan by risk band at the user's subscriber count, assumptions, model, risks and caveats. Exports: print / save as PDF, Markdown download (`churn-retention-summary-k<K>.md`) and copy text.
- `explainRecord`, `fmtImpact`, `summarizeFactors` and `BASELINE_RISK` in `frontend/src/lib/demoScore.ts`; `buildExecReport` and `reportToMarkdown` in `frontend/src/lib/report.ts`.
- Tests: `frontend/src/lib/demoScore.test.ts` and `frontend/src/lib/report.test.ts` (front-end suite now 54 tests).
- CI: a `frontend` job in `.github/workflows/deploy.yml` runs `npm ci`, typecheck, tests and build.
- Docs: `docs/frontend-architecture.md`, `docs/demo-script.md`, and a proposed `explanations` field for `/predict` and `/predict_batch` (README §7.5; not implemented).

### Changed

- **Potentially breaking: batch CSV export columns.** `risk_band`, `suggested_action` and `top_reasons` are inserted after `in_top_k`. New order: `row_index, rank, churn_probability, label_at_0_68, in_top_k, risk_band, suggested_action, top_reasons, <original input columns...>`. Anything that reads the export by column position must be updated; readers that use header names are unaffected.
- `demoEstimate` refactored into named additive log-odds terms. Its output is numerically unchanged.
- Risk bands (`frontend/src/pages/score/risk.ts`) now carry a suggested action: Contact now, ≥ 0.68: personal outreach with retention incentive. Capacity tier, 0.2058–0.68: low-cost email or in-app nudge. Monitor, < 0.2058: no action; re-score next cycle.
- Print styles added to `frontend/src/index.css` for the report.
- `.dockerignore` excludes `frontend/` from the API image build context.
- Root README: live UI link, §7.4 front end, §7.5 proposed API explanations, status and next steps updated; the stale "Streamlit *(planned)*" and "executive dashboard *(planned)*" entries were corrected.
