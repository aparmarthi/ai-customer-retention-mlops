# Churn Decision Intelligence

Executive and operator front end for the FastAPI churn-prediction service trained on the KKBox subscription-music dataset (Kaggle WSDM 2018).

React + Vite + TypeScript + Tailwind, Recharts for charts, lucide-react for icons. Built in bolt.new and hosted on Bolt at https://churn-decision-intel-3o63.bolt.host/. This folder is the source; run every command below from `frontend/`.

## Pages

| Route | Purpose |
| --- | --- |
| `#/overview` | Executive one-pager: KPIs scaled to your subscriber base, model vs random, pipeline |
| `#/simulator` | Policy simulator: K contacts, V / C / S assumptions, net-value curve, sensitivity, tiered policy, printable executive summary (PDF / Markdown) |
| `#/evidence` | Time-based validation, random-split model-family benchmark, SHAP and threshold plots, SageMaker parity |
| `#/score` | Live API: single customer, batch CSV, API console. Each score shows its top risk drivers, and the batch export adds risk band, suggested action and top reasons |
| `#/mlops` | Deployment, monitoring plan (mock), deployment policy JSON |
| `#/experiment` | Holdout plan, sample-size calculator, decision log, risks |

Only **Score Customers** calls the API. Every other page uses static validation data in `src/data/*.ts`. All business-value numbers are simulated on the Feb-2017 validation window (193,205 subscribers, 1 cycle = 1 month) and are not realized revenue.

## Why this score (indicative risk drivers)

Every single-customer score shows a "Why this score" panel: up to 5 diverging bars (rose raises risk, emerald lowers it) and a suggested action under the risk band. These are **indicative, rule-of-thumb drivers, not the trained LightGBM model's attributions**. They come from the transparent estimator in `src/lib/demoScore.ts`, the same one Demo Mode uses. When the headline score comes from the model, the panel also shows the rule of thumb's own estimate next to the model's score, so any disagreement is visible.

`explainRecord(record, limit)` splits the estimate into named additive log-odds terms: auto-renew rate, cancel rate, membership expiry vs the 2017-02-28 cutoff, days since last listen, transaction count, active days, discount vs list price, full-listen share and listening seconds per active day. Each term's impact in probability points is:

```
impact_i = logit_i × (p − baseline) / (z − baseline_logit)      baseline = sigmoid(−3) ≈ 4.7%
```

That is, the term's logit contribution times the sigmoid's average slope between the baseline and this customer. Across all terms, impacts sum exactly to `p − baseline`, and the method stays stable near 0 and 1. Leave-one-out deltas were tried first and rejected because they saturate at extreme probabilities. The panel shows only inputs present in the record, drops factors under 0.2 pts and shows "<1 pt" for values that round to zero, so the visible bars can sum to less than the full gap. `demoEstimate` output is unchanged by this refactor.

| Risk band | Score | Suggested action |
| --- | --- | --- |
| Contact now | ≥ 0.68 | Personal outreach with retention incentive |
| Capacity tier | 0.2058 – 0.68 | Low-cost email or in-app nudge |
| Monitor | < 0.2058 | No action; re-score next cycle |

Real per-customer attributions need the API to return them, e.g. top-N SHAP values from `/predict` (see the [proposed next step](../README.md#75-proposed-next-step-model-explanations-in-the-api) in the root README). Design notes: [docs/frontend-architecture.md](../docs/frontend-architecture.md).

## Batch results and CSV export

The results table adds a "Top reasons (indicative)" column with the top 2 driver chips per row (computed for the first 100 rows shown). **Export CSV** writes `churn_scores_top<K>.csv` with every row and these columns, in order:

```
row_index, rank, churn_probability, label_at_0_68, in_top_k, risk_band, suggested_action, top_reasons, <original input columns...>
```

`risk_band`, `suggested_action` and `top_reasons` are new, inserted after `in_top_k`. `top_reasons` holds the top 3 drivers, `;`-separated, e.g. `Auto-renew on 0% of payments (+29 pts); ...`. Anything parsing the export by column position must be updated.

## Executive summary report

**Executive summary** on `#/simulator` opens a print-ready white report built from the current scenario (K, V, C, S, subscribers):

- **Recommendation:** contacts per cycle at your scale, expected net per cycle and per year, ROI multiple and uplift vs random targeting. A loss-making scenario says so and gives the break-even value per save and save rate.
- **Key numbers:** expected net / cycle, return on contact spend, churners reached (recall), hit rate (precision) and lift vs random.
- **Contact plan by risk band**, scaled from the Feb-2017 curve (193,205 subs, 2,402 churners): Contact now (top 1,478), Capacity tier (ranks 1,479–10,000), Monitor (the rest), each with subscribers, share, expected churners and action.
- **Assumptions, Model, Risks and sensitivities** (most sensitive assumption at ±50%, break-even points, save rate to be validated with a holdout) and **Caveats**, plus a shareable scenario link.

Exports: **Print or save PDF** (browser print; `@media print` rules in `src/index.css` hide the app shell), **Download** as Markdown (`churn-retention-summary-k<K>.md`) and **Copy text**. The logic is pure and unit-tested: `buildExecReport()` and `reportToMarkdown()` in `src/lib/report.ts`. All figures are simulated on the validation window, not realised revenue.

## Configuration

| Variable | Default |
| --- | --- |
| `VITE_API_URL` | `https://churn-api-995576020018.us-central1.run.app` (live Cloud Run service) |

The base URL can also be overridden at runtime from the **Settings** drawer (gear icon). The override is stored in `localStorage` and `/health` is re-pinged on save. "Reset" returns to `VITE_API_URL`.

## Cold start and Demo Mode

The Cloud Run service scales to zero. On load the app calls `GET /health` with a **12 s** timeout and shows "Waking API (cold start)…". Subsequent health checks use **6 s**.

If `/health` fails or times out the app switches to **Demo Mode** (amber status pill):

- Single customer: the 5 recorded validation customers return their real champion scores (`demo` badge). The 11 generated customer profiles and any custom edits are scored by a transparent rule of thumb in `src/lib/demoScore.ts` (`demo estimate · not the model` badge).
- Batch: "Use 5 samples" and the Paste JSON tab rank locally using recorded scores where available and the rule of thumb otherwise. CSV uploads require the live API.

Profiles and the 60-customer example batch are generated deterministically in `src/data/profiles.ts` from 11 behavioural archetypes (loyal auto-renewer to expired one-off buyer), so they span all three risk bands. Tests in `src/data/profiles.test.ts` enforce that spread.

Click the status pill to retry.

## `/predict_batch` input modes

1. **CSV upload (multipart)** — the file is sent unchanged as field `file`:
   ```bash
   curl -X POST "$API/predict_batch?policy=top_k&k=10000" -F "file=@customers.csv"
   ```
   The browser parses the CSV only to show input columns next to scores; it is never converted to JSON for sending.
2. **JSON records** — used for the samples and for the **Paste JSON** tab on the Score page. Paste either an array of flat records or `{"records": [...]}`; nulls are dropped and booleans become 0/1. Up to 50,000 records. The tab is pre-filled with a 60-customer example that includes extra `customer_id` and `segment` fields (ignored by the model, useful for reading results).
   ```bash
   curl -X POST "$API/predict_batch" -H "Content-Type: application/json" \
     -d '{"records":[{...}],"policy":"top_k","k":10}'
   ```

Items come back in input order and are joined to input rows by index. With `top_k`, `rank` 1 is the highest risk. K is capped at the number of rows.

Missing feature columns are treated as NaN by the server; extra columns are ignored. `gender` is `male | female | unknown`. Dates are `YYYYMMDD` integers.

## Core math

`src/lib/economics.ts` holds pure functions. Precision is always derived from the operating curve (`tpAtK(k) / k`), never a user input.

At defaults (V = $120, C = $5, S = 20%), K = 1,478 gives TP 1,044, precision 70.6%, recall 43.5%, net $17,666/cycle, ROI 3.4×, break-even save rate 5.9%, break-even contact cost $16.95.

These values are locked in `src/lib/sanity.ts`. `npm test` asserts them (plus interpolation, sample-size and link round-trip tests), and the Simulator header shows a live "Math verified · N/N" badge computed from the same checks.

## Shareable scenarios

The Simulator mirrors its inputs into the URL, e.g. `#/simulator?k=1478&v=120&c=5&s=0.2&subs=1000000`. "Copy link" copies that URL and "Copy scenario" appends it to the text summary. Out-of-range or malformed values are clamped or fall back to defaults.

## Scripts

```bash
cd frontend
npm ci
npm run dev        # local dev server
npm run build      # production build
npm run typecheck
npm test           # vitest: 54 tests
npm run lint
```

Test files: `src/lib/economics.test.ts`, `src/lib/scenario.test.ts`, `src/lib/stats.test.ts`, `src/data/profiles.test.ts`, `src/pages/score/jsonBatch.test.ts`, `src/lib/demoScore.test.ts` (driver ranking, sum-to-gap, no invented inputs, `demoEstimate` parity) and `src/lib/report.test.ts` (report numbers and Markdown export). CI runs typecheck, tests and build on every push and PR to `main` (`frontend` job in `.github/workflows/deploy.yml`).

The live app is published from bolt.new. Routing is hash-based, so `dist/` also works on any static host (Vercel, Netlify, Cloud Storage) with no rewrites. Set `VITE_API_URL` at build time to target another API.
