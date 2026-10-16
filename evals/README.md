# Evals

`evals/run.py` checks that the deployed service, the committed artifacts and the documented figures agree. Unit tests (`src/serving/test_*.py`) check code paths; evals check the numbers a stakeholder would quote.

```bash
python -m evals.run                                   # offline: artifacts only, no API
python -m evals.run --api-url local                   # + in-process API (current code)
python -m evals.run --api-url https://churn-api-995576020018.us-central1.run.app
```

Each run writes `results/<UTC timestamp>_<offline|local|remote>.json` and exits 1 if any check fails. Set `API_KEY` if the target requires it. CI runs the local eval on every push and PR, and the remote eval after every Cloud Run deploy.

## What is checked

| Group | Check | Pass condition |
|---|---|---|
| Economics | Net value per cycle for threshold-only, top-10K single channel, tiered | Recomputed from `deployment_policy_v2.json` tier counts within $1 of the recorded $17,666 / −$6,776 / $24,306 |
| Lift | PR-AUC, P@5K, P@10K over the 1.24% holdout churn rate | Match the documented 43.4× / 21.6× / 14.5× to one decimal |
| Calibration *(info)* | Observed churn rate per tier vs the tier's score range | Reported, not gated (see below) |
| Parity | `/predict` on the 5 reference records in `reference_records.json` | Within 0.001 of the recorded champion score |
| Contract | Health, 5 explanations, valid `action`, empty-record warning, threshold 1.5 → 422, top-K on single → 400, bad batch body → 422, empty batch → 400, top-K without k → 400 | All as stated |
| Latency | 50 sequential `/predict` calls with explanations; the same without; one 1,000-record batch | `/predict` p95 ≤ 500 ms; batch returns 200 |

Latency is client-measured. Local runs exclude the network; remote runs include it and any cold start (the harness calls `/health` first to absorb one).

## Latest results

Local run `20261016T060826Z_local.json`, API 1.1.0: **26 checks, 0 failed.**

| Metric | Value |
|---|---:|
| Net value, threshold-only / top-10K / tiered | $17,666 / −$6,776 / $24,306 |
| Lift, PR-AUC / P@5K / P@10K | 43.4× / 21.6× / 14.5× |
| Reference-record parity | 5 of 5 within 0.001 |
| `/predict` with explanations, p50 / p95 | 51.3 ms / 52.7 ms |
| `/predict` without explanations, p50 | 3.0 ms |
| 1,000-record batch | 27.1 ms |

## Known finding: scores are not calibrated

| Tier | Score range | Observed churn on validation |
|---|---|---:|
| `contact_now` | ≥ 0.68 | 70.6% |
| `nudge` | 0.2058–0.68 | **8.9%** |
| `monitor` | < 0.2058 | 0.33% |

The nudge tier churns far below its scores. The model was trained at 6.4% prevalence and evaluated at 1.24%, so scores rank well but overstate risk in the middle. The economics are unaffected because they use observed precision per tier, not raw scores. Calibrate before treating a score as a probability ([decision D7](../docs/decisions.md#d7-scores-are-rankings-not-calibrated-probabilities)). The check stays in the harness as an info-level item, so a recalibrated model will show it passing.

## Adding a check

Add a function returning `_check(name, passed, **details)` items to `run.py` and include it in `main()`. Use `severity="info"` for findings you want tracked but not gated.
