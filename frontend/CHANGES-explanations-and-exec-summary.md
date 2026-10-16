# Score explanations + executive summary

Copy these files over the matching paths in `frontend/`. No new dependencies.

## New files
- `src/lib/report.ts`: builds the executive summary data and the Markdown export
- `src/lib/report.test.ts`
- `src/lib/demoScore.test.ts`
- `src/pages/score/RiskFactors.tsx`: "Why this score" panel
- `src/pages/simulator/ExecutiveSummary.tsx`: printable report modal (print/PDF, Markdown download, copy)

## Modified files
- `src/lib/demoScore.ts`: rule of thumb split into named terms; adds `explainRecord`, `fmtImpact`, `summarizeFactors`, `BASELINE_RISK`. `demoEstimate` output is unchanged.
- `src/pages/score/risk.ts`: adds a suggested `action` per risk band
- `src/pages/score/SingleScore.tsx`: suggested action and risk drivers panel
- `src/pages/score/BatchResults.tsx`: "Top reasons" column; CSV adds `risk_band`, `suggested_action`, `top_reasons`
- `src/pages/simulator/SimulatorPage.tsx`: "Executive summary" button
- `src/index.css`: print styles for the report
- `README.md`: notes on the new features

## Verify
npm run typecheck && npm test && npm run build
