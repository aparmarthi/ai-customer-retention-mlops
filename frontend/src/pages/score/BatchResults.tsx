import { useMemo } from 'react';
import { Download, Timer } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useAssumptions } from '@/context/AssumptionsContext';
import { DEFAULT_THRESHOLD } from '@/data/validation';
import { downloadText, toCsv } from '@/lib/csv';
import { explainRecord, fmtImpact, summarizeFactors } from '@/lib/demoScore';
import { fmtInt, fmtMs, fmtUsd } from '@/lib/format';
import { Badge, Card, Stat } from '@/components/ui/primitives';
import { CaveatChip } from '@/components/ui/CaveatChip';
import type { BatchRun } from '@/pages/score/batch';
import { RISK_BANDS, riskBand } from '@/pages/score/risk';

const TABLE_LIMIT = 100;
const BINS = 20;

export function BatchResults({ run }: { run: BatchRun }) {
  const { valuePerSave: V, costPerContact: C, saveRate: S } = useAssumptions();
  const n = run.rows.length;
  const topK = run.rows.filter((r) => r.inTopK);
  const above = run.rows.filter((r) => r.probability >= DEFAULT_THRESHOLD).length;
  const expectedSaves = topK.reduce((s, r) => s + r.probability * S, 0);
  const expectedNet = expectedSaves * V - topK.length * C;
  const cutoff = topK.length ? topK.reduce((m, r) => Math.min(m, r.probability), Infinity) : null;
  const inputCols = run.headers.slice(0, 4);

  const hist = useMemo(() => {
    const bins = Array.from({ length: BINS }, (_, i) => ({ x: (i + 0.5) / BINS, count: 0 }));
    run.rows.forEach((r) => {
      bins[Math.min(BINS - 1, Math.floor(r.probability * BINS))].count++;
    });
    return bins;
  }, [run.rows]);

  const sorted = useMemo(
    () => [...run.rows].sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity) || b.probability - a.probability).slice(0, TABLE_LIMIT),
    [run.rows],
  );

  const reasons = useMemo(() => new Map(sorted.map((r) => [r.index, explainRecord(r.input, 2).factors])), [sorted]);

  const exportCsv = () => {
    const headers = ['row_index', 'rank', 'churn_probability', 'label_at_0_68', 'in_top_k', 'risk_band', 'suggested_action', 'top_reasons', ...run.headers];
    const rows = run.rows.map((r) => {
      const band = RISK_BANDS[riskBand(r.probability)];
      return {
        row_index: r.index,
        rank: r.rank ?? '',
        churn_probability: r.probability,
        label_at_0_68: r.labelAt068,
        in_top_k: r.inTopK ? 1 : 0,
        risk_band: band.label,
        suggested_action: band.action,
        top_reasons: summarizeFactors(explainRecord(r.input, 3).factors),
        ...r.input,
      };
    });
    downloadText(`churn_scores_top${run.k}.csv`, toCsv(headers, rows));
  };

  return (
    <div className="animate-fade-in space-y-6">
      <Card
        title="Batch summary"
        subtitle={`${run.origin} · policy ${run.policyUsed}${run.thresholdUsed !== null ? ` · equivalent cutoff ${run.thresholdUsed}` : ''}`}
        action={
          <div className="flex items-center gap-2">
            {run.source === 'estimate' ? (
              <Badge tone="warn">demo estimate · not the model</Badge>
            ) : run.source === 'demo' ? (
              <Badge tone="warn">demo</Badge>
            ) : (
              run.latencyMs !== null && <Badge><Timer className="h-3 w-3" />{fmtMs(run.latencyMs)}</Badge>
            )}
            <button onClick={exportCsv} className="btn-secondary px-2.5 py-1.5 text-xs"><Download className="h-3.5 w-3.5" /> Export CSV</button>
          </div>
        }
      >
        <div className="grid grid-cols-2 gap-6 md:grid-cols-5">
          <Stat label="Scored" value={fmtInt(n)} />
          <Stat label="≥ 0.68" value={fmtInt(above)} tone="risk" />
          <Stat label={`In top-K (K=${fmtInt(run.k)})`} value={fmtInt(topK.length)} />
          <Stat label="Expected saves" value={expectedSaves.toFixed(1)} sub={`Σ p × ${Math.round(S * 100)}% over top-K`} />
          <Stat label="Expected net" value={fmtUsd(expectedNet)} tone={expectedNet >= 0 ? 'value' : 'risk'} sub={`V ${fmtUsd(V)} · C $${C.toFixed(2)}`} />
        </div>
        <CaveatChip className="mt-5" />
      </Card>

      <Card title="Probability distribution" subtitle="Customers per probability bin, with the 0.68 threshold and the top-K cutoff">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hist} margin={{ top: 16, right: 12, left: 0, bottom: 0 }} barCategoryGap={1}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="x" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tick={{ fill: '#71717a', fontSize: 11 }} axisLine={{ stroke: '#3f3f46' }} tickLine={false} />
              <YAxis allowDecimals={false} tick={{ fill: '#71717a', fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
                labelFormatter={(x: number) => `p ≈ ${(x - 0.5 / BINS).toFixed(2)}–${(x + 0.5 / BINS).toFixed(2)}`}
                formatter={(v: number) => [fmtInt(v), 'customers']}
              />
              <Bar dataKey="count" fill="#71717a" radius={[2, 2, 0, 0]} />
              <ReferenceLine x={DEFAULT_THRESHOLD} stroke="#fb7185" strokeDasharray="4 3" label={{ value: '0.68', fill: '#fb7185', fontSize: 10, position: 'top' }} />
              {cutoff !== null && (
                <ReferenceLine x={cutoff} stroke="#34d399" strokeDasharray="1 3" strokeWidth={2} label={{ value: `top-K ${cutoff.toFixed(3)}`, fill: '#34d399', fontSize: 10, position: 'insideTopLeft' }} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Scored customers" subtitle={n > TABLE_LIMIT ? `Top ${TABLE_LIMIT} by rank of ${fmtInt(n)} — export for the full list` : 'Sorted by rank'} bodyClassName="p-0">
        <div className="max-h-[480px] overflow-auto">
          <table className="w-full min-w-[960px] text-sm">
            <thead className="sticky top-0 bg-zinc-900">
              <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500">
                <th scope="col" className="px-4 py-2.5 font-medium">Rank</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Probability</th>
                <th scope="col" className="px-4 py-2.5 text-center font-medium">Label @0.68</th>
                <th scope="col" className="px-4 py-2.5 text-center font-medium">Top-K</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Top reasons <span className="font-normal text-zinc-600">(indicative)</span></th>
                {inputCols.map((h) => (
                  <th key={h} scope="col" className="px-4 py-2.5 font-mono text-[11px] font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.index} className={`border-b border-zinc-800/60 last:border-0 ${r.inTopK ? 'bg-emerald-500/[0.04]' : ''}`}>
                  <td className="px-4 py-2 tabular-nums text-zinc-300">{r.rank ?? '—'}</td>
                  <td className="px-4 py-2 text-right font-medium tabular-nums text-zinc-100">{r.probability.toFixed(4)}</td>
                  <td className="px-4 py-2 text-center">
                    <span className={`inline-block rounded px-1.5 text-xs tabular-nums ${r.labelAt068 ? 'bg-rose-500/15 text-rose-300' : 'text-zinc-500'}`}>{r.labelAt068}</span>
                  </td>
                  <td className="px-4 py-2 text-center text-xs">{r.inTopK ? <span className="text-emerald-300">Yes</span> : <span className="text-zinc-600">No</span>}</td>
                  <td className="px-4 py-2">
                    <div className="flex flex-wrap gap-1">
                      {(reasons.get(r.index) ?? []).map((f) => (
                        <span
                          key={f.id}
                          className={`whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] ${f.impact > 0 ? 'border-rose-500/25 bg-rose-500/[0.07] text-rose-200' : 'border-emerald-500/25 bg-emerald-500/[0.07] text-emerald-200'}`}
                        >
                          {f.text} <span className="tabular-nums opacity-80">{fmtImpact(f.impact)}</span>
                        </span>
                      ))}
                    </div>
                  </td>
                  {inputCols.map((h) => (
                    <td key={h} className="px-4 py-2 tabular-nums text-zinc-400">{String(r.input[h] ?? '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
