import { AlertTriangle, TrendingUp } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { VALID } from '@/data/validation';
import { scale, type Scenario } from '@/lib/economics';
import { fmtCompact, fmtInt, fmtMultiple, fmtPct, fmtUsd, fmtUsd2, fmtUsdCompact } from '@/lib/format';
import { CaveatChip } from '@/components/ui/CaveatChip';

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className={`text-sm tabular-nums ${strong ? 'font-semibold text-zinc-50' : 'text-zinc-200'}`}>{value}</dd>
    </div>
  );
}

export function ResultsPanel({ s }: { s: Scenario }) {
  const { subscribers, costPerContact: C, saveRate: S } = useAssumptions();
  const at = (x: number) => scale(x, subscribers);
  const negative = s.net < 0;
  const breakEvenV = s.tp > 0 && S > 0 ? (s.k * C) / (s.tp * S) : Infinity;
  const tone = negative ? 'text-rose-300' : 'text-emerald-300';

  return (
    <section
      aria-live="polite"
      className={`rounded-xl border p-5 transition-colors duration-300 ${negative ? 'border-rose-500/40 bg-rose-500/[0.06]' : 'border-emerald-500/25 bg-emerald-500/[0.04]'}`}
    >
      <div className="flex items-center gap-2">
        {negative ? <AlertTriangle className="h-4 w-4 text-rose-400" /> : <TrendingUp className="h-4 w-4 text-emerald-400" />}
        <h2 className="text-sm font-semibold text-zinc-100">
          Contact top {fmtPct(s.k / VALID.n, 2)} · {fmtInt(at(s.k))} of {fmtCompact(subscribers)} subs
        </h2>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-zinc-500">Net value / cycle</p>
          <p className={`mt-1 text-2xl font-semibold tabular-nums ${tone}`}>{fmtUsdCompact(at(s.net))}</p>
          <p className="text-[11px] text-zinc-500 tabular-nums">{fmtUsdCompact(at(s.net) * 12)}/yr</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Net ROI (net ÷ spend)</p>
          <p className={`mt-1 text-2xl font-semibold tabular-nums ${tone}`}>{fmtPct(s.netRoi, 0)}</p>
          <p className="text-[11px] text-zinc-500 tabular-nums">{fmtUsd2(s.returnOnSpend)} back per $1 spent</p>
        </div>
      </div>

      {negative && (
        <p className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          At {fmtUsd2(C)}/contact, each save must be worth &gt; {Number.isFinite(breakEvenV) ? fmtUsd2(breakEvenV) : '—'}.
        </p>
      )}

      <dl className="mt-4 divide-y divide-zinc-800/80 border-t border-zinc-800/80">
        <Row label="Churners reached" value={`${fmtInt(at(s.tp))} of ${fmtInt(at(VALID.churners))}`} />
        <Row label="Precision (hit rate)" value={fmtPct(s.precision)} />
        <Row label="Recall" value={fmtPct(s.recall)} />
        <Row label="Value retained" value={fmtUsd(at(s.valueSaved))} />
        <Row label="Contact spend" value={fmtUsd(at(s.spend))} />
        <Row label="Return on spend (gross)" value={fmtMultiple(s.returnOnSpend)} strong />
        <Row label="Break-even save rate" value={fmtPct(s.breakEvenSaveRate)} />
        <Row label="Break-even contact cost" value={fmtUsd2(s.breakEvenCost)} />
        <Row label="Random targeting, same contacts" value={fmtUsd(at(s.randomNet))} />
      </dl>
      <p className="mt-4 text-[11px] leading-relaxed text-zinc-500 tabular-nums">
        Measured on the validation window ({fmtInt(VALID.n)} subs): {fmtInt(s.k)} contacts, {fmtInt(s.tp)} churners reached, net {fmtUsd(s.net)}/cycle.
        Your figures scale this linearly (×{(subscribers / VALID.n).toFixed(2)}).
      </p>
      <CaveatChip className="mt-3" />
    </section>
  );
}
