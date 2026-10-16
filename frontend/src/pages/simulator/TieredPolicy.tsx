import { useState } from 'react';
import { Layers } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { TIERS, tieredNet } from '@/lib/economics';
import { fmtInt, fmtPct, fmtUsd, fmtUsd2 } from '@/lib/format';
import { Badge, Card } from '@/components/ui/primitives';

export function TieredPolicy() {
  const { valuePerSave: V } = useAssumptions();
  const [enabled, setEnabled] = useState(false);
  const t = tieredNet(V);

  return (
    <Card
      title="Tiered policy"
      subtitle="Spend more where risk is highest, nudge cheaply below it."
      icon={<Layers className="h-4 w-4" />}
      action={
        <button
          role="switch"
          aria-checked={enabled}
          aria-label="Toggle tiered policy"
          onClick={() => setEnabled((e) => !e)}
          className={`relative h-6 w-11 rounded-full transition-colors ${enabled ? 'bg-emerald-500' : 'bg-zinc-700'}`}
        >
          <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${enabled ? 'left-[22px]' : 'left-0.5'}`} />
        </button>
      }
    >
      {!enabled ? (
        <p className="text-sm text-zinc-400">Turn on to compare a two-tier contact plan against the single-tier ROI-max policy.</p>
      ) : (
        <div className="animate-fade-in space-y-4">
          <Badge tone="warn">Illustrative — tier-2 save rate is an assumption</Badge>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
              <p className="label">Tier 1 · ranks 1–{fmtInt(TIERS.tier1.k)}</p>
              <p className="mt-2 text-xs text-zinc-400 tabular-nums">{fmtUsd2(TIERS.tier1.cost)} incentive · {fmtPct(TIERS.tier1.saveRate, 0)} save rate</p>
              <p className="mt-2 text-lg font-semibold tabular-nums text-emerald-300">{fmtUsd(t.net1)}</p>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
              <p className="label">Tier 2 · ranks {fmtInt(TIERS.tier1.k + 1)}–{fmtInt(TIERS.tier2.kEnd)}</p>
              <p className="mt-2 text-xs text-zinc-400 tabular-nums">{fmtUsd2(TIERS.tier2.cost)} email/push · {fmtPct(TIERS.tier2.saveRate, 0)} save rate</p>
              <p className={`mt-2 text-lg font-semibold tabular-nums ${t.net2 >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{fmtUsd(t.net2)}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-baseline justify-between gap-2 rounded-lg border border-emerald-500/25 bg-emerald-500/[0.05] p-4">
            <span className="text-sm text-zinc-300">Combined net / cycle (validation scale)</span>
            <span className="tabular-nums">
              <span className="text-xl font-semibold text-emerald-300">{fmtUsd(t.combined)}</span>
              <span className="ml-2 text-xs text-zinc-400">{t.upliftVsSingle >= 0 ? '+' : ''}{fmtPct(t.upliftVsSingle)} vs single-tier</span>
            </span>
          </div>
          <p className="text-[11px] text-zinc-500">Uses your value per save ({fmtUsd(V)}); tier costs and save rates are fixed illustrative values.</p>
        </div>
      )}
    </Card>
  );
}
