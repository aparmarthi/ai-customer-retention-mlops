import { useId } from 'react';
import { RotateCcw } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { BOUNDS } from '@/lib/scenario';
import { VALID } from '@/data/validation';
import { knownThreshold, precisionAtK, recallAtK, scale } from '@/lib/economics';
import { fmtInt, fmtPct, fmtUsd, fmtUsd2 } from '@/lib/format';
import { Card } from '@/components/ui/primitives';
import { SubscriberInput } from '@/components/layout/SubscriberInput';
import { LockedChip, RangeField } from '@/pages/simulator/fields';

const SNAPS = [1262, 1478, 5000, 10000, 20000];
const K_MIN = BOUNDS.k.min;
const K_MAX = BOUNDS.k.max;
const LN_MIN = Math.log(K_MIN);
const LN_MAX = Math.log(K_MAX);
const toPos = (k: number) => Math.round(((Math.log(k) - LN_MIN) / (LN_MAX - LN_MIN)) * 1000);
const fromPos = (p: number) => Math.round(Math.exp(LN_MIN + (p / 1000) * (LN_MAX - LN_MIN)));

export function SimulatorInputs() {
  const a = useAssumptions();
  const kId = useId();
  const threshold = knownThreshold(a.k);

  return (
    <Card
      title="Scenario inputs"
      subtitle="Business assumptions are yours; precision is the model's."
      action={
        <button onClick={a.reset} className="btn-ghost px-2 py-1 text-xs" aria-label="Reset to defaults">
          <RotateCcw className="h-3.5 w-3.5" /> Reset
        </button>
      }
      bodyClassName="space-y-6 p-5"
    >
      <div>
        <div className="flex items-center justify-between gap-3">
          <label htmlFor={kId} className="text-xs font-medium text-zinc-300">K contacts per cycle (validation window)</label>
          <input
            type="number"
            aria-label="K contacts value"
            min={K_MIN}
            max={K_MAX}
            value={a.k}
            onChange={(e) => e.target.value !== '' && a.setK(Number(e.target.value))}
            className="w-24 rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1 text-right text-xs tabular-nums text-zinc-100 focus:border-emerald-500/60 focus:outline-none"
          />
        </div>
        <input
          id={kId}
          type="range"
          min={0}
          max={1000}
          value={toPos(a.k)}
          onChange={(e) => a.setK(fromPos(Number(e.target.value)))}
          aria-valuetext={`${fmtInt(a.k)} contacts`}
          className="mt-3"
        />
        <div className="mt-1 flex justify-between text-[10px] text-zinc-600 tabular-nums">
          <span>100</span><span>1K</span><span>10K</span><span>50K</span>
        </div>
        <p className="mt-2 text-[11px] text-zinc-400 tabular-nums">
          Top {fmtPct(a.k / VALID.n, 2)} of the base = {fmtInt(scale(a.k, a.subscribers))} contacts at {fmtInt(a.subscribers)} subscribers.
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5" role="group" aria-label="Snap K to a known operating point">
          {SNAPS.map((s) => (
            <button
              key={s}
              onClick={() => a.setK(s)}
              aria-pressed={a.k === s}
              className={`rounded-md border px-2 py-1 text-[11px] font-medium tabular-nums transition-colors ${
                a.k === s ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-300' : 'border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
              }`}
            >
              {fmtInt(s)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3 lg:grid-cols-1">
        <LockedChip label="Precision" value={fmtPct(precisionAtK(a.k))} />
        <LockedChip label="Recall" value={fmtPct(recallAtK(a.k))} />
        <LockedChip label="Threshold" value={threshold !== undefined ? String(threshold) : '—'} />
      </div>

      <div className="space-y-5 border-t border-zinc-800/80 pt-5">
        <RangeField label="Value per save (V)" value={a.valuePerSave} min={10} max={500} step={1} onChange={a.setValuePerSave} display={fmtUsd(a.valuePerSave)} prefix="$" hint="Revenue retained when a churner is saved (roughly remaining lifetime value)." />
        <RangeField label="Cost per contact (C)" value={a.costPerContact} min={0.1} max={50} step={0.1} onChange={a.setCostPerContact} display={fmtUsd2(a.costPerContact)} prefix="$" hint="Incentive plus handling cost for every contacted subscriber, churner or not." />
        <RangeField label="Save rate (S)" value={a.saveRate} min={0.01} max={0.6} step={0.01} onChange={a.setSaveRate} display={fmtPct(a.saveRate, 0)} suffix="%" inputScale={100} hint="Share of contacted true churners who stay. Assumed, not measured — the #1 risk." />
      </div>

      <div className="border-t border-zinc-800/80 pt-5">
        <SubscriberInput id="sim-subscriber-base" />
      </div>
    </Card>
  );
}
