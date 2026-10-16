import { AlertTriangle } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { sensitivity } from '@/lib/economics';
import { fmtSignedUsdCompact, fmtUsd } from '@/lib/format';
import { Card } from '@/components/ui/primitives';

export function SensitivityTornado() {
  const { k, valuePerSave, costPerContact, saveRate } = useAssumptions();
  const { base, bars } = sensitivity(k, { valuePerSave, costPerContact, saveRate });
  const maxAbs = Math.max(1, ...bars.flatMap((b) => [Math.abs(b.low), Math.abs(b.high)]));
  const pct = (v: number) => (Math.abs(v) / maxAbs) * 50;
  const top = bars[0];

  return (
    <Card title="Sensitivity (±50%)" subtitle={`Change in net vs current ${fmtUsd(base)} / cycle at validation scale`}>
      <div className="space-y-4" role="list">
        {bars.map((b) => {
          const neg = Math.min(b.low, b.high);
          const posv = Math.max(b.low, b.high);
          return (
            <div key={b.driver} role="listitem">
              <div className="mb-1.5 flex items-center justify-between text-xs">
                <span className="font-medium text-zinc-300">{b.driver}</span>
                <span className="tabular-nums text-zinc-500">
                  <span className="text-rose-300">{fmtSignedUsdCompact(neg)}</span> / <span className="text-emerald-300">{fmtSignedUsdCompact(posv)}</span>
                </span>
              </div>
              <div className="relative h-6 rounded bg-zinc-950/80">
                <div className="absolute inset-y-0 left-1/2 w-px bg-zinc-600" />
                {neg < 0 && (
                  <div
                    className="absolute inset-y-1 rounded-l bg-rose-400/70 transition-all duration-300"
                    style={{ right: '50%', width: `${pct(neg)}%`, backgroundImage: 'repeating-linear-gradient(45deg, transparent 0 4px, rgba(0,0,0,0.25) 4px 6px)' }}
                    title={`${b.driver} worse case: ${fmtSignedUsdCompact(neg)}`}
                  />
                )}
                {posv > 0 && (
                  <div
                    className="absolute inset-y-1 rounded-r bg-emerald-400/80 transition-all duration-300"
                    style={{ left: '50%', width: `${pct(posv)}%` }}
                    title={`${b.driver} better case: ${fmtSignedUsdCompact(posv)}`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
      {top && (
        <p className="mt-5 flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/[0.06] px-3 py-2 text-xs text-amber-200">
          <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />
          <span>
            {top.driver} is the largest swing factor at this K{top.driver === 'Save rate' ? ' — and it is assumed, not measured. Validate it with a holdout before committing budget.' : '.'}
          </span>
        </p>
      )}
    </Card>
  );
}
