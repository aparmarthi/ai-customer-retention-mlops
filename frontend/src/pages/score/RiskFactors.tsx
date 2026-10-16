import { ArrowDownRight, ArrowUpRight, Lightbulb } from 'lucide-react';
import { BASELINE_RISK, fmtImpact, type Explanation } from '@/lib/demoScore';
import { fmtPct, fmtProb } from '@/lib/format';
import { InfoTip } from '@/components/ui/InfoTip';

interface RiskFactorsProps {
  explanation: Explanation;
  /** Shown when the headline score came from somewhere other than the rule of thumb. */
  modelScore?: number;
}

export function RiskFactors({ explanation, modelScore }: RiskFactorsProps) {
  const { factors, estimate } = explanation;
  const max = Math.max(0.01, ...factors.map((f) => Math.abs(f.impact)));

  return (
    <section aria-labelledby="why-score" className="w-full rounded-lg border border-zinc-800 bg-zinc-950/50 p-4 text-left">
      <header className="flex items-center justify-between gap-2">
        <h3 id="why-score" className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
          <Lightbulb className="h-3.5 w-3.5 text-amber-400" aria-hidden />
          Why this score
        </h3>
        <InfoTip
          label="How these factors are calculated"
          align="end"
          content={`Points show how much each factor moves the rule-of-thumb estimate away from a ${fmtPct(BASELINE_RISK)} baseline. They are indicative drivers, not the trained model's own attributions.`}
        />
      </header>

      {factors.length === 0 ? (
        <p className="mt-3 text-xs text-zinc-500">Not enough billing or listening detail to explain this score.</p>
      ) : (
        <ul className="mt-3 space-y-2.5">
          {factors.map((f) => {
            const up = f.impact > 0;
            return (
              <li key={f.id}>
                <div className="flex items-baseline justify-between gap-3 text-xs">
                  <span className="flex min-w-0 items-center gap-1.5 text-zinc-300">
                    {up ? <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-rose-400" aria-hidden /> : <ArrowDownRight className="h-3.5 w-3.5 shrink-0 text-emerald-400" aria-hidden />}
                    <span className="truncate" title={f.text}>{f.text}</span>
                  </span>
                  <span className={`shrink-0 font-medium tabular-nums ${up ? 'text-rose-300' : 'text-emerald-300'}`}>
                    {fmtImpact(f.impact)}
                    <span className="sr-only">{up ? ' raises risk' : ' lowers risk'}</span>
                  </span>
                </div>
                <div className="mt-1 h-1 overflow-hidden rounded-full bg-zinc-800" aria-hidden>
                  <div
                    className={`h-full rounded-full transition-[width] duration-500 ${up ? 'bg-rose-400/80' : 'bg-emerald-400/80'}`}
                    style={{ width: `${(Math.abs(f.impact) / max) * 100}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <p className="mt-3 border-t border-zinc-800/80 pt-2.5 text-[11px] leading-relaxed text-zinc-500">
        Indicative drivers from the transparent rule of thumb
        {modelScore !== undefined && Math.abs(modelScore - estimate) >= 0.001 && (
          <>
            {' '}(which estimates <span className="tabular-nums text-zinc-400">{fmtProb(estimate)}</span> vs the model&apos;s{' '}
            <span className="tabular-nums text-zinc-400">{fmtProb(modelScore)}</span>)
          </>
        )}
        . Per-customer model attributions require the API to return them.
      </p>
    </section>
  );
}
