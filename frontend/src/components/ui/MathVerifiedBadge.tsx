import { useMemo } from 'react';
import { Check, ShieldCheck, ShieldAlert, X } from 'lucide-react';
import { InfoTip } from '@/components/ui/InfoTip';
import { runSanityChecks } from '@/lib/sanity';

export function MathVerifiedBadge() {
  const results = useMemo(runSanityChecks, []);
  const passed = results.filter((r) => r.pass).length;
  const ok = passed === results.length;
  const Icon = ok ? ShieldCheck : ShieldAlert;

  const content = (
    <span className="block">
      <span className="mb-2 block text-zinc-200">
        {ok
          ? 'Every documented sanity value is recomputed live in your browser and matches.'
          : 'Some documented sanity values no longer match. Treat the numbers with caution.'}
      </span>
      <span className="block max-h-56 space-y-1 overflow-y-auto pr-1">
        {results.map((r) => (
          <span key={r.label} className="flex items-start gap-1.5">
            {r.pass ? (
              <Check className="mt-0.5 h-3 w-3 shrink-0 text-emerald-400" />
            ) : (
              <X className="mt-0.5 h-3 w-3 shrink-0 text-rose-400" />
            )}
            <span className={r.pass ? 'text-zinc-400' : 'text-rose-300'}>{r.label}</span>
          </span>
        ))}
      </span>
    </span>
  );

  return (
    <InfoTip content={content} label="Math verification details" side="bottom" align="end">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium tabular-nums transition-colors ${
          ok
            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/15'
            : 'border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/15'
        }`}
      >
        <Icon className="h-3.5 w-3.5" />
        Math verified · {passed}/{results.length}
      </span>
    </InfoTip>
  );
}
