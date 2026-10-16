import { FlaskConical } from 'lucide-react';

export const CAVEAT_TEXT =
  'Simulated on the time-based validation window (193,205 subscribers, 1 cycle = 1 month). Not realized revenue.';

export function CaveatChip({ className = '' }: { className?: string }) {
  return (
    <p className={`inline-flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/70 px-3 py-2 text-[11px] text-zinc-400 ${className}`}>
      <FlaskConical className="mt-px h-3.5 w-3.5 shrink-0 text-amber-400/80" aria-hidden />
      <span>{CAVEAT_TEXT}</span>
    </p>
  );
}
