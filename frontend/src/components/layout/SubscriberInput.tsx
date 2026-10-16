import { useEffect, useState } from 'react';
import { Users } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { clampTo } from '@/lib/scenario';
import { fmtInt } from '@/lib/format';
import { InfoTip } from '@/components/ui/InfoTip';

export function SubscriberInput({ id = 'subscriber-base', compact = false }: { id?: string; compact?: boolean }) {
  const { subscribers, setSubscribers } = useAssumptions();
  const [draft, setDraft] = useState(fmtInt(subscribers));

  useEffect(() => setDraft(fmtInt(subscribers)), [subscribers]);

  const commit = () => {
    const raw = draft.trim().toLowerCase().replace(/[, ]/g, '');
    const m = /^(\d*\.?\d+)([km]?)$/.exec(raw);
    if (!m) {
      setDraft(fmtInt(subscribers));
      return;
    }
    const mult = m[2] === 'm' ? 1e6 : m[2] === 'k' ? 1e3 : 1;
    const v = Number(m[1]) * mult;
    setSubscribers(v);
    setDraft(fmtInt(clampTo('subscribers', v)));
  };

  return (
    <div className="flex items-center gap-2">
      <label htmlFor={id} className={`flex items-center gap-1.5 text-xs text-zinc-500 ${compact ? 'sr-only' : ''}`}>
        <Users className="h-3.5 w-3.5" aria-hidden />
        Subscriber base
      </label>
      <input
        id={id}
        inputMode="numeric"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && (e.currentTarget as HTMLInputElement).blur()}
        className="w-28 rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1 text-right text-xs tabular-nums text-zinc-100 focus:border-emerald-500/60 focus:outline-none focus:ring-2 focus:ring-emerald-500/20"
      />
      <InfoTip
        align="end"
        side="bottom"
        content="Scales every business-value figure from the 193,205-subscriber validation window to your base (100K to 50M). Accepts shorthand like 2.5m or 800k."
      />
    </div>
  );
}
