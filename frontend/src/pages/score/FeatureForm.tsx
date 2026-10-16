import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { CustomerRecord } from '@/data/samples';
import { FEATURE_GROUPS, GENDERS, type FeatureDef } from '@/data/features';
import { fromYmdInt, toYmdInt } from '@/lib/dates';

interface Props {
  record: CustomerRecord;
  onChange: (r: CustomerRecord) => void;
}

function Field({ f, value, onChange }: { f: FeatureDef; value: CustomerRecord[string] | undefined; onChange: (v: CustomerRecord[string] | undefined) => void }) {
  const id = `feat-${f.key}`;
  return (
    <div>
      <label htmlFor={id} className="block truncate text-[11px] text-zinc-400" title={f.key}>
        {f.label} <span className="font-mono text-zinc-600">{f.key}</span>
      </label>
      {f.kind === 'gender' ? (
        <select id={id} className="input mt-1 py-1.5" value={String(value ?? 'unknown')} onChange={(e) => onChange(e.target.value)}>
          {GENDERS.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      ) : f.kind === 'date' ? (
        <input
          id={id}
          type="date"
          className="input mt-1 py-1.5"
          value={fromYmdInt(value)}
          onChange={(e) => onChange(toYmdInt(e.target.value) ?? undefined)}
        />
      ) : (
        <input
          id={id}
          type="number"
          step="any"
          className="input mt-1 py-1.5"
          value={value === undefined ? '' : String(value)}
          onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
        />
      )}
    </div>
  );
}

export function FeatureForm({ record, onChange }: Props) {
  const [open, setOpen] = useState<string>('profile');
  const set = (key: string, v: CustomerRecord[string] | undefined) => {
    const next = { ...record };
    if (v === undefined || (typeof v === 'number' && !Number.isFinite(v))) delete next[key];
    else next[key] = v;
    onChange(next);
  };

  return (
    <div className="space-y-2">
      {FEATURE_GROUPS.map((g) => {
        const isOpen = open === g.id;
        const filled = g.features.filter((f) => record[f.key] !== undefined).length;
        return (
          <div key={g.id} className="overflow-hidden rounded-lg border border-zinc-800">
            <button
              type="button"
              onClick={() => setOpen(isOpen ? '' : g.id)}
              aria-expanded={isOpen}
              className="flex w-full items-center justify-between bg-zinc-900/80 px-4 py-2.5 text-left text-sm font-medium text-zinc-200 hover:bg-zinc-800/80"
            >
              {g.title}
              <span className="flex items-center gap-2 text-xs text-zinc-500 tabular-nums">
                {filled}/{g.features.length}
                <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
              </span>
            </button>
            {isOpen && (
              <div className="grid animate-fade-in grid-cols-1 gap-3 p-4 sm:grid-cols-2">
                {g.features.map((f) => (
                  <Field key={f.key} f={f} value={record[f.key]} onChange={(v) => set(f.key, v)} />
                ))}
              </div>
            )}
          </div>
        );
      })}
      <p className="text-[11px] text-zinc-500">Blank fields are sent as missing (the server treats them as NaN). Dates are sent as YYYYMMDD integers.</p>
    </div>
  );
}
