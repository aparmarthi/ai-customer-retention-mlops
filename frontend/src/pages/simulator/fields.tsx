import { useId } from 'react';
import { Lock } from 'lucide-react';
import { InfoTip } from '@/components/ui/InfoTip';

interface RangeFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display: string;
  prefix?: string;
  suffix?: string;
  inputScale?: number;
  hint?: string;
}

export function RangeField({ label, value, min, max, step, onChange, display, prefix, suffix, inputScale = 1, hint }: RangeFieldProps) {
  const id = useId();
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={id} className="flex items-center gap-1.5 text-xs font-medium text-zinc-300">
          {label}
          {hint && <InfoTip content={hint} />}
        </label>
        <div className="flex items-center rounded-md border border-zinc-800 bg-zinc-950 focus-within:border-emerald-500/60">
          {prefix && <span className="pl-2 text-xs text-zinc-500">{prefix}</span>}
          <input
            type="number"
            aria-label={`${label} value`}
            value={Number((value * inputScale).toFixed(4))}
            min={min * inputScale}
            max={max * inputScale}
            step={step * inputScale}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && e.target.value !== '') onChange(v / inputScale);
            }}
            className="w-20 bg-transparent px-2 py-1 text-right text-xs tabular-nums text-zinc-100 focus:outline-none"
          />
          {suffix && <span className="pr-2 text-xs text-zinc-500">{suffix}</span>}
        </div>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-valuetext={display}
        className="mt-3"
      />
      <div className="mt-1 flex justify-between text-[10px] text-zinc-600 tabular-nums">
        <span>{prefix}{min * inputScale}{suffix}</span>
        <span>{prefix}{max * inputScale}{suffix}</span>
      </div>
    </div>
  );
}

export function LockedChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2">
      <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
        <InfoTip label={`${label} is derived`} content="Derived from the model's ranking — not adjustable.">
          <Lock className="h-3 w-3" />
        </InfoTip>
        {label}
      </span>
      <span className="text-sm font-medium tabular-nums text-zinc-200">{value}</span>
    </div>
  );
}
