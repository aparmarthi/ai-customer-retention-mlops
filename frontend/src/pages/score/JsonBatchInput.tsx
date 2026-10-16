import { useId } from 'react';
import { AlertCircle, Braces, CheckCircle2, Eraser, RotateCcw } from 'lucide-react';
import { EXAMPLE_BATCH_SIZE } from '@/data/profiles';
import { fmtInt } from '@/lib/format';
import type { JsonParseResult } from '@/pages/score/jsonBatch';

interface Props {
  text: string;
  parsed: JsonParseResult;
  onChange: (text: string) => void;
  onLoadExample: () => void;
}

export function JsonBatchInput({ text, parsed, onChange, onLoadExample }: Props) {
  const id = useId();
  const statusId = `${id}-status`;
  const empty = text.trim() === '';

  return (
    <div>
      <div className={`overflow-hidden rounded-xl border transition-colors ${empty || parsed.ok ? 'border-zinc-800 focus-within:border-emerald-500/50' : 'border-rose-500/40'}`}>
        <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/60 px-3 py-2">
          <label htmlFor={id} className="flex items-center gap-2 text-xs text-zinc-400">
            <Braces className="h-3.5 w-3.5" /> JSON records
          </label>
          <span className="text-[11px] text-zinc-500">Array or {'{"records": [...]}'}</span>
        </div>
        <textarea
          id={id}
          value={text}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          aria-describedby={statusId}
          aria-invalid={!empty && !parsed.ok}
          placeholder={'[\n  { "city": 1, "bd": 28, "gender": "female", "txn_cnt": 12, ... },\n  { "city": 13, "bd": 0, "gender": "unknown", "txn_cnt": 3, ... }\n]'}
          className="block h-[220px] w-full resize-y bg-zinc-950/60 p-3 font-mono text-xs leading-relaxed text-zinc-200 placeholder:text-zinc-600 focus:outline-none"
        />
      </div>
      <p id={statusId} className="mt-2 flex flex-wrap items-start gap-1.5 text-[11px] tabular-nums" aria-live="polite">
        {empty ? (
          <span className="text-zinc-500">Missing feature fields are treated as missing; extra fields are ignored.</span>
        ) : parsed.ok ? (
          <>
            <CheckCircle2 className="mt-px h-3.5 w-3.5 shrink-0 text-emerald-400" />
            <span className="text-emerald-300">
              {fmtInt(parsed.records.length)} records · {parsed.headers.length} fields
            </span>
            <span className="text-zinc-500">· tip: set K below the record count to see who makes the cut</span>
          </>
        ) : (
          <>
            <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0 text-rose-400" />
            <span className="break-words text-rose-300">{parsed.error}</span>
          </>
        )}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <button onClick={onLoadExample} className="btn-ghost text-xs">
          <RotateCcw className="h-3.5 w-3.5" /> Reset to {EXAMPLE_BATCH_SIZE}-customer example
        </button>
        {!empty && (
          <button onClick={() => onChange('')} className="btn-ghost text-xs">
            <Eraser className="h-3.5 w-3.5" /> Clear
          </button>
        )}
      </div>
    </div>
  );
}
