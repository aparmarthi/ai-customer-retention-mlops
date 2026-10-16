import { useId, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';

interface InfoTipProps {
  content: ReactNode;
  label?: string;
  children?: ReactNode;
  side?: 'top' | 'bottom';
  align?: 'center' | 'start' | 'end';
}

export function InfoTip({ content, label = 'More information', children, side = 'top', align = 'center' }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const pos = side === 'top' ? 'bottom-full mb-2' : 'top-full mt-2';
  const al = align === 'center' ? 'left-1/2 -translate-x-1/2' : align === 'start' ? 'left-0' : 'right-0';
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center text-zinc-500 transition-colors hover:text-zinc-300"
      >
        {children ?? <Info className="h-3.5 w-3.5" />}
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className={`absolute z-50 w-64 animate-fade-in rounded-lg border border-zinc-700 bg-zinc-900 p-3 text-left text-xs font-normal normal-case leading-relaxed tracking-normal text-zinc-300 shadow-2xl shadow-black/60 ${pos} ${al}`}
        >
          {content}
        </span>
      )}
    </span>
  );
}
