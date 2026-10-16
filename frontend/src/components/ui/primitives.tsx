import type { ReactNode } from 'react';

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}

export function Card({ title, subtitle, action, icon, className = '', bodyClassName = 'p-5', children }: CardProps) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-zinc-800/80 px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            {icon && <div className="mt-0.5 text-zinc-500">{icon}</div>}
            <div className="min-w-0">
              {title && <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>}
              {subtitle && <p className="mt-1 text-xs text-zinc-500">{subtitle}</p>}
            </div>
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

type Tone = 'neutral' | 'value' | 'risk' | 'warn' | 'info';

const tones: Record<Tone, string> = {
  neutral: 'border-zinc-700/80 bg-zinc-800/60 text-zinc-300',
  value: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
  risk: 'border-rose-500/30 bg-rose-500/10 text-rose-300',
  warn: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
  info: 'border-sky-500/30 bg-sky-500/10 text-sky-300',
};

export function Badge({ tone = 'neutral', children, className = '' }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]} ${className}`}>
      {children}
    </span>
  );
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl">
        <p className="label text-emerald-400/90">{eyebrow}</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-zinc-50 md:text-3xl">{title}</h1>
        {description && <p className="mt-2 text-sm text-zinc-400 md:text-base">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-md bg-zinc-800/60 ${className}`} aria-hidden>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-zinc-700/30 to-transparent" />
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: 'value' | 'risk' }) {
  const color = tone === 'value' ? 'text-emerald-300' : tone === 'risk' ? 'text-rose-300' : 'text-zinc-100';
  return (
    <div className="min-w-0">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className={`mt-1 truncate text-lg font-semibold tabular-nums ${color}`}>{value}</p>
      {sub && <p className="mt-0.5 text-[11px] text-zinc-500 tabular-nums">{sub}</p>}
    </div>
  );
}
