import { useEffect, useMemo, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, CheckCircle2, FileDown, Printer, ShieldCheck, TrendingUp, X } from 'lucide-react';
import { downloadText } from '@/lib/csv';
import { buildExecReport, reportToMarkdown, type ReportItem } from '@/lib/report';
import type { ScenarioState } from '@/lib/scenario';
import { CopyButton } from '@/components/ui/CopyButton';

interface ExecutiveSummaryProps {
  state: ScenarioState;
  link: string;
  onClose: () => void;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="break-inside-avoid">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">{title}</h3>
      {children}
    </section>
  );
}

function ItemList({ items }: { items: ReportItem[] }) {
  return (
    <dl className="divide-y divide-zinc-200">
      {items.map((i) => (
        <div key={i.label} className="flex items-baseline justify-between gap-4 py-2 text-sm">
          <dt className="text-zinc-600">{i.label}</dt>
          <dd className="text-right tabular-nums text-zinc-900">
            {i.value}
            {i.note && <span className="block text-[11px] text-zinc-500">{i.note}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

const BAND_DOT = ['bg-rose-500', 'bg-amber-500', 'bg-emerald-500'];

export function ExecutiveSummary({ state, link, onClose }: ExecutiveSummaryProps) {
  const report = useMemo(() => buildExecReport(state, link), [state, link]);
  const markdown = useMemo(() => reportToMarkdown(report), [report]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const download = () => downloadText(`churn-retention-summary-k${state.k}.md`, markdown, 'text/markdown');

  return createPortal(
    <div className="print-root fixed inset-0 z-[60] animate-fade-in overflow-y-auto bg-black/80 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="exec-title" onClick={onClose}>
      <div className="mx-auto max-w-4xl px-4 py-6 sm:py-10" onClick={(e) => e.stopPropagation()}>
        <div className="no-print mb-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-zinc-300">Executive summary preview</p>
          <div className="flex flex-wrap items-center gap-2">
            <CopyButton text={markdown} label="Copy text" />
            <button onClick={download} className="btn-secondary px-2.5 py-1.5 text-xs"><FileDown className="h-3.5 w-3.5" /> Download</button>
            <button onClick={() => window.print()} className="btn-primary px-3 py-1.5 text-xs"><Printer className="h-3.5 w-3.5" /> Print or save PDF</button>
            <button onClick={onClose} autoFocus className="btn-ghost p-1.5" aria-label="Close summary"><X className="h-4 w-4" /></button>
          </div>
        </div>

        <article className="print-page rounded-2xl bg-white text-zinc-900 shadow-2xl shadow-black/50">
          <header className="flex flex-wrap items-start justify-between gap-4 border-b border-zinc-200 px-6 py-6 sm:px-10 sm:py-8">
            <div>
              <p className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-emerald-700">
                <ShieldCheck className="h-3.5 w-3.5" aria-hidden /> Churn Decision Intelligence
              </p>
              <h2 id="exec-title" className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{report.title}</h2>
              <p className="mt-1 text-sm text-zinc-500">Executive summary · {report.generatedOn}</p>
            </div>
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${report.positive ? 'bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200' : 'bg-rose-50 text-rose-800 ring-1 ring-rose-200'}`}>
              {report.positive ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
              {report.positive ? 'Recommended' : 'Not profitable as configured'}
            </span>
          </header>

          <div className="space-y-8 px-6 py-6 sm:px-10 sm:py-8">
            <Section title="Recommendation">
              <p className={`rounded-xl border-l-4 p-4 text-[15px] leading-relaxed ${report.positive ? 'border-emerald-500 bg-emerald-50/60' : 'border-rose-500 bg-rose-50/60'}`}>
                {report.headline}
              </p>
            </Section>

            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {report.metrics.map((m) => (
                <div key={m.label} className="break-inside-avoid rounded-xl border border-zinc-200 p-4">
                  <p className="text-[11px] text-zinc-500">{m.label}</p>
                  <p className={`mt-1 text-xl font-semibold tabular-nums ${m.tone === 'value' ? 'text-emerald-700' : m.tone === 'risk' ? 'text-rose-700' : 'text-zinc-900'}`}>{m.value}</p>
                  {m.note && <p className="mt-0.5 text-[11px] tabular-nums text-zinc-500">{m.note}</p>}
                </div>
              ))}
            </div>

            <Section title="Contact plan by risk band">
              <div className="overflow-x-auto rounded-xl border border-zinc-200">
                <table className="w-full min-w-[560px] text-sm">
                  <thead className="bg-zinc-50 text-left text-[11px] uppercase tracking-wider text-zinc-500">
                    <tr>
                      <th scope="col" className="px-4 py-2.5 font-medium">Band</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-medium">Subscribers</th>
                      <th scope="col" className="px-4 py-2.5 text-right font-medium">Expected churners</th>
                      <th scope="col" className="px-4 py-2.5 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200">
                    {report.bands.map((b, i) => (
                      <tr key={b.label}>
                        <td className="px-4 py-3">
                          <span className="flex items-center gap-2 font-medium"><span className={`h-2 w-2 rounded-full ${BAND_DOT[i]}`} aria-hidden />{b.label}</span>
                          <span className="ml-4 text-[11px] tabular-nums text-zinc-500">{b.criterion}</span>
                        </td>
                        <td className="px-4 py-3 text-right tabular-nums">{b.customers}<span className="block text-[11px] text-zinc-500">{b.share}</span></td>
                        <td className="px-4 py-3 text-right tabular-nums">{b.churners}</td>
                        <td className="px-4 py-3 text-zinc-700">{b.action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            <div className="grid gap-8 md:grid-cols-2">
              <Section title="Assumptions"><ItemList items={report.assumptions} /></Section>
              <Section title="Model"><ItemList items={report.model} /></Section>
            </div>

            <div className="grid gap-8 md:grid-cols-2">
              <Section title="Risks and sensitivities">
                <ul className="space-y-2 text-sm leading-relaxed text-zinc-700">
                  {report.risks.map((r) => (
                    <li key={r} className="flex gap-2"><TrendingUp className="mt-1 h-3.5 w-3.5 shrink-0 text-amber-600" aria-hidden />{r}</li>
                  ))}
                </ul>
              </Section>
              <Section title="Caveats">
                <ul className="space-y-2 text-sm leading-relaxed text-zinc-700">
                  {report.caveats.map((c) => (
                    <li key={c} className="flex gap-2"><AlertTriangle className="mt-1 h-3.5 w-3.5 shrink-0 text-zinc-400" aria-hidden />{c}</li>
                  ))}
                </ul>
              </Section>
            </div>
          </div>

          <footer className="border-t border-zinc-200 px-6 py-4 text-[11px] text-zinc-500 sm:px-10">
            Open this scenario: <span className="break-all text-zinc-700">{report.link}</span>
          </footer>
        </article>
      </div>
    </div>,
    document.body,
  );
}
