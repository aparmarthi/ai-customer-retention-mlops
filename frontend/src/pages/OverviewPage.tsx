import { useState } from 'react';
import { ArrowRight, Cloud, Container, Cpu, Database, FlaskConical, GitBranch, HardDrive, LayoutDashboard, Maximize2, MessageSquareQuote, Server, Sparkles } from 'lucide-react';
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, XAxis, YAxis, ReferenceLine } from 'recharts';
import { useAssumptions } from '@/context/AssumptionsContext';
import { CAPACITY_K, ROI_MAX_K, VALID } from '@/data/validation';
import { liftAtK, net, randomNet, recallAtK, revenueAtRisk, roiMultiple, scale, tpAtK } from '@/lib/economics';
import { fmtInt, fmtMultiple, fmtPct, fmtSignedUsdCompact, fmtUsdMillions } from '@/lib/format';
import { Badge, Card } from '@/components/ui/primitives';
import { InfoTip } from '@/components/ui/InfoTip';
import { Lightbox } from '@/components/ui/Lightbox';
import { CaveatChip } from '@/components/ui/CaveatChip';

const ARCH_IMG = 'https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main/docs/architecture.png';

const PIPELINE = [
  { icon: HardDrive, title: '31 GB raw CSV', sub: 'KKBox logs + billing' },
  { icon: Database, title: 'DuckDB', sub: '38 engineered features' },
  { icon: Cpu, title: '12-model benchmark', sub: '+ FLAML tuning' },
  { icon: FlaskConical, title: 'MLflow', sub: 'tracking + registry' },
  { icon: Server, title: 'FastAPI', sub: '/predict, /predict_batch' },
  { icon: Container, title: 'Docker', sub: 'reproducible image' },
  { icon: Cloud, title: 'Cloud Run', sub: 'us-central1' },
  { icon: LayoutDashboard, title: 'Streamlit / this UI', sub: 'decision layer' },
];

function Kpi({ label, value, sub, formula }: { label: string; value: string; sub: string; formula: string }) {
  return (
    <div className="card group relative overflow-hidden p-5 transition-colors hover:border-zinc-700">
      <div className="pointer-events-none absolute -right-10 -top-10 h-28 w-28 rounded-full bg-emerald-500/5 blur-2xl transition-opacity group-hover:bg-emerald-500/10" />
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-zinc-400">{label}</p>
        <InfoTip
          label={`How "${label}" is calculated`}
          align="end"
          content={
            <>
              <span className="block font-mono text-[11px] text-zinc-200">{formula}</span>
              <span className="mt-2 block text-amber-300/90">Simulated, validation window (Feb 2017). Not realized revenue.</span>
            </>
          }
        />
      </div>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-emerald-300 tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-zinc-500 tabular-nums">{sub}</p>
    </div>
  );
}

export function OverviewPage() {
  const { valuePerSave: V, costPerContact: C, saveRate: S, subscribers } = useAssumptions();
  const [archOpen, setArchOpen] = useState(false);

  const modelNet = net(ROI_MAX_K, V, C, S);
  const randNet = randomNet(ROI_MAX_K, V, C, S);
  const annual = scale(modelNet, subscribers) * 12;
  const roi = roiMultiple(ROI_MAX_K, V, C, S);
  const recall = recallAtK(ROI_MAX_K);
  const lift10k = liftAtK(CAPACITY_K);
  const atRisk = scale(revenueAtRisk(V), subscribers);

  const compare = [
    { name: 'Model (top 1,478)', value: modelNet },
    { name: 'Random 1,478', value: randNet },
  ];

  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-2xl border border-zinc-800/80 bg-gradient-to-br from-zinc-900 via-zinc-900/80 to-zinc-950 p-6 md:p-10">
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="relative max-w-3xl">
          <Badge tone="value"><Sparkles className="h-3 w-3" /> Executive overview</Badge>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight text-zinc-50 md:text-5xl">
            Find the <span className="text-emerald-300">0.8%</span> of subscribers who drive <span className="text-rose-300">43%</span> of churn, before they leave.
          </h1>
          <p className="mt-4 text-base text-zinc-400 md:text-lg">
            LightGBM churn model + capacity-aware decision policy on 31 GB of KKBox subscription and listening logs.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <a href="#/simulator" className="btn-primary">Open policy simulator <ArrowRight className="h-4 w-4" /></a>
            <a href="#/score" className="btn-secondary">Score customers</a>
          </div>
          <p className="mt-6 text-xs text-zinc-500 tabular-nums">
            Revenue at risk (churners × value per save), scaled to {fmtInt(subscribers)} subscribers:{' '}
            <span className="font-medium text-rose-300">{fmtUsdMillions(atRisk)}/cycle</span> · simulated, validation window
          </p>
        </div>
      </section>

      <section aria-label="Key metrics" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi
          label="Annualized net retention value"
          value={`${fmtUsdMillions(annual)}/yr`}
          sub={`Top ${fmtInt(ROI_MAX_K)} policy · scaled to ${fmtInt(subscribers)} subs`}
          formula={`(TP(K)·S·V − K·C) × subs/193,205 × 12, K=${fmtInt(ROI_MAX_K)}`}
        />
        <Kpi label="Return on retention spend" value={fmtMultiple(roi)} sub="Value saved ÷ contact spend" formula="(TP(K)·S·V) ÷ (K·C), K=1,478" />
        <Kpi
          label="Churners caught by contacting 0.76% of the base"
          value={fmtPct(recall)}
          sub={`${fmtInt(tpAtK(ROI_MAX_K))} of ${fmtInt(VALID.churners)} churners in validation`}
          formula="Recall = TP(1,478) ÷ 2,402 churners"
        />
        <Kpi label="Lift vs random at top 10K" value={fmtMultiple(lift10k)} sub="Precision@10K ÷ 1.24% base rate" formula="(TP(10,000)/10,000) ÷ 0.01243" />
      </section>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3" title="Model vs random targeting" subtitle="Net value per cycle at validation scale (193,205 subscribers)">
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={compare} layout="vertical" margin={{ left: 8, right: 64, top: 4, bottom: 4 }}>
                <XAxis type="number" hide domain={['dataMin', 'dataMax']} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#a1a1aa', fontSize: 12 }} axisLine={false} tickLine={false} />
                <ReferenceLine x={0} stroke="#52525b" />
                <Bar dataKey="value" radius={[4, 4, 4, 4]} barSize={28} isAnimationActive>
                  {compare.map((d) => (
                    <Cell key={d.name} fill={d.value >= 0 ? '#34d399' : '#fb7185'} fillOpacity={d.value >= 0 ? 0.9 : 0.75} />
                  ))}
                  <LabelList dataKey="value" position="right" formatter={(v: number) => fmtSignedUsdCompact(v)} fill="#e4e4e7" fontSize={13} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-2 text-sm text-zinc-300">
            Same budget, same {fmtInt(ROI_MAX_K)} contacts — the model is the difference between profit and loss.
          </p>
          <CaveatChip className="mt-4" />
        </Card>

        <Card
          className="lg:col-span-2"
          title="What I'd tell the VP of Retention"
          icon={<MessageSquareQuote className="h-4 w-4 text-emerald-400" />}
        >
          <ol className="space-y-4">
            {[
              'Start with top-1,478 at the 0.68 threshold for max ROI.',
              'Scale to top-10K only when the team has capacity and a cheaper nudge channel.',
              'Re-validate the save-rate assumption with a holdout A/B before trusting the dollar figures.',
            ].map((t, i) => (
              <li key={t} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-emerald-500/30 bg-emerald-500/10 text-xs font-semibold text-emerald-300 tabular-nums">{i + 1}</span>
                <span className="text-sm text-zinc-300">{t}</span>
              </li>
            ))}
          </ol>
        </Card>
      </div>

      <Card
        title="End-to-end pipeline"
        subtitle="Click to view the full architecture diagram"
        icon={<GitBranch className="h-4 w-4" />}
        action={
          <button onClick={() => setArchOpen(true)} className="btn-ghost px-2 py-1 text-xs">
            <Maximize2 className="h-3.5 w-3.5" /> Architecture
          </button>
        }
      >
        <button onClick={() => setArchOpen(true)} className="w-full text-left" aria-label="Open architecture diagram">
          <ol className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
            {PIPELINE.map(({ icon: Icon, title, sub }, i) => (
              <li key={title} className="group relative rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 transition-all hover:-translate-y-0.5 hover:border-emerald-500/40">
                <div className="flex items-center justify-between">
                  <Icon className="h-4 w-4 text-emerald-400" aria-hidden />
                  <span className="text-[10px] text-zinc-600 tabular-nums">{String(i + 1).padStart(2, '0')}</span>
                </div>
                <p className="mt-3 text-xs font-medium text-zinc-100">{title}</p>
                <p className="mt-0.5 text-[11px] text-zinc-500">{sub}</p>
              </li>
            ))}
          </ol>
        </button>
      </Card>

      {archOpen && <Lightbox src={ARCH_IMG} alt="System architecture diagram" onClose={() => setArchOpen(false)} />}
    </div>
  );
}
