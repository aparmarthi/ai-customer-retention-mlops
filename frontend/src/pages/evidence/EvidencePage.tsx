import { AlertTriangle, CheckCircle2, Cloud, Info, Trophy } from 'lucide-react';
import { SAGEMAKER, LIFT } from '@/data/leaderboard';
import { VALID } from '@/data/validation';
import { fmtInt, fmtMultiple, fmtPct } from '@/lib/format';
import { Badge, Card, PageHeader } from '@/components/ui/primitives';
import { FigureCard } from '@/components/ui/Lightbox';
import { Leaderboard } from '@/pages/evidence/Leaderboard';

const REPO_RAW = 'https://raw.githubusercontent.com/aparmarthi/ai-customer-retention-mlops/main';

const FIGURES = [
  { src: `${REPO_RAW}/reports/shap_summary.png`, title: 'SHAP summary', caption: 'Global feature attribution for the champion LightGBM model.' },
  { src: `${REPO_RAW}/reports/threshold_vs_precision_recall.png`, title: 'Threshold vs precision / recall', caption: 'How the cutoff trades precision against recall on validation.' },
  { src: `${REPO_RAW}/reports/threshold_vs_roi.png`, title: 'Threshold vs ROI', caption: 'ROI by cutoff; 0.68 is the ROI-max fallback threshold in the deployment policy.' },
];

const WHY = [
  'Best PR-AUC of the seven model families benchmarked (12 experiments in total).',
  'Roughly 3-minute CPU training vs 23–32 minutes on GPU for the deep tabular models.',
  'Native categorical handling (gender) with no one-hot encoding step.',
  'Cheap serving: an 18.7 MB artifact on 1 vCPU / 1 GiB.',
];

function Metric({ label, value, note, tone }: { label: string; value: string; note: string; tone?: 'value' }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className={`mt-1 text-2xl font-semibold tabular-nums ${tone === 'value' ? 'text-emerald-300' : 'text-zinc-100'}`}>{value}</p>
      <p className="mt-1 text-[11px] text-zinc-500">{note}</p>
    </div>
  );
}

export function EvidencePage() {
  const prLift = VALID.prAuc / VALID.churnRate;
  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Model Evidence"
        title="Why we trust the ranking"
        description="Time-based validation is what ships. The random-split benchmark is how the model family was chosen."
      />

      <Card
        title="Champion validation · time-based"
        subtitle={`${VALID.champion} · ${VALID.window} · ${fmtInt(VALID.n)} subscribers · ${fmtInt(VALID.churners)} churners (${fmtPct(VALID.churnRate, 2)})`}
        action={<Badge tone="value">Ship numbers</Badge>}
      >
        <div className="grid gap-3 sm:grid-cols-3">
          <Metric label="ROC-AUC" value={VALID.rocAuc.toFixed(3)} note="Ranking quality across all cutoffs" />
          <Metric label="PR-AUC" value={VALID.prAuc.toFixed(3)} note={`Base rate ${fmtPct(VALID.churnRate, 2)} → ${Math.round(prLift)}× better than random`} tone="value" />
          <Metric label="F1 @ 0.5" value={VALID.f1At05.toFixed(3)} note="Shown for completeness — see note" />
        </div>
        <p className="mt-4 flex items-start gap-2 text-xs text-zinc-400">
          <Info className="mt-px h-3.5 w-3.5 shrink-0 text-sky-400" />
          0.5 is the wrong cutoff for a 1.2%-prevalence problem — that's why the policy picks K, not 0.5.
        </p>
      </Card>

      <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/[0.06] p-4 text-sm text-amber-100">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        <p>
          Benchmark used a random split (6.4% prevalence); champion was re-validated on a time-based holdout of subscribers whose last transaction is after 2017-02-24 (1.24%). Numbers are not directly comparable — the time-based ones are what we'd ship on.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <Leaderboard />
        </div>
        <div className="space-y-6">
          <Card title="Why LightGBM" icon={<Trophy className="h-4 w-4 text-emerald-400" />}>
            <ul className="space-y-3">
              {WHY.map((w) => (
                <li key={w} className="flex gap-2.5 text-sm text-zinc-300">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                  {w}
                </li>
              ))}
            </ul>
            <div className="mt-4 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-xs text-zinc-400">
              <span className="font-medium text-zinc-200">Ensemble collapsed to LightGBM.</span> The soft-vote search put weight 1.0 on LightGBM, so an ensemble adds serving cost for no lift.
            </div>
            <dl className="mt-4 grid grid-cols-3 gap-2 text-center">
              {[
                ['PR-AUC lift', LIFT.prAuc],
                ['P@5K lift', LIFT.at5k],
                ['P@10K lift', LIFT.at10k],
              ].map(([l, v]) => (
                <div key={l as string} className="rounded-lg border border-zinc-800 p-2">
                  <dd className="text-base font-semibold tabular-nums text-emerald-300">{fmtMultiple(v as number)}</dd>
                  <dt className="mt-0.5 text-[10px] text-zinc-500">{l}</dt>
                </div>
              ))}
            </dl>
          </Card>

          <Card title="SageMaker portability" subtitle="Workflow check, not the champion." icon={<Cloud className="h-4 w-4" />}>
            <p className="text-xs text-zinc-400">
              Same hyperparameters retrained on SageMaker ({SAGEMAKER.instance}) on a 10% sample with a 2017-01-31 cutoff, registered in {SAGEMAKER.registry}. Different validation set, so not like-for-like.
            </p>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center">
              {[
                ['ROC-AUC', SAGEMAKER.rocAuc],
                ['PR-AUC', SAGEMAKER.prAuc],
                ['F1', SAGEMAKER.f1],
              ].map(([l, v]) => (
                <div key={l as string} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-2">
                  <p className="text-base font-semibold tabular-nums text-zinc-100">{(v as number).toFixed(3)}</p>
                  <p className="mt-0.5 text-[10px] text-zinc-500">{l}</p>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <section aria-label="Diagnostic figures">
        <h2 className="mb-3 text-sm font-semibold text-zinc-200">Diagnostics</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {FIGURES.map((f) => (
            <FigureCard key={f.src} {...f} />
          ))}
        </div>
      </section>
    </div>
  );
}
