import { Box, Cloud, Cpu, FileJson, GitBranch, HeartPulse, Layers, Timer } from 'lucide-react';
import { POLICY } from '@/data/validation';
import { Badge, Card, PageHeader } from '@/components/ui/primitives';
import { CodeBlock } from '@/components/ui/CopyButton';

const DEPLOY = [
  { icon: Cloud, title: 'FastAPI on Cloud Run', detail: 'us-central1 · 1 vCPU / 1 GiB · min 0 / max 3 instances · scale-to-zero' },
  { icon: GitBranch, title: 'CI: GitHub Actions', detail: 'pytest → build image → deploy' },
  { icon: Box, title: 'Docker image', detail: 'Reproducible serving container' },
  { icon: HeartPulse, title: '/health checks', detail: 'Model, threshold, feature list and categorical columns loaded' },
  { icon: Timer, title: 'Latency logging', detail: 'Request-latency logging middleware' },
  { icon: Layers, title: 'MLflow', detail: 'Experiment tracking + model registry' },
];

const MONITORS = [
  { signal: 'Feature drift (PSI)', threshold: 'warn > 0.20, alert > 0.25', action: '—', seed: 1 },
  { signal: 'Null rate', threshold: '> 2× baseline', action: '—', seed: 2 },
  { signal: 'Scoring volume', threshold: '±20% vs trailing', action: '—', seed: 3 },
  { signal: 'Label feedback', threshold: '30-day delay before precision can be measured', action: '—', seed: 4 },
  { signal: 'Precision@K', threshold: '< 1.5× base rate for 2 consecutive months', action: 'Retrain', seed: 5 },
  { signal: 'Retrain cadence', threshold: 'Quarterly', action: 'Registry rollback to previous champion on regression', seed: 6 },
];

function MockSparkline({ seed }: { seed: number }) {
  const pts = Array.from({ length: 16 }, (_, i) => 12 + Math.sin(i * 0.9 + seed * 1.7) * 5 + Math.cos(i * 0.4 + seed) * 3);
  const d = pts.map((y, i) => `${i === 0 ? 'M' : 'L'}${(i / 15) * 96 + 2},${y.toFixed(1)}`).join(' ');
  return (
    <div className="flex items-center gap-2">
      <svg viewBox="0 0 100 24" className="h-6 w-24" aria-hidden>
        <path d={d} fill="none" stroke="#52525b" strokeWidth="1.5" strokeDasharray="3 2" />
      </svg>
      <span className="text-[10px] uppercase tracking-wider text-zinc-600">Mock</span>
    </div>
  );
}

export function MlopsPage() {
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="MLOps & Monitoring" title="How the model ships and stays honest" description="Serving, CI/CD and the monitoring plan that decides when to retrain or roll back." />

      <Card title="Deployment" icon={<Cpu className="h-4 w-4" />}>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {DEPLOY.map(({ icon: Icon, title, detail }) => (
            <div key={title} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4 transition-colors hover:border-zinc-700">
              <Icon className="h-4 w-4 text-emerald-400" aria-hidden />
              <p className="mt-3 text-sm font-medium text-zinc-100">{title}</p>
              <p className="mt-1 text-xs text-zinc-500">{detail}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card
        title="Monitoring"
        subtitle="Signal → threshold → action. There is no live telemetry yet; sparklines are placeholders."
        action={<Badge tone="warn">Mock — wiring planned</Badge>}
        bodyClassName="p-0"
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500">
                <th scope="col" className="px-5 py-3 font-medium">Signal</th>
                <th scope="col" className="px-5 py-3 font-medium">Threshold</th>
                <th scope="col" className="px-5 py-3 font-medium">Action</th>
                <th scope="col" className="px-5 py-3 font-medium">Trend</th>
              </tr>
            </thead>
            <tbody>
              {MONITORS.map((m) => (
                <tr key={m.signal} className="border-b border-zinc-800/60 last:border-0 hover:bg-zinc-800/20">
                  <td className="px-5 py-3 font-medium text-zinc-100">{m.signal}</td>
                  <td className="px-5 py-3 text-zinc-300 tabular-nums">{m.threshold}</td>
                  <td className="px-5 py-3 text-zinc-400">{m.action}</td>
                  <td className="px-5 py-3"><MockSparkline seed={m.seed} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Deployment policy" subtitle={`${POLICY.name} · ${POLICY.version}`} icon={<FileJson className="h-4 w-4" />}>
        <div className="grid gap-6 lg:grid-cols-2">
          <CodeBlock code={JSON.stringify(POLICY, null, 2)} copyLabel="Copy JSON" />
          <div className="space-y-3">
            <div className="rounded-lg border border-emerald-500/25 bg-emerald-500/[0.05] p-4">
              <p className="label text-emerald-400">Primary · top-K = capacity</p>
              <p className="mt-2 text-sm text-zinc-300">
                Each cycle the retention team works the top {POLICY.primary.k.toLocaleString('en-US')} by score. The constraint is how many people ops can actually contact, so K is set by headcount, not by a probability cutoff.
              </p>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
              <p className="label">Fallback · threshold = ROI</p>
              <p className="mt-2 text-sm text-zinc-300">
                When capacity isn't the binding constraint, contact everyone scoring ≥ {POLICY.fallback.threshold} — the ROI-max cutoff on validation.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
