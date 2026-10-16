import { useState } from 'react';
import { AlertTriangle, Calculator, CheckCircle2, ClipboardList, FlaskRound, ShieldAlert, Split, Target } from 'lucide-react';
import { sampleSizePerArm } from '@/lib/stats';
import { fmtInt, fmtPct } from '@/lib/format';
import { Card, PageHeader } from '@/components/ui/primitives';

const PLAN = [
  { icon: Split, title: 'Randomized holdout', body: '50/50 within the top 10K: half get the retention offer, half get nothing.' },
  { icon: Target, title: 'Primary metric', body: '30-day retention rate delta between treatment and holdout.' },
  { icon: ShieldAlert, title: 'Guardrails', body: 'Complaint/unsubscribe rate and incentive cost per retained user.' },
  { icon: ClipboardList, title: 'Duration', body: '2 billing cycles.' },
];

const DECISIONS = [
  { title: 'LightGBM over the ensemble', why: 'The ensemble tied; the simpler model wins on serving cost and debuggability.' },
  { title: 'K-based over threshold-based primary policy', why: 'Ops capacity is the real constraint, so the policy is expressed in contacts per cycle.' },
  { title: 'PR-AUC as the selection metric', why: '1.24% prevalence makes ROC-AUC flattering; PR-AUC tracks the positives we act on.' },
  { title: 'Time-based validation', why: 'Train on earlier months, validate on Feb 2017: no leakage from future months.' },
  { title: 'CPU serving', why: 'No GPU needed for an 18.7 MB artifact.' },
];

const RISKS = [
  { title: 'Save rate is assumed, not measured', body: 'Every dollar figure scales linearly with it. The holdout above is how it gets measured.' },
  { title: 'Uplift modeling', body: 'Target persuadables rather than just high-risk users, so incentives go where they change the outcome.' },
  { title: 'Cost-sensitive tiers per channel', body: 'Match incentive and channel cost to risk tier instead of one offer for everyone.' },
];

function PctInput({ id, label, value, onChange, step = 0.5 }: { id: string; label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <div>
      <label htmlFor={id} className="text-xs text-zinc-400">{label}</label>
      <div className="relative mt-1.5">
        <input id={id} type="number" step={step} value={Number((value * 100).toFixed(2))} onChange={(e) => e.target.value !== '' && onChange(Number(e.target.value) / 100)} className="input pr-8" />
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500">%</span>
      </div>
    </div>
  );
}

function SampleSizeCalculator() {
  const [baseline, setBaseline] = useState(0.18);
  const [mde, setMde] = useState(0.03);
  const [alpha, setAlpha] = useState(0.05);
  const [power, setPower] = useState(0.8);
  const r = sampleSizePerArm(baseline, mde, alpha, power);

  return (
    <Card title="Sample-size calculator" subtitle="Two-sided two-proportion z-test" icon={<Calculator className="h-4 w-4" />}>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="grid grid-cols-2 gap-4">
          <PctInput id="ss-baseline" label="Baseline churn in top-K" value={baseline} onChange={setBaseline} />
          <PctInput id="ss-mde" label="Min. detectable effect (abs.)" value={mde} onChange={setMde} />
          <PctInput id="ss-alpha" label="α (significance)" value={alpha} onChange={setAlpha} step={1} />
          <PctInput id="ss-power" label="Power (1 − β)" value={power} onChange={setPower} step={5} />
          <p className="col-span-2 text-[11px] text-zinc-500 tabular-nums">Default baseline = 1,801 churners in the top 10,000 on validation (18%).</p>
        </div>
        <div className="flex flex-col justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950/60 p-5">
          {r ? (
            <>
              <div>
                <p className="text-xs text-zinc-500">Required per arm</p>
                <p className="mt-1 text-4xl font-semibold tabular-nums text-emerald-300">{fmtInt(r.nPerArm)}</p>
                <p className="mt-1 text-xs text-zinc-400 tabular-nums">
                  {fmtInt(r.nPerArm * 2)} total · detects churn {fmtPct(r.p1)} → {fmtPct(r.p2)}
                  {r.nPerArm * 2 <= 10000 ? ' · fits inside the top 10K' : ' · exceeds the top 10K in one cycle'}
                </p>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 font-mono text-[11px] leading-relaxed text-zinc-300">
                n = [ z<sub>1−α/2</sub>·√(2p̄(1−p̄)) + z<sub>1−β</sub>·√(p₁(1−p₁) + p₂(1−p₂)) ]² / (p₁ − p₂)²
                <p className="mt-2 text-zinc-500 tabular-nums">
                  z<sub>1−α/2</sub> = {r.zAlpha.toFixed(3)} · z<sub>1−β</sub> = {r.zBeta.toFixed(3)} · p̄ = {((r.p1 + r.p2) / 2).toFixed(3)}
                </p>
              </div>
            </>
          ) : (
            <p className="flex items-start gap-2 text-sm text-rose-300">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              Inputs must give probabilities between 0% and 100%, with an effect smaller than the baseline.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

export function ExperimentPage() {
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Experiment Plan" title="From simulated ROI to measured ROI" description="The simulator's dollars rest on an assumed save rate. This is the plan to replace that assumption with a measurement." />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {PLAN.map(({ icon: Icon, title, body }, i) => (
          <div key={title} className="card p-5">
            <div className="flex items-center justify-between">
              <Icon className="h-4 w-4 text-emerald-400" aria-hidden />
              <span className="text-[10px] text-zinc-600 tabular-nums">Step {i + 1}</span>
            </div>
            <p className="mt-3 text-sm font-medium text-zinc-100">{title}</p>
            <p className="mt-1 text-xs text-zinc-400">{body}</p>
          </div>
        ))}
      </div>

      <SampleSizeCalculator />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Decision log" icon={<CheckCircle2 className="h-4 w-4" />}>
          <ol className="space-y-3">
            {DECISIONS.map((d, i) => (
              <li key={d.title} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
                <p className="flex items-center gap-2 text-sm font-medium text-zinc-100">
                  <span className="text-xs text-zinc-600 tabular-nums">D{i + 1}</span>
                  {d.title}
                </p>
                <p className="mt-1 text-xs text-zinc-400">{d.why}</p>
              </li>
            ))}
          </ol>
        </Card>
        <Card title="Risks and next steps" icon={<FlaskRound className="h-4 w-4" />}>
          <ul className="space-y-3">
            {RISKS.map((r, i) => (
              <li key={r.title} className={`rounded-lg border p-4 ${i === 0 ? 'border-rose-500/30 bg-rose-500/[0.05]' : 'border-zinc-800 bg-zinc-950/60'}`}>
                <p className={`text-sm font-medium ${i === 0 ? 'text-rose-200' : 'text-zinc-100'}`}>{r.title}</p>
                <p className="mt-1 text-xs text-zinc-400">{r.body}</p>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
