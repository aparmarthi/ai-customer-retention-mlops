import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, Braces, CheckCircle2, ListChecks, Loader2, Timer, Zap } from 'lucide-react';
import { SAMPLES, type CustomerRecord, type Sample } from '@/data/samples';
import { PROFILE_SAMPLES } from '@/data/profiles';
import { demoEstimate, explainRecord, type RiskFactor } from '@/lib/demoScore';
import { modelBaseline, toModelFactors } from '@/lib/attributions';
import { DEFAULT_THRESHOLD } from '@/data/validation';
import { useApi } from '@/context/ApiContext';
import { useToast } from '@/context/ToastContext';
import { ApiError, predict } from '@/lib/api';
import { fmtMs, fmtProb } from '@/lib/format';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { FeatureForm } from '@/pages/score/FeatureForm';
import { Gauge } from '@/pages/score/Gauge';
import { RISK_BANDS, riskBand } from '@/pages/score/risk';
import { RiskFactors } from '@/pages/score/RiskFactors';

interface Result {
  probability: number;
  label: 0 | 1;
  threshold: number;
  latencyMs: number | null;
  source: 'live' | 'demo' | 'estimate';
  expected?: number;
  model?: { factors: RiskFactor[]; baseline: number };
  warnings?: string[];
}

const PROFILES = [...PROFILE_SAMPLES].sort((a, b) => a.expected - b.expected);
const CHOICES: Sample[] = [...PROFILES, ...SAMPLES];
const GROUPS = [
  { title: 'Customer profiles', hint: 'Low to high risk · demo estimates until the API is connected', start: 0, items: PROFILES },
  { title: 'Recorded model scores', hint: 'Reference records with scores recorded from the champion model', start: PROFILES.length, items: SAMPLES },
];

const sameRecord = (a: CustomerRecord, b: CustomerRecord) => {
  const ka = Object.keys(a);
  return ka.length === Object.keys(b).length && ka.every((k) => a[k] === b[k]);
};

function isRecord(x: unknown): x is CustomerRecord {
  return !!x && typeof x === 'object' && !Array.isArray(x) && Object.values(x).every((v) => typeof v === 'number' || typeof v === 'string');
}

export function SingleScore() {
  const { baseUrl, isLive, status } = useApi();
  const { notify } = useToast();
  const [sampleIdx, setSampleIdx] = useState<number | null>(0);
  const [record, setRecord] = useState<CustomerRecord>(CHOICES[0].record);
  const [jsonText, setJsonText] = useState(JSON.stringify(CHOICES[0].record, null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [mode, setMode] = useState<'json' | 'form'>('json');
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sample = sampleIdx !== null ? CHOICES[sampleIdx] : null;
  const unmodified = !!sample && sameRecord(record, sample.record);
  const recorded = unmodified && sample && !sample.estimated ? sample : null;

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [record, threshold]);

  const pickSample = (i: number) => {
    setSampleIdx(i);
    setRecord(CHOICES[i].record);
    setJsonText(JSON.stringify(CHOICES[i].record, null, 2));
    setJsonError(null);
  };

  const updateFromForm = (r: CustomerRecord) => {
    setRecord(r);
    setJsonText(JSON.stringify(r, null, 2));
    setJsonError(null);
  };

  const updateFromJson = (text: string) => {
    setJsonText(text);
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isRecord(parsed)) throw new Error('Expected a flat object of feature: number | string');
      setRecord(parsed);
      setJsonError(null);
    } catch (e) {
      setJsonError(e instanceof Error ? e.message : 'Invalid JSON');
    }
  };

  const thresholdValid = threshold >= 0 && threshold <= 1;

  const score = async () => {
    if (jsonError || !thresholdValid) return;
    setError(null);
    if (!isLive) {
      const p = recorded ? recorded.expected : demoEstimate(record);
      setResult({ probability: p, label: p >= threshold ? 1 : 0, threshold, latencyMs: null, source: recorded ? 'demo' : 'estimate' });
      return;
    }
    setLoading(true);
    try {
      const { data, latencyMs } = await predict(baseUrl, record, threshold);
      if (typeof data?.churn_probability !== 'number') throw new ApiError('Unexpected /predict response shape');
      setResult({
        probability: data.churn_probability,
        label: data.churn_label,
        threshold: data.threshold_used,
        latencyMs,
        source: 'live',
        expected: recorded?.expected,
        model:
          data.explanations?.length && typeof data.base_value === 'number'
            ? { factors: toModelFactors(data.explanations, data.base_value, data.churn_probability), baseline: modelBaseline(data.base_value) }
            : undefined,
        warnings: data.warnings,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Prediction failed';
      const latency = e instanceof ApiError && e.latencyMs ? ` (${fmtMs(e.latencyMs)})` : '';
      setError(msg);
      notify('error', `POST /predict failed${latency}`, msg);
    } finally {
      setLoading(false);
    }
  };

  const band = result ? RISK_BANDS[riskBand(result.probability)] : null;
  const explanation = useMemo(() => explainRecord(record), [record]);
  const busy = status === 'waking' || status === 'checking';

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <div className="space-y-6 xl:col-span-3">
        <Card title="Pick a sample customer" subtitle="Profiles across the full risk range, plus five records with recorded champion scores." bodyClassName="space-y-5 p-4">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 px-0.5">
                <h3 className="text-xs font-medium text-zinc-300">{g.title}</h3>
                <p className="text-[11px] text-zinc-500">{g.hint}</p>
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {g.items.map((s, j) => {
                  const i = g.start + j;
                  const active = sampleIdx === i;
                  const b = RISK_BANDS[riskBand(s.expected)];
                  return (
                    <button
                      key={s.label}
                      onClick={() => pickSample(i)}
                      aria-pressed={active}
                      className={`group relative overflow-hidden rounded-lg border p-3 pl-4 text-left transition-all ${active ? 'border-emerald-500/50 bg-emerald-500/[0.07]' : 'border-zinc-800 bg-zinc-950/60 hover:border-zinc-700 hover:bg-zinc-900/60'}`}
                    >
                      <span aria-hidden className="absolute inset-y-0 left-0 w-1 opacity-70 transition-opacity group-hover:opacity-100" style={{ background: b.hex }} />
                      <p className="text-xs font-medium leading-snug text-zinc-200">{s.label}</p>
                      <p className="mt-2 flex items-center justify-between text-[11px] tabular-nums">
                        <span className="text-zinc-500">{s.estimated ? 'Estimate' : 'Recorded'}</span>
                        <span className={`font-semibold ${b.text}`}>{fmtProb(s.expected)}</span>
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </Card>

        <Card
          title="Customer record"
          subtitle={sample ? (unmodified ? `${sample.estimated ? 'Profile' : 'Sample'}: ${sample.label}` : 'Custom edits') : 'Custom record'}
          action={
            <div className="flex rounded-lg border border-zinc-800 p-0.5" role="tablist" aria-label="Editor mode">
              {(['json', 'form'] as const).map((m) => (
                <button
                  key={m}
                  role="tab"
                  aria-selected={mode === m}
                  onClick={() => setMode(m)}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs ${mode === m ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                  {m === 'json' ? <Braces className="h-3.5 w-3.5" /> : <ListChecks className="h-3.5 w-3.5" />}
                  {m === 'json' ? 'JSON' : 'Form'}
                </button>
              ))}
            </div>
          }
        >
          {mode === 'json' ? (
            <div>
              <textarea
                aria-label="Customer record JSON"
                value={jsonText}
                onChange={(e) => updateFromJson(e.target.value)}
                spellCheck={false}
                className={`h-80 w-full resize-y rounded-lg border bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-300 focus:outline-none focus:ring-2 ${jsonError ? 'border-rose-500/60 focus:ring-rose-500/20' : 'border-zinc-800 focus:border-emerald-500/60 focus:ring-emerald-500/20'}`}
              />
              {jsonError && <p className="mt-2 text-xs text-rose-400">{jsonError}</p>}
            </div>
          ) : (
            <FeatureForm record={record} onChange={updateFromForm} />
          )}
        </Card>
      </div>

      <div className="xl:col-span-2">
        <div className="space-y-4 xl:sticky xl:top-20">
          <Card title="Score" subtitle="POST /predict · threshold policy">
            <div className="flex items-end gap-3">
              <div className="flex-1">
                <label htmlFor="single-threshold" className="text-xs text-zinc-400">Threshold</label>
                <input id="single-threshold" type="number" min={0} max={1} step={0.01} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="input mt-1.5" />
              </div>
              <button onClick={score} disabled={loading || !!jsonError || !thresholdValid || busy} className="btn-primary h-[38px] px-5">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                Score
              </button>
            </div>
            {!thresholdValid && <p className="mt-2 text-xs text-rose-400">Threshold must be between 0 and 1.</p>}

            <div className="mt-6 min-h-[260px]">
              {loading ? (
                <div className="flex flex-col items-center gap-4">
                  <Skeleton className="h-28 w-56 rounded-t-full" />
                  <Skeleton className="h-5 w-32" />
                  <Skeleton className="h-4 w-48" />
                </div>
              ) : error ? (
                <div className="flex flex-col items-center gap-3 py-10 text-center">
                  <AlertCircle className="h-8 w-8 text-rose-400" />
                  <p className="text-sm font-medium text-zinc-200">Scoring failed</p>
                  <p className="max-w-xs break-words text-xs text-zinc-400">{error}</p>
                </div>
              ) : result && band ? (
                <div className="flex animate-fade-in flex-col items-center">
                  <Gauge value={result.probability} />
                  <span className={`mt-2 rounded-full border px-3 py-1 text-sm font-semibold ${band.bg} ${band.text}`}>{band.label}</span>
                  <div className="mt-4 flex flex-wrap justify-center gap-2">
                    {result.source === 'demo' && <Badge tone="warn">demo · recorded score</Badge>}
                    {result.source === 'estimate' && <Badge tone="warn">demo estimate · not the model</Badge>}
                    {result.source === 'live' && result.expected !== undefined && Math.abs(result.probability - result.expected) <= 0.001 && (
                      <Badge tone="value"><CheckCircle2 className="h-3 w-3" /> matches validation run</Badge>
                    )}
                    {result.source === 'live' && result.expected !== undefined && Math.abs(result.probability - result.expected) > 0.001 && (
                      <Badge tone="warn">differs from validation run ({fmtProb(result.expected)})</Badge>
                    )}
                  </div>
                  <dl className="mt-5 grid w-full grid-cols-3 gap-2 text-center">
                    <div className="rounded-lg border border-zinc-800 p-2">
                      <dd className="text-sm font-semibold tabular-nums text-zinc-100">{result.label}</dd>
                      <dt className="text-[10px] text-zinc-500">churn_label</dt>
                    </div>
                    <div className="rounded-lg border border-zinc-800 p-2">
                      <dd className="text-sm font-semibold tabular-nums text-zinc-100">{result.threshold}</dd>
                      <dt className="text-[10px] text-zinc-500">threshold_used</dt>
                    </div>
                    <div className="rounded-lg border border-zinc-800 p-2">
                      <dd className="flex items-center justify-center gap-1 text-sm font-semibold tabular-nums text-zinc-100">
                        <Timer className="h-3 w-3 text-zinc-500" />
                        {result.latencyMs !== null ? fmtMs(result.latencyMs) : '—'}
                      </dd>
                      <dt className="text-[10px] text-zinc-500">latency</dt>
                    </div>
                  </dl>
                  <p className="mt-3 text-xs text-zinc-500 tabular-nums">probability {result.probability.toFixed(4)}</p>
                  <p className="mt-1 text-xs text-zinc-400">Suggested action: <span className="text-zinc-200">{band.action}</span></p>
                  {result.warnings?.map((w) => (
                    <p key={w} role="alert" className="mt-3 flex max-w-xs items-start gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-left text-[11px] leading-relaxed text-amber-200">
                      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
                      {w}
                    </p>
                  ))}
                  {result.source === 'estimate' && (
                    <p className="mt-3 max-w-xs text-center text-[11px] leading-relaxed text-zinc-500">
                      A rule of thumb based on auto-renew, cancellations, expiry and recent listening. Connect the API in Settings for the trained model's score.
                    </p>
                  )}
                  <div className="mt-5 w-full">
                    <RiskFactors explanation={explanation} modelScore={result.source === 'estimate' ? undefined : result.probability} model={result.model} />
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 py-12 text-center">
                  <Zap className="h-8 w-8 text-zinc-700" />
                  <p className="text-sm text-zinc-400">Pick a sample or edit the record, then score.</p>
                </div>
              )}
            </div>
          </Card>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-[11px] text-zinc-500">
            Bands: ≥ 0.68 <span className="text-rose-300">Contact now</span> · 0.2058–0.68 <span className="text-amber-300">Nudge</span> · &lt; 0.2058 <span className="text-emerald-300">Monitor</span>
          </div>
        </div>
      </div>
    </div>
  );
}
