import { useMemo, useRef, useState, type DragEvent } from 'react';
import { AlertCircle, Braces, FileSpreadsheet, FileUp, Loader2, PlugZap, Send, Sparkles, Upload, X } from 'lucide-react';
import { SAMPLES } from '@/data/samples';
import { FEATURES } from '@/data/features';
import { CAPACITY_K } from '@/data/validation';
import { useApi } from '@/context/ApiContext';
import { useToast } from '@/context/ToastContext';
import { ApiError, predictBatchFile, predictBatchRecords, type BatchResponse, type Timed } from '@/lib/api';
import { downloadText, parseCsv, toCsv, type ParsedCsv } from '@/lib/csv';
import { fmtInt, fmtMs } from '@/lib/format';
import { Card, Skeleton } from '@/components/ui/primitives';
import { BatchResults } from '@/pages/score/BatchResults';
import { isBatchResponse, joinItems, rankLocally, type BatchRun } from '@/pages/score/batch';
import { JsonBatchInput } from '@/pages/score/JsonBatchInput';
import { demoProbabilities, exampleBatchJson, parseJsonRecords } from '@/pages/score/jsonBatch';

type InputMode = 'csv' | 'json';

const MAX_FILE_BYTES = 200 * 1024 * 1024;

export function BatchScore() {
  const { baseUrl, isLive } = useApi();
  const { notify } = useToast();
  const [k, setK] = useState(CAPACITY_K);
  const [file, setFile] = useState<{ file: File; parsed: ParsedCsv } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoBlocked, setDemoBlocked] = useState(false);
  const [run, setRun] = useState<BatchRun | null>(null);
  const [mode, setMode] = useState<InputMode>('csv');
  const [jsonText, setJsonText] = useState(exampleBatchJson);
  const jsonParsed = useMemo(() => parseJsonRecords(jsonText), [jsonText]);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = async (f: File | undefined) => {
    if (!f) return;
    setError(null);
    if (!/\.csv$/i.test(f.name) && f.type !== 'text/csv') {
      notify('error', 'Unsupported file', 'Please upload a .csv file with a header row.');
      return;
    }
    if (f.size > MAX_FILE_BYTES) {
      notify('error', 'File too large', 'Keep uploads under 200 MB.');
      return;
    }
    const parsed = parseCsv(await f.text());
    if (parsed.headers.length === 0 || parsed.rows.length === 0) {
      notify('error', 'Empty CSV', 'The file needs a header row and at least one customer.');
      return;
    }
    setFile({ file: f, parsed });
    setRun(null);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    accept(e.dataTransfer.files?.[0]);
  };

  const finish = (res: Timed<BatchResponse>, inputs: ParsedCsv, kEff: number, origin: string) => {
    if (!isBatchResponse(res.data)) throw new ApiError('Unexpected /predict_batch response shape');
    if (res.data.items.length !== inputs.rows.length) {
      notify('info', 'Row count mismatch', `Sent ${fmtInt(inputs.rows.length)} rows, received ${fmtInt(res.data.items.length)} scores.`);
    }
    const first = res.data.items[0];
    setRun({
      rows: joinItems(inputs.rows, res.data.items, kEff),
      headers: inputs.headers,
      k: kEff,
      policyUsed: res.data.policy_used,
      thresholdUsed: first?.threshold_used ?? null,
      latencyMs: res.latencyMs,
      source: 'live',
      origin,
    });
    notify('success', `Scored ${fmtInt(res.data.n)} customers`, fmtMs(res.latencyMs));
  };

  const fail = (e: unknown, endpoint: string) => {
    const msg = e instanceof Error ? e.message : 'Batch scoring failed';
    const latency = e instanceof ApiError && e.latencyMs ? ` (${fmtMs(e.latencyMs)})` : '';
    setError(msg);
    notify('error', `${endpoint} failed${latency}`, msg);
  };

  const kValid = Number.isInteger(k) && k >= 1;

  const scoreFile = async () => {
    if (!file || !kValid) return;
    setError(null);
    setRun(null);
    setDemoBlocked(!isLive);
    if (!isLive) return;
    const kEff = Math.min(k, file.parsed.rows.length);
    setLoading(true);
    try {
      finish(await predictBatchFile(baseUrl, file.file, kEff), file.parsed, kEff, file.file.name);
    } catch (e) {
      fail(e, 'POST /predict_batch (file)');
    } finally {
      setLoading(false);
    }
  };

  const scoreSamples = async () => {
    if (!kValid) return;
    setError(null);
    setDemoBlocked(false);
    setRun(null);
    setFile(null);
    const records = SAMPLES.map((s) => s.record);
    const inputs: ParsedCsv = { headers: FEATURES, rows: records };
    const kEff = Math.min(k, records.length);
    if (!isLive) {
      setRun({
        rows: joinItems(records, rankLocally(SAMPLES.map((s) => s.expected), kEff), kEff),
        headers: FEATURES,
        k: kEff,
        policyUsed: `top_k(${kEff})`,
        thresholdUsed: null,
        latencyMs: null,
        source: 'demo',
        origin: '5 sample customers',
      });
      return;
    }
    setLoading(true);
    try {
      finish(await predictBatchRecords(baseUrl, records, kEff), inputs, kEff, '5 sample customers');
    } catch (e) {
      fail(e, 'POST /predict_batch (JSON)');
    } finally {
      setLoading(false);
    }
  };

  const scoreJson = async () => {
    if (!jsonParsed.ok || !kValid) return;
    setError(null);
    setDemoBlocked(false);
    setRun(null);
    const { records, headers } = jsonParsed;
    const kEff = Math.min(k, records.length);
    const origin = `${fmtInt(records.length)} pasted JSON records`;
    if (!isLive) {
      const { probs, estimated } = demoProbabilities(records);
      setRun({
        rows: joinItems(records, rankLocally(probs, kEff), kEff),
        headers,
        k: kEff,
        policyUsed: `top_k(${kEff})`,
        thresholdUsed: null,
        latencyMs: null,
        source: estimated ? 'estimate' : 'demo',
        origin,
      });
      return;
    }
    setLoading(true);
    try {
      finish(await predictBatchRecords(baseUrl, records, kEff), { headers, rows: records }, kEff, origin);
    } catch (e) {
      fail(e, 'POST /predict_batch (JSON)');
    } finally {
      setLoading(false);
    }
  };

  const downloadTemplate = () =>
    downloadText('churn_template.csv', toCsv(FEATURES, SAMPLES.map((s) => s.record)));

  const n = mode === 'csv' ? file?.parsed.rows.length : jsonParsed.ok ? jsonParsed.records.length : undefined;
  const canScore = mode === 'csv' ? !!file : jsonParsed.ok;

  return (
    <div className="space-y-6">
      <Card title="Batch scoring" subtitle="Top-K ranking happens on the server. Results join back to your rows by position.">
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <div role="tablist" aria-label="Input format" className="mb-4 inline-flex rounded-lg border border-zinc-800 bg-zinc-950/60 p-1">
              {([
                { id: 'csv', label: 'CSV upload', icon: FileSpreadsheet },
                { id: 'json', label: 'Paste JSON', icon: Braces },
              ] as const).map((t) => (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={mode === t.id}
                  onClick={() => setMode(t.id)}
                  className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${mode === t.id ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                  <t.icon className="h-3.5 w-3.5" /> {t.label}
                </button>
              ))}
            </div>
            {mode === 'json' ? (
              <JsonBatchInput text={jsonText} parsed={jsonParsed} onChange={setJsonText} onLoadExample={() => setJsonText(exampleBatchJson())} />
            ) : (
            <>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              className={`flex min-h-[180px] flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-colors ${dragOver ? 'border-emerald-400 bg-emerald-500/[0.06]' : 'border-zinc-800 bg-zinc-950/40'}`}
            >
              {file ? (
                <div className="flex w-full max-w-md items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-3 text-left">
                  <FileSpreadsheet className="h-8 w-8 shrink-0 text-emerald-400" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-zinc-100">{file.file.name}</p>
                    <p className="text-xs text-zinc-500 tabular-nums">
                      {fmtInt(file.parsed.rows.length)} rows · {file.parsed.headers.length} columns · {(file.file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  <button onClick={() => setFile(null)} className="btn-ghost p-1.5" aria-label="Remove file">
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <>
                  <Upload className="h-8 w-8 text-zinc-600" />
                  <p className="mt-3 text-sm text-zinc-300">Drop a CSV with a header row</p>
                  <p className="mt-1 text-xs text-zinc-500">Missing feature columns are treated as missing; extra columns are ignored.</p>
                </>
              )}
              <input ref={inputRef} type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => accept(e.target.files?.[0])} aria-label="Choose CSV file" />
              <button onClick={() => inputRef.current?.click()} className="btn-secondary mt-4 text-xs">
                <FileUp className="h-3.5 w-3.5" /> {file ? 'Choose another file' : 'Browse files'}
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <button onClick={downloadTemplate} className="btn-ghost text-xs">
                <FileSpreadsheet className="h-3.5 w-3.5" /> Download template CSV
              </button>
              <button onClick={scoreSamples} disabled={loading || !kValid} className="btn-ghost text-xs">
                <Sparkles className="h-3.5 w-3.5" /> Use 5 samples
              </button>
            </div>
            </>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <label htmlFor="batch-k" className="text-xs text-zinc-400">K (top-K contacts)</label>
              <input id="batch-k" type="number" min={1} step={1} value={k} onChange={(e) => setK(Math.floor(Number(e.target.value)))} className="input mt-1.5" />
              <p className="mt-1.5 text-[11px] text-zinc-500 tabular-nums">
                {!kValid ? <span className="text-rose-400">K must be a whole number ≥ 1.</span> : n !== undefined && k > n ? `Capped at n = ${fmtInt(n)}` : 'Default 10,000 (capacity policy)'}
              </p>
            </div>
            <button onClick={mode === 'csv' ? scoreFile : scoreJson} disabled={!canScore || loading || !kValid} className="btn-primary w-full">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : mode === 'csv' ? <Upload className="h-4 w-4" /> : <Send className="h-4 w-4" />}
              {mode === 'csv' ? 'Score file' : 'Score JSON'}
            </button>
            <p className="text-[11px] text-zinc-500">
              {mode === 'csv' ? 'The CSV is uploaded as-is (multipart). Sample rows are sent as JSON.' : 'Records are sent as one JSON body with policy top_k and your K. Max 50,000 records.'}
            </p>
          </div>
        </div>
      </Card>

      {loading ? (
        <div className="space-y-4" aria-busy>
          <Skeleton className="h-32 w-full rounded-xl" />
          <Skeleton className="h-56 w-full rounded-xl" />
        </div>
      ) : demoBlocked ? (
        <div className="card flex flex-col items-center gap-3 p-10 text-center">
          <PlugZap className="h-8 w-8 text-amber-400" />
          <p className="text-sm font-medium text-zinc-200">Connect API to score uploaded files.</p>
          <p className="max-w-sm text-xs text-zinc-500">In Demo Mode, use the Paste JSON tab or the 5 samples instead. Set an API URL in Settings to score files.</p>
        </div>
      ) : error ? (
        <div className="card flex flex-col items-center gap-3 p-10 text-center">
          <AlertCircle className="h-8 w-8 text-rose-400" />
          <p className="text-sm font-medium text-zinc-200">Batch scoring failed</p>
          <p className="max-w-md break-words text-xs text-zinc-400">{error}</p>
        </div>
      ) : run ? (
        <BatchResults run={run} />
      ) : (
        <div className="card flex flex-col items-center gap-2 p-10 text-center">
          <FileSpreadsheet className="h-8 w-8 text-zinc-700" />
          <p className="text-sm text-zinc-400">No batch scored yet. Upload a CSV, paste JSON, or try the 5 samples.</p>
        </div>
      )}
    </div>
  );
}
