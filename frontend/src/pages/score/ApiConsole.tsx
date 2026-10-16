import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, ExternalLink, RefreshCw, Terminal } from 'lucide-react';
import { SAMPLES } from '@/data/samples';
import { useApi } from '@/context/ApiContext';
import { getRoot, type RootResponse } from '@/lib/api';
import { fmtMs } from '@/lib/format';
import { Badge, Card, Skeleton } from '@/components/ui/primitives';
import { CodeBlock, CopyButton } from '@/components/ui/CopyButton';

function JsonPanel({ title, data, loading, error, latency, onRefresh }: { title: string; data: unknown; loading: boolean; error: string | null; latency: number | null; onRefresh: () => void }) {
  return (
    <Card
      title={title}
      action={
        <div className="flex items-center gap-2">
          {latency !== null && <Badge>{fmtMs(latency)}</Badge>}
          <button onClick={onRefresh} className="btn-ghost p-1.5" aria-label={`Refresh ${title}`} disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      }
    >
      {loading ? (
        <div className="space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      ) : error ? (
        <p className="flex items-start gap-2 text-xs text-rose-300">
          <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0" /> {error}
        </p>
      ) : data ? (
        <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-300">{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p className="text-xs text-zinc-500">No response yet.</p>
      )}
    </Card>
  );
}

export function ApiConsole() {
  const { baseUrl, health, healthLatencyMs, status, lastError, recheck } = useApi();
  const [root, setRoot] = useState<RootResponse | null>(null);
  const [rootLatency, setRootLatency] = useState<number | null>(null);
  const [rootError, setRootError] = useState<string | null>(null);
  const [rootLoading, setRootLoading] = useState(false);

  const loadRoot = useCallback(async () => {
    setRootLoading(true);
    setRootError(null);
    try {
      const { data, latencyMs } = await getRoot(baseUrl);
      setRoot(data);
      setRootLatency(latencyMs);
    } catch (e) {
      setRoot(null);
      setRootLatency(null);
      setRootError(e instanceof Error ? e.message : 'Request failed');
    } finally {
      setRootLoading(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    if (status === 'live') loadRoot();
  }, [status, loadRoot]);

  const sampleJson = JSON.stringify({ record: SAMPLES[3].record, policy: 'threshold', threshold: 0.68 });
  const batchJson = JSON.stringify({ records: SAMPLES.slice(0, 2).map((s) => s.record), policy: 'top_k', k: 1 });

  const curls = [
    { title: 'Health', code: `curl -s ${baseUrl}/health` },
    { title: 'POST /predict (JSON)', code: `curl -s -X POST ${baseUrl}/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '${sampleJson}'` },
    { title: 'POST /predict_batch (JSON, top-K)', code: `curl -s -X POST ${baseUrl}/predict_batch \\\n  -H 'Content-Type: application/json' \\\n  -d '${batchJson}'` },
    { title: 'POST /predict_batch (CSV upload, top-K)', code: `curl -s -X POST '${baseUrl}/predict_batch?policy=top_k&k=10000' \\\n  -F 'file=@customers.csv'` },
  ];

  const healthBusy = status === 'waking' || status === 'checking';

  return (
    <div className="space-y-6">
      <Card title="Endpoint" icon={<Terminal className="h-4 w-4" />}>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <code className="min-w-0 flex-1 truncate rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-200">{baseUrl}</code>
          <div className="flex gap-2">
            <CopyButton text={baseUrl} label="Copy URL" />
            <a href={`${baseUrl}/docs`} target="_blank" rel="noopener noreferrer" className="btn-secondary px-2.5 py-1.5 text-xs">
              <ExternalLink className="h-3.5 w-3.5" /> Open /docs
            </a>
          </div>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <JsonPanel
          title="GET /health"
          data={health}
          loading={healthBusy}
          error={status === 'demo' ? lastError ?? 'API unreachable' : null}
          latency={healthLatencyMs}
          onRefresh={recheck}
        />
        <JsonPanel
          title="GET /"
          data={root}
          loading={rootLoading}
          error={status === 'demo' && !root ? 'API unreachable (Demo Mode)' : rootError}
          latency={rootLatency}
          onRefresh={loadRoot}
        />
      </div>

      <Card title="curl examples" subtitle="Built from the current base URL">
        <div className="space-y-4">
          {curls.map((c) => (
            <div key={c.title}>
              <p className="mb-1.5 text-xs font-medium text-zinc-400">{c.title}</p>
              <CodeBlock code={c.code} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
