import { Loader2 } from 'lucide-react';
import { useApi } from '@/context/ApiContext';

export function StatusPill() {
  const { status, health, recheck } = useApi();

  if (status === 'waking' || status === 'checking') {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-zinc-700 bg-zinc-900 px-3 py-1 text-xs text-zinc-300" role="status">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400" aria-hidden />
        {status === 'waking' ? 'Waking API (cold start)\u2026' : 'Checking API\u2026'}
      </span>
    );
  }

  if (status === 'live' && health) {
    return (
      <button
        onClick={recheck}
        title="Re-check /health"
        className="inline-flex max-w-full items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300 transition-colors hover:bg-emerald-500/15"
      >
        <span className="relative flex h-2 w-2 shrink-0">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
        </span>
        <span className="truncate tabular-nums">
          Live API<span className="hidden sm:inline"> · {health.model_loaded ? 'model_loaded' : 'model not loaded'} · {health.n_features} features · default threshold {health.default_threshold}</span>
        </span>
      </button>
    );
  }

  return (
    <button
      onClick={recheck}
      title="Retry connection"
      className="inline-flex max-w-full items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs text-amber-300 transition-colors hover:bg-amber-500/15"
    >
      <span className="h-2 w-2 shrink-0 rounded-full bg-amber-400" />
      <span className="truncate">Demo Mode<span className="hidden sm:inline"> — embedded validation data</span></span>
    </button>
  );
}
