import { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, RotateCcw, Server, X } from 'lucide-react';
import { useApi } from '@/context/ApiContext';
import { DEFAULT_API_URL, envBaseUrl, storedBaseUrl } from '@/lib/api';
import { fmtMs } from '@/lib/format';

export function SettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { baseUrl, updateBaseUrl, status, health, healthLatencyMs, lastError } = useApi();
  const [draft, setDraft] = useState(baseUrl);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDraft(baseUrl);
      setError(null);
    }
  }, [open, baseUrl]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const save = () => {
    try {
      const u = new URL(draft.trim());
      if (!/^https?:$/.test(u.protocol)) throw new Error();
    } catch {
      setError('Enter a full URL starting with http:// or https://');
      return;
    }
    updateBaseUrl(draft.trim());
  };

  const busy = status === 'waking' || status === 'checking';
  const overridden = storedBaseUrl() !== null;

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="absolute inset-0 animate-fade-in bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <aside className="absolute right-0 top-0 flex h-full w-full max-w-md animate-slide-in-right flex-col border-l border-zinc-800 bg-zinc-950 shadow-2xl">
        <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
          <div>
            <h2 id="settings-title" className="text-base font-semibold text-zinc-100">Settings</h2>
            <p className="text-xs text-zinc-500">Point the console at a different scoring service.</p>
          </div>
          <button onClick={onClose} className="btn-ghost p-2" aria-label="Close settings">
            <X className="h-4 w-4" />
          </button>
        </header>
        <div className="flex-1 space-y-6 overflow-y-auto p-6">
          <div>
            <label htmlFor="api-url" className="label">API base URL</label>
            <input
              id="api-url"
              className="input mt-2 font-mono text-xs"
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                setError(null);
              }}
              onKeyDown={(e) => e.key === 'Enter' && save()}
              placeholder={DEFAULT_API_URL}
              spellCheck={false}
            />
            {error && <p className="mt-2 text-xs text-rose-400">{error}</p>}
            <p className="mt-2 text-xs text-zinc-500">
              Saved in this browser only. Build default: <span className="break-all font-mono text-zinc-400">{envBaseUrl()}</span>
            </p>
            <div className="mt-4 flex gap-2">
              <button onClick={save} disabled={busy} className="btn-primary">
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Server className="h-4 w-4" />}
                Save &amp; check health
              </button>
              {overridden && (
                <button onClick={() => updateBaseUrl(null)} className="btn-secondary" disabled={busy}>
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </button>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <p className="label">Connection</p>
            <div className="mt-3 space-y-2 text-sm">
              {busy && (
                <p className="flex items-center gap-2 text-zinc-300">
                  <Loader2 className="h-4 w-4 animate-spin text-amber-400" />
                  {status === 'waking' ? 'Waking API (cold start)\u2026' : 'Checking /health\u2026'}
                </p>
              )}
              {status === 'live' && health && (
                <>
                  <p className="flex items-center gap-2 text-emerald-300">
                    <CheckCircle2 className="h-4 w-4" /> Connected
                    {healthLatencyMs !== null && <span className="text-xs text-zinc-500 tabular-nums">({fmtMs(healthLatencyMs)})</span>}
                  </p>
                  <dl className="grid grid-cols-2 gap-y-1 text-xs tabular-nums">
                    <dt className="text-zinc-500">Model loaded</dt>
                    <dd className="text-zinc-200">{String(health.model_loaded)}</dd>
                    <dt className="text-zinc-500">Features</dt>
                    <dd className="text-zinc-200">{health.n_features}</dd>
                    <dt className="text-zinc-500">Default threshold</dt>
                    <dd className="text-zinc-200">{health.default_threshold}</dd>
                  </dl>
                </>
              )}
              {status === 'demo' && (
                <p className="text-xs text-amber-300">
                  Demo Mode: {lastError ?? 'API unreachable'}. Pages use embedded validation data; scoring custom records is disabled.
                </p>
              )}
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
