import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  COLD_START_TIMEOUT_MS,
  WARM_TIMEOUT_MS,
  getHealth,
  isHealth,
  normalizeBase,
  resolveBaseUrl,
  saveBaseUrl,
  type HealthResponse,
} from '@/lib/api';

export type ApiStatus = 'waking' | 'checking' | 'live' | 'demo';

interface ApiContextValue {
  baseUrl: string;
  status: ApiStatus;
  health: HealthResponse | null;
  healthLatencyMs: number | null;
  lastError: string | null;
  isLive: boolean;
  recheck: () => void;
  updateBaseUrl: (url: string | null) => void;
}

const ApiContext = createContext<ApiContextValue | null>(null);

export function ApiProvider({ children }: { children: ReactNode }) {
  const [baseUrl, setBaseUrl] = useState(resolveBaseUrl);
  const [status, setStatus] = useState<ApiStatus>('waking');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLatencyMs, setHealthLatencyMs] = useState<number | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const firstCall = useRef(true);
  const seq = useRef(0);

  const ping = useCallback(async (base: string) => {
    const id = ++seq.current;
    const cold = firstCall.current;
    firstCall.current = false;
    setStatus(cold ? 'waking' : 'checking');
    try {
      const { data, latencyMs } = await getHealth(base, cold ? COLD_START_TIMEOUT_MS : WARM_TIMEOUT_MS);
      if (id !== seq.current) return;
      if (!isHealth(data)) throw new Error('Unexpected /health response');
      setHealth(data);
      setHealthLatencyMs(latencyMs);
      setLastError(null);
      setStatus('live');
    } catch (err) {
      if (id !== seq.current) return;
      setHealth(null);
      setHealthLatencyMs(null);
      setLastError(err instanceof Error ? err.message : 'Health check failed');
      setStatus('demo');
    }
  }, []);

  useEffect(() => {
    ping(baseUrl);
  }, [baseUrl, ping]);

  const recheck = useCallback(() => ping(baseUrl), [baseUrl, ping]);

  const updateBaseUrl = useCallback(
    (url: string | null) => {
      saveBaseUrl(url);
      const next = url ? normalizeBase(url) : resolveBaseUrl();
      if (next === baseUrl) ping(next);
      else setBaseUrl(next);
    },
    [baseUrl, ping],
  );

  return (
    <ApiContext.Provider
      value={{ baseUrl, status, health, healthLatencyMs, lastError, isLive: status === 'live', recheck, updateBaseUrl }}
    >
      {children}
    </ApiContext.Provider>
  );
}

export function useApi(): ApiContextValue {
  const ctx = useContext(ApiContext);
  if (!ctx) throw new Error('useApi must be used inside ApiProvider');
  return ctx;
}
