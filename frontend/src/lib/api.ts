import type { CustomerRecord } from '@/data/samples';

export const DEFAULT_API_URL = 'https://churn-api-995576020018.us-central1.run.app';
const STORAGE_KEY = 'cdi.apiBaseUrl';
export const COLD_START_TIMEOUT_MS = 12_000;
export const WARM_TIMEOUT_MS = 6_000;
const PREDICT_TIMEOUT_MS = 60_000;

export interface HealthResponse {
  status: 'ok';
  model_loaded: boolean;
  artifacts: { model: string; threshold: string; feature_list: string; categorical_cols: string };
  n_features: number;
  default_threshold: number;
  /** Added in API 1.1.0; absent on older deployments. */
  policy_version?: string;
}

export interface RootResponse {
  service: string;
  version: string;
  docs: string;
  health: string;
  endpoints: string[];
}

/** One TreeSHAP contribution from the trained model, in log-odds. */
export interface ModelExplanation {
  feature: string;
  value: number | string | null;
  contribution: number;
}

export interface PredictResponse {
  churn_probability: number;
  churn_label: 0 | 1;
  policy_used: 'threshold';
  threshold_used: number;
  metadata: { note: string };
  // Added in API 1.1.0; optional so older deployments still parse.
  action?: 'contact_now' | 'nudge' | 'monitor';
  explanations?: ModelExplanation[];
  base_value?: number | null;
  n_missing_features?: number;
  warnings?: string[];
}

export interface BatchItem {
  churn_probability: number;
  churn_label: 0 | 1;
  rank: number | null;
  threshold_used: number | null;
  policy_used: string;
  metadata: Record<string, unknown>;
}

export interface BatchResponse {
  n: number;
  policy_used: string;
  items: BatchItem[];
}

export interface Timed<T> {
  data: T;
  latencyMs: number;
}

export class ApiError extends Error {
  constructor(message: string, public status?: number, public latencyMs?: number) {
    super(message);
  }
}

export function envBaseUrl(): string {
  const v = import.meta.env.VITE_API_URL as string | undefined;
  return normalizeBase(v && v.trim() ? v : DEFAULT_API_URL);
}

export function normalizeBase(url: string): string {
  return url.trim().replace(/\/+$/, '');
}

export function storedBaseUrl(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveBaseUrl(url: string | null): void {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, normalizeBase(url));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable (private mode) */
  }
}

export function resolveBaseUrl(): string {
  return storedBaseUrl() ?? envBaseUrl();
}

function detailFrom(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : JSON.stringify(d)))
        .join('; ');
    }
    return JSON.stringify(detail);
  }
  return `Request failed with status ${status}`;
}

async function request<T>(base: string, path: string, init: RequestInit, timeoutMs: number): Promise<Timed<T>> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const start = performance.now();
  try {
    const res = await fetch(`${base}${path}`, { ...init, signal: controller.signal });
    const latencyMs = performance.now() - start;
    const text = await res.text();
    let body: unknown = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
    if (!res.ok) throw new ApiError(detailFrom(body, res.status), res.status, latencyMs);
    return { data: body as T, latencyMs };
  } catch (err) {
    if (err instanceof ApiError) throw err;
    const latencyMs = performance.now() - start;
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(`Timed out after ${Math.round(timeoutMs / 1000)} s`, undefined, latencyMs);
    }
    throw new ApiError('Network error: could not reach the API (check the URL or CORS).', undefined, latencyMs);
  } finally {
    clearTimeout(timer);
  }
}

export function isHealth(x: unknown): x is HealthResponse {
  return !!x && typeof x === 'object' && (x as HealthResponse).status === 'ok' && typeof (x as HealthResponse).n_features === 'number';
}

export const getHealth = (base: string, timeoutMs: number) =>
  request<HealthResponse>(base, '/health', { method: 'GET' }, timeoutMs);

export const getRoot = (base: string) => request<RootResponse>(base, '/', { method: 'GET' }, WARM_TIMEOUT_MS);

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const predict = (base: string, record: CustomerRecord, threshold: number) =>
  request<PredictResponse>(base, '/predict', json({ record, policy: 'threshold', threshold }), PREDICT_TIMEOUT_MS);

export const predictBatchRecords = (base: string, records: CustomerRecord[], k: number) =>
  request<BatchResponse>(base, '/predict_batch', json({ records, policy: 'top_k', k }), PREDICT_TIMEOUT_MS);

export function predictBatchFile(base: string, file: File, k: number) {
  const form = new FormData();
  form.append('file', file, file.name);
  const qs = new URLSearchParams({ policy: 'top_k', k: String(k) });
  return request<BatchResponse>(base, `/predict_batch?${qs}`, { method: 'POST', body: form }, PREDICT_TIMEOUT_MS * 3);
}
