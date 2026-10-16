import { DEFAULTS, DEFAULT_SUBSCRIBERS, ROI_MAX_K } from '@/data/validation';

export interface ScenarioState {
  k: number;
  valuePerSave: number;
  costPerContact: number;
  saveRate: number;
  subscribers: number;
}

export const BOUNDS = {
  k: { min: 100, max: 50_000 },
  valuePerSave: { min: 10, max: 500 },
  costPerContact: { min: 0.1, max: 50 },
  saveRate: { min: 0.01, max: 0.6 },
  subscribers: { min: 100_000, max: 50_000_000 },
} as const;

export const DEFAULT_SCENARIO: ScenarioState = {
  k: ROI_MAX_K,
  valuePerSave: DEFAULTS.valuePerSave,
  costPerContact: DEFAULTS.costPerContact,
  saveRate: DEFAULTS.saveRate,
  subscribers: DEFAULT_SUBSCRIBERS,
};

export const clampTo = (key: keyof ScenarioState, v: number): number => {
  const b = BOUNDS[key];
  const c = Math.min(b.max, Math.max(b.min, v));
  return key === 'k' || key === 'subscribers' ? Math.round(c) : c;
};

const PARAMS: Record<keyof ScenarioState, string> = {
  k: 'k',
  valuePerSave: 'v',
  costPerContact: 'c',
  saveRate: 's',
  subscribers: 'subs',
};

export function encodeScenario(s: ScenarioState): string {
  const q = new URLSearchParams();
  (Object.keys(PARAMS) as Array<keyof ScenarioState>).forEach((key) => q.set(PARAMS[key], String(Number(s[key].toFixed(4)))));
  return q.toString();
}

export function decodeScenario(query: string): ScenarioState {
  const q = new URLSearchParams(query);
  const out = { ...DEFAULT_SCENARIO };
  (Object.keys(PARAMS) as Array<keyof ScenarioState>).forEach((key) => {
    const raw = q.get(PARAMS[key]);
    if (raw === null || raw.trim() === '') return;
    const v = Number(raw);
    if (Number.isFinite(v)) out[key] = clampTo(key, v);
  });
  return out;
}

export const hashQuery = (hash: string): string => {
  const i = hash.indexOf('?');
  return i === -1 ? '' : hash.slice(i + 1);
};

export const scenarioLink = (s: ScenarioState): string =>
  `${window.location.origin}${window.location.pathname}#/simulator?${encodeScenario(s)}`;
