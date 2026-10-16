import { CURVE, VALID, type CurvePoint } from '@/data/validation';

export interface Assumptions {
  valuePerSave: number;
  costPerContact: number;
  saveRate: number;
}

export function tpAtK(k: number, curve: CurvePoint[] = CURVE): number {
  if (k <= 0) return 0;
  const last = curve[curve.length - 1];
  if (k >= last.k) return last.tp;
  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1];
    const b = curve[i];
    if (k <= b.k) return a.tp + ((k - a.k) / (b.k - a.k)) * (b.tp - a.tp);
  }
  return last.tp;
}

export const precisionAtK = (k: number): number => (k > 0 ? tpAtK(k) / k : 0);
export const recallAtK = (k: number): number => tpAtK(k) / VALID.churners;
export const liftAtK = (k: number): number => precisionAtK(k) / VALID.churnRate;

export const valueSaved = (k: number, V: number, S: number): number => tpAtK(k) * S * V;
export const spend = (k: number, C: number): number => k * C;

export const net = (k: number, V: number, C: number, S: number): number => tpAtK(k) * S * V - k * C;

export const randomNet = (k: number, V: number, C: number, S: number): number =>
  k * VALID.churnRate * S * V - k * C;

export const scale = (x: number, subs: number): number => (x * subs) / VALID.n;

export const breakEvenSaveRate = (k: number, V: number, C: number): number => (k * C) / (tpAtK(k) * V);

export const breakEvenCost = (k: number, V: number, S: number): number => (tpAtK(k) * S * V) / k;

// Gross: retained value per $1 of contact spend. Net ROI subtracts the spend first.
export const returnOnSpend = (k: number, V: number, C: number, S: number): number =>
  (tpAtK(k) * S * V) / (k * C);

export const netRoi = (k: number, V: number, C: number, S: number): number => net(k, V, C, S) / (k * C);

export const revenueAtRisk = (V: number): number => VALID.churners * V;

export function knownThreshold(k: number): number | undefined {
  return CURVE.find((p) => p.k === Math.round(k) && p.threshold !== undefined)?.threshold;
}

export interface Scenario {
  k: number;
  tp: number;
  precision: number;
  recall: number;
  valueSaved: number;
  spend: number;
  net: number;
  randomNet: number;
  returnOnSpend: number;
  netRoi: number;
  breakEvenSaveRate: number;
  breakEvenCost: number;
}

export function scenario(k: number, a: Assumptions): Scenario {
  const { valuePerSave: V, costPerContact: C, saveRate: S } = a;
  return {
    k,
    tp: tpAtK(k),
    precision: precisionAtK(k),
    recall: recallAtK(k),
    valueSaved: valueSaved(k, V, S),
    spend: spend(k, C),
    net: net(k, V, C, S),
    randomNet: randomNet(k, V, C, S),
    returnOnSpend: returnOnSpend(k, V, C, S),
    netRoi: netRoi(k, V, C, S),
    breakEvenSaveRate: breakEvenSaveRate(k, V, C),
    breakEvenCost: breakEvenCost(k, V, S),
  };
}

export interface SensitivityBar {
  driver: string;
  low: number;
  high: number;
  lowLabel: string;
  highLabel: string;
}

// Net is linear in S·V, so save rate and value per save always swing net by the same amount.
// `leaders` lists every driver tied for the largest swing so the UI never names just one of a tie.
export function sensitivity(k: number, a: Assumptions, pct = 0.5): { base: number; bars: SensitivityBar[]; leaders: string[] } {
  const { valuePerSave: V, costPerContact: C, saveRate: S } = a;
  const base = net(k, V, C, S);
  const bars: SensitivityBar[] = [
    { driver: 'Save rate', low: net(k, V, C, S * (1 - pct)) - base, high: net(k, V, C, S * (1 + pct)) - base, lowLabel: '-50%', highLabel: '+50%' },
    { driver: 'Value per save', low: net(k, V * (1 - pct), C, S) - base, high: net(k, V * (1 + pct), C, S) - base, lowLabel: '-50%', highLabel: '+50%' },
    { driver: 'Cost per contact', low: net(k, V, C * (1 + pct), S) - base, high: net(k, V, C * (1 - pct), S) - base, lowLabel: '+50%', highLabel: '-50%' },
  ];
  const swing = (b: SensitivityBar) => Math.abs(b.high - b.low);
  bars.sort((x, y) => swing(y) - swing(x));
  const leaders = bars.filter((b) => swing(bars[0]) - swing(b) < 1e-6 * Math.max(1, swing(bars[0]))).map((b) => b.driver);
  return { base, bars, leaders };
}

export function joinDrivers(drivers: string[]): string {
  return drivers.map((d, i) => (i === 0 ? d : d.toLowerCase())).join(' and ');
}

// Tier 1 uses the user's incentive assumptions; tier 2 is a fixed, illustrative low-cost nudge.
export const TIERS = {
  tier1: { k: 1478 },
  tier2: { kEnd: 10000, cost: 0.5, saveRate: 0.12 },
};

export function tieredNet(V: number, C: number, S: number) {
  const { tier1, tier2 } = TIERS;
  const tp1 = tpAtK(tier1.k);
  const tp2 = tpAtK(tier2.kEnd) - tp1;
  const k2 = tier2.kEnd - tier1.k;
  const net1 = tp1 * S * V - tier1.k * C;
  const net2 = tp2 * tier2.saveRate * V - k2 * tier2.cost;
  const spend = tier1.k * C + k2 * tier2.cost;
  return { net1, net2, combined: net1 + net2, k2, tp1, tp2, spend, upliftVsSingle: net1 > 0 ? (net1 + net2) / net1 - 1 : NaN };
}

export function logSpace(min: number, max: number, steps: number): number[] {
  const lo = Math.log(min);
  const hi = Math.log(max);
  return Array.from({ length: steps }, (_, i) => Math.round(Math.exp(lo + ((hi - lo) * i) / (steps - 1))));
}
