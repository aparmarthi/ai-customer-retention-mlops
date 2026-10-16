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

export const roiMultiple = (k: number, V: number, C: number, S: number): number =>
  (tpAtK(k) * S * V) / (k * C);

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
  roi: number;
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
    roi: roiMultiple(k, V, C, S),
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

export function sensitivity(k: number, a: Assumptions, pct = 0.5): { base: number; bars: SensitivityBar[] } {
  const { valuePerSave: V, costPerContact: C, saveRate: S } = a;
  const base = net(k, V, C, S);
  const bars: SensitivityBar[] = [
    { driver: 'Save rate', low: net(k, V, C, S * (1 - pct)) - base, high: net(k, V, C, S * (1 + pct)) - base, lowLabel: '-50%', highLabel: '+50%' },
    { driver: 'Value per save', low: net(k, V * (1 - pct), C, S) - base, high: net(k, V * (1 + pct), C, S) - base, lowLabel: '-50%', highLabel: '+50%' },
    { driver: 'Cost per contact', low: net(k, V, C * (1 + pct), S) - base, high: net(k, V, C * (1 - pct), S) - base, lowLabel: '+50%', highLabel: '-50%' },
  ];
  bars.sort((x, y) => Math.abs(y.high - y.low) - Math.abs(x.high - x.low));
  return { base, bars };
}

export const TIERS = {
  tier1: { k: 1478, cost: 5, saveRate: 0.2 },
  tier2: { kEnd: 10000, cost: 0.5, saveRate: 0.12 },
};

export function tieredNet(V: number) {
  const { tier1, tier2 } = TIERS;
  const tp1 = tpAtK(tier1.k);
  const tp2 = tpAtK(tier2.kEnd) - tp1;
  const k2 = tier2.kEnd - tier1.k;
  const net1 = tp1 * tier1.saveRate * V - tier1.k * tier1.cost;
  const net2 = tp2 * tier2.saveRate * V - k2 * tier2.cost;
  return { net1, net2, combined: net1 + net2, k2, tp1, tp2, upliftVsSingle: (net1 + net2) / net1 - 1 };
}

export function logSpace(min: number, max: number, steps: number): number[] {
  const lo = Math.log(min);
  const hi = Math.log(max);
  return Array.from({ length: steps }, (_, i) => Math.round(Math.exp(lo + ((hi - lo) * i) / (steps - 1))));
}
