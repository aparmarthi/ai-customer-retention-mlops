import {
  breakEvenCost,
  breakEvenSaveRate,
  liftAtK,
  net,
  netRoi,
  precisionAtK,
  randomNet,
  recallAtK,
  returnOnSpend,
  revenueAtRisk,
  scale,
  spend,
  tieredNet,
  tpAtK,
  valueSaved,
} from '@/lib/economics';
import { sampleSizePerArm } from '@/lib/stats';

export interface SanityCheck {
  label: string;
  actual: number;
  expected: number;
  tolerance: number;
}

export interface SanityResult extends SanityCheck {
  pass: boolean;
}

const V = 120;
const C = 5;
const S = 0.2;
const K = 1478;
const SUBS = 1_000_000;

export function sanityChecks(): SanityCheck[] {
  const tier = tieredNet(V, C, S);
  return [
    { label: 'True positives @ K=1,478', actual: tpAtK(K), expected: 1044, tolerance: 0.5 },
    { label: 'Precision @ K', actual: precisionAtK(K), expected: 0.706, tolerance: 0.0005 },
    { label: 'Recall @ K', actual: recallAtK(K), expected: 0.435, tolerance: 0.0005 },
    { label: 'Value saved', actual: valueSaved(K, V, S), expected: 25056, tolerance: 1 },
    { label: 'Outreach spend', actual: spend(K, C), expected: 7390, tolerance: 0.01 },
    { label: 'Net value', actual: net(K, V, C, S), expected: 17666, tolerance: 1 },
    { label: 'Return on spend (gross)', actual: returnOnSpend(K, V, C, S), expected: 3.39, tolerance: 0.01 },
    { label: 'Net ROI', actual: netRoi(K, V, C, S), expected: 2.39, tolerance: 0.01 },
    { label: 'Top-10K net at incentive cost', actual: net(10000, V, C, S), expected: -6776, tolerance: 1 },
    { label: 'Break-even save rate', actual: breakEvenSaveRate(K, V, C), expected: 0.059, tolerance: 0.0005 },
    { label: 'Break-even contact cost', actual: breakEvenCost(K, V, S), expected: 16.95, tolerance: 0.01 },
    { label: 'Random targeting net', actual: randomNet(K, V, C, S), expected: -6950, tolerance: 2 },
    { label: 'Net per cycle @ 1M subs', actual: scale(net(K, V, C, S), SUBS), expected: 91436, tolerance: 50 },
    { label: 'Net per year @ 1M subs', actual: scale(net(K, V, C, S), SUBS) * 12, expected: 1_097_000, tolerance: 1000 },
    { label: 'Revenue at risk @ 1M subs', actual: scale(revenueAtRisk(V), SUBS), expected: 1_490_000, tolerance: 5000 },
    { label: 'Lift @ K=10,000', actual: liftAtK(10000), expected: 14.5, tolerance: 0.05 },
    { label: 'Tier 1 net', actual: tier.net1, expected: 17666, tolerance: 1 },
    { label: 'Tier 2 net', actual: tier.net2, expected: 6639.8, tolerance: 1 },
    { label: 'Tiered combined net', actual: tier.combined, expected: 24305.8, tolerance: 1 },
    { label: 'Tiered uplift vs single tier', actual: tier.upliftVsSingle, expected: 0.376, tolerance: 0.001 },
    { label: 'A/B sample size per arm', actual: sampleSizePerArm(0.18, 0.03, 0.05, 0.8)?.nPerArm ?? NaN, expected: 2402, tolerance: 0 },
  ];
}

export function runSanityChecks(): SanityResult[] {
  return sanityChecks().map((c) => ({ ...c, pass: Math.abs(c.actual - c.expected) <= c.tolerance }));
}
