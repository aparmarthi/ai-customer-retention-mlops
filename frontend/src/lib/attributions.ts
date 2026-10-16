import { FEATURE_GROUPS } from '@/data/features';
import { fromYmdInt } from '@/lib/dates';
import type { RiskFactor } from '@/lib/demoScore';
import type { ModelExplanation } from '@/lib/api';

const DEFS = new Map(FEATURE_GROUPS.flatMap((g) => g.features).map((f) => [f.key, f]));
const num = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (p: number) => Math.log(p / (1 - p));

function valueText(feature: string, value: ModelExplanation['value']): string {
  if (value === null || value === undefined || value === '') return 'missing';
  if (DEFS.get(feature)?.kind === 'date') return fromYmdInt(value) || String(value);
  return typeof value === 'number' ? num.format(value) : String(value);
}

/**
 * Convert the API's TreeSHAP contributions (log-odds) into probability points, using the
 * average slope of the sigmoid between the model's base value and this score. The full set
 * of contributions sums exactly to score - sigmoid(base); the API returns the top few.
 */
export function toModelFactors(explanations: ModelExplanation[], baseValue: number, probability: number): RiskFactor[] {
  const p = Math.min(Math.max(probability, 1e-6), 1 - 1e-6);
  const dz = logit(p) - baseValue;
  const slope = Math.abs(dz) < 1e-9 ? p * (1 - p) : (p - sigmoid(baseValue)) / dz;
  return explanations.map((e) => ({
    id: e.feature,
    text: `${DEFS.get(e.feature)?.label ?? e.feature}: ${valueText(e.feature, e.value)}`,
    impact: e.contribution * slope,
  }));
}

export const modelBaseline = (baseValue: number): number => sigmoid(baseValue);
