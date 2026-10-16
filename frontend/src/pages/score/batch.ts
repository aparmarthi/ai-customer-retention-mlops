import type { CustomerRecord } from '@/data/samples';
import type { BatchItem } from '@/lib/api';
import { DEFAULT_THRESHOLD } from '@/data/validation';

export interface ScoredRow {
  index: number;
  input: CustomerRecord;
  probability: number;
  rank: number | null;
  labelAt068: 0 | 1;
  inTopK: boolean;
}

export interface BatchRun {
  rows: ScoredRow[];
  headers: string[];
  k: number;
  policyUsed: string;
  thresholdUsed: number | null;
  latencyMs: number | null;
  source: 'live' | 'demo' | 'estimate';
  origin: string;
}

export function joinItems(inputs: CustomerRecord[], items: BatchItem[], k: number): ScoredRow[] {
  return items.map((it, i) => ({
    index: i,
    input: inputs[i] ?? {},
    probability: it.churn_probability,
    rank: it.rank,
    labelAt068: it.churn_probability >= DEFAULT_THRESHOLD ? 1 : 0,
    inTopK: it.rank !== null && it.rank <= k,
  }));
}

// Demo-mode stand-in for server-side ranking: rank 1 = highest probability.
export function rankLocally(probs: number[], k: number): BatchItem[] {
  const order = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const rank = new Array<number>(probs.length);
  order.forEach((o, r) => (rank[o.i] = r + 1));
  return probs.map((p, i) => ({
    churn_probability: p,
    churn_label: rank[i] <= k ? 1 : 0,
    rank: rank[i],
    threshold_used: null,
    policy_used: `top_k(${k})`,
    metadata: {},
  }));
}

export function isBatchResponse(x: unknown): x is { n: number; policy_used: string; items: BatchItem[] } {
  return !!x && typeof x === 'object' && Array.isArray((x as { items?: unknown }).items);
}
