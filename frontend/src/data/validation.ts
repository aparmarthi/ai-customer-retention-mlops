export const VALID = {
  n: 193205,
  churners: 2402,
  churnRate: 0.01243,
  window: 'Feb 2017 (1 month)',
  champion: 'LightGBM (FLAML-tuned)',
  rocAuc: 0.966,
  prAuc: 0.5392,
  f1At05: 0.3678,
};

export interface CurvePoint {
  k: number;
  tp: number;
  threshold?: number;
}

// Operating curve: contacts K -> true churners caught (TP), from champion scores ranked desc.
export const CURVE: CurvePoint[] = [
  { k: 0, tp: 0 },
  { k: 1262, tp: 992, threshold: 0.71 },
  { k: 1478, tp: 1044, threshold: 0.68 },
  { k: 5000, tp: 1345 },
  { k: 10000, tp: 1801, threshold: 0.2058 },
  { k: 20000, tp: 2179 },
  { k: 193205, tp: 2402 },
];

export const POLICY = {
  name: 'hybrid_topk_primary_roi_threshold_fallback',
  version: 'v1',
  primary: { type: 'top_k', k: 10000, note: 'Capacity-driven: retention team works the top 10K each cycle' },
  fallback: { type: 'threshold', threshold: 0.68, note: 'ROI-max cutoff when capacity is unconstrained' },
};

export const DEFAULTS = { valuePerSave: 120, costPerContact: 5, saveRate: 0.2 };

export const ROI_MAX_K = 1478;
export const CAPACITY_K = 10000;
export const DEFAULT_THRESHOLD = 0.68;
export const CAPACITY_THRESHOLD = 0.2058;
export const DEFAULT_SUBSCRIBERS = 1_000_000;
