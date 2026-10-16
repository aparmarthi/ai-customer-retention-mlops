export const VALID = {
  n: 193205,
  churners: 2402,
  churnRate: 0.01243,
  window: 'Time-based holdout (last transaction after 2017-02-24)',
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

/** Mirrors artifacts/champion/deployment_policy_v2.json, which the API loads to assign `action`. */
export const POLICY = {
  policy_name: 'tiered_roi_threshold_primary',
  version: 'v2',
  tiers: [
    { action: 'contact_now', min_score: 0.68, channel: 'personal outreach with retention incentive', cost_per_contact: 5.0, assumed_save_rate: 0.2 },
    { action: 'nudge', min_score: 0.2058, channel: 'low-cost email or in-app nudge', cost_per_contact: 0.5, assumed_save_rate: 0.12 },
    { action: 'monitor', min_score: 0.0, channel: 'none; re-score next cycle', cost_per_contact: 0.0, assumed_save_rate: 0.0 },
  ],
  value_per_save: 120.0,
};

export const DEFAULTS = { valuePerSave: 120, costPerContact: 5, saveRate: 0.2 };

export const ROI_MAX_K = 1478;
export const CAPACITY_K = 10000;
export const DEFAULT_THRESHOLD = 0.68;
export const CAPACITY_THRESHOLD = 0.2058;
export const DEFAULT_SUBSCRIBERS = 1_000_000;
