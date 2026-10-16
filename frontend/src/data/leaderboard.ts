export interface LeaderboardRow {
  model: string;
  prAuc: number;
  rocAuc: number;
  f1?: number;
  precision?: number;
  recall?: number;
  train: string;
  champion?: boolean;
}

// Model-family benchmark (random 80/20 split, 6.39% prevalence). Not comparable to time-based validation.
export const LEADERBOARD: LeaderboardRow[] = [
  { model: 'LightGBM', prAuc: 0.8887, rocAuc: 0.9894, f1: 0.7845, precision: 0.6805, recall: 0.926, train: '~3 min CPU', champion: true },
  { model: 'XGBoost', prAuc: 0.8771, rocAuc: 0.9875, train: '~2 min CPU' },
  { model: 'CatBoost', prAuc: 0.8737, rocAuc: 0.9865, train: '3.7 min GPU' },
  { model: 'FT-Transformer', prAuc: 0.8214, rocAuc: 0.9824, train: '23 min GPU/AMP' },
  { model: 'Random Forest', prAuc: 0.7935, rocAuc: 0.9782, train: '5 min CPU' },
  { model: 'NODE', prAuc: 0.7719, rocAuc: 0.9737, train: '11 min GPU' },
  { model: 'TabNet', prAuc: 0.5233, rocAuc: 0.9085, train: '32 min GPU (5 failed runs)' },
];

// Champion on the time-based holdout, vs the 1.24% base rate.
export const LIFT = { prAuc: 43.4, at5k: 21.6, at10k: 14.5 };

// 10% random sample, fixed 2017-01-31 cutoff: a different validation set from the champion's.
export const SAGEMAKER = { rocAuc: 0.9484, prAuc: 0.4707, f1: 0.4658, instance: 'ml.m5.large', registry: 'SageMaker Model Registry' };
