import { CAPACITY_THRESHOLD, DEFAULT_THRESHOLD } from '@/data/validation';

export type RiskBand = 'contact' | 'capacity' | 'monitor';

export const RISK_BANDS: Record<RiskBand, { label: string; action: string; text: string; bg: string; hex: string }> = {
  contact: { label: 'Contact now', action: 'Personal outreach with retention incentive', text: 'text-rose-300', bg: 'border-rose-500/30 bg-rose-500/10', hex: '#fb7185' },
  capacity: { label: 'Capacity tier', action: 'Low-cost email or in-app nudge', text: 'text-amber-300', bg: 'border-amber-500/30 bg-amber-500/10', hex: '#fbbf24' },
  monitor: { label: 'Monitor', action: 'No action; re-score next cycle', text: 'text-emerald-300', bg: 'border-emerald-500/30 bg-emerald-500/10', hex: '#34d399' },
};

export function riskBand(p: number): RiskBand {
  if (p >= DEFAULT_THRESHOLD) return 'contact';
  if (p >= CAPACITY_THRESHOLD) return 'capacity';
  return 'monitor';
}
