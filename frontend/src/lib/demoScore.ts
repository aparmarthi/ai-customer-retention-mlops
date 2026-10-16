import type { CustomerRecord } from '@/data/samples';
import { ymdToDays } from '@/lib/dates';

// The KKBox data ends here; expiry and recency are measured against it.
const CUTOFF = 20170228;
const BASE_LOGIT = -3;
const MIN_IMPACT = 0.002;

const num = (r: CustomerRecord, key: string): number | undefined => {
  const v = r[key];
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isFinite(n) ? n : undefined;
};

const isYmd = (v: number | undefined): v is number => v !== undefined && v >= 19000101 && v <= 21001231;

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

const pct = (x: number) => `${Math.round(x * 100)}%`;

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

interface Term {
  id: string;
  text: string;
  logit: number;
  // False when the input was missing and a default was assumed; still counts toward the score.
  provided: boolean;
}

function terms(r: CustomerRecord): Term[] {
  const autoRate = num(r, 'auto_renew_rate');
  const cancelRate = num(r, 'cancel_rate');
  const txn = num(r, 'txn_cnt');
  const active = num(r, 'log_active_days');
  const list = num(r, 'plan_list_price_mean');
  const paid = num(r, 'actual_paid_mean');
  const fullShare = num(r, 'listen_full_share');
  const expire = num(r, 'membership_expire_date_max');
  const lastListen = num(r, 'log_last_date');
  const secs = num(r, 'total_secs_mean_per_active_day');
  const cutoff = ymdToDays(CUTOFF);
  const out: Term[] = [];

  const a = autoRate ?? 0.5;
  out.push({ id: 'auto_renew_rate', text: `Auto-renew on ${pct(clamp(a, 0, 1))} of payments`, logit: 2.4 * (1 - clamp(a, 0, 1)), provided: autoRate !== undefined });

  const c = cancelRate ?? 0;
  out.push({ id: 'cancel_rate', text: `Cancelled ${pct(clamp(c, 0, 1))} of transactions`, logit: 1.6 * clamp(c, 0, 1), provided: cancelRate !== undefined });

  if (isYmd(expire)) {
    const gap = ymdToDays(expire) - cutoff;
    const logit = gap < 0 ? 1.2 + 0.03 * Math.min(-gap, 30) : gap > 45 ? -1.2 : 0.3 - 0.015 * gap;
    const text = gap < 0 ? `Membership lapsed ${plural(-gap, 'day')} ago` : gap === 0 ? 'Membership expires today' : `Membership runs ${plural(gap, 'more day')}`;
    out.push({ id: 'membership_expire_date_max', text, logit, provided: true });
  }

  if (isYmd(lastListen)) {
    const since = cutoff - ymdToDays(lastListen);
    out.push({ id: 'log_last_date', text: `Last listened ${plural(Math.max(0, since), 'day')} ago`, logit: 0.045 * clamp(since - 7, 0, 60), provided: true });
  }

  const t = txn ?? 3;
  out.push({ id: 'txn_cnt', text: `${plural(t, 'transaction')} on record`, logit: -0.06 * clamp(t, 0, 20) + (t <= 2 ? 0.7 : 0), provided: txn !== undefined });

  const d = active ?? 30;
  out.push({ id: 'log_active_days', text: `${plural(d, 'active listening day')}`, logit: -0.0015 * clamp(d, 0, 400) + (d < 15 ? 0.6 : 0), provided: active !== undefined });

  if (list !== undefined && paid !== undefined && list > 0 && paid < 0.75 * list) {
    out.push({ id: 'actual_paid_mean', text: `Paying ${pct(1 - paid / list)} below list price`, logit: 0.6, provided: true });
  }

  if (fullShare !== undefined) {
    out.push({ id: 'listen_full_share', text: `${pct(clamp(fullShare, 0, 1))} of plays heard to the end`, logit: 1.2 * (0.7 - clamp(fullShare, 0, 1)), provided: true });
  }

  if (secs !== undefined) {
    out.push({ id: 'total_secs_mean_per_active_day', text: `${(Math.max(0, secs) / 3600).toFixed(1)} h listening per active day`, logit: 0.3 - 0.00008 * clamp(secs, 0, 10000), provided: true });
  }

  return out;
}

const totalLogit = (ts: Term[]) => ts.reduce((z, t) => z + t.logit, BASE_LOGIT);

const round4 = (p: number) => Math.round(p * 10000) / 10000;

/**
 * Transparent rule-of-thumb churn estimate used only in Demo Mode for records
 * without a recorded model score. It is NOT the trained model.
 */
export function demoEstimate(r: CustomerRecord): number {
  return round4(sigmoid(totalLogit(terms(r))));
}

export const BASELINE_RISK = sigmoid(BASE_LOGIT);

export interface RiskFactor {
  id: string;
  text: string;
  /** Share of (estimate - baseline) attributed to this factor; positive raises risk. */
  impact: number;
}

export interface Explanation {
  estimate: number;
  factors: RiskFactor[];
}

export function explainRecord(r: CustomerRecord, limit = 5): Explanation {
  const ts = terms(r);
  const z = totalLogit(ts);
  const p = sigmoid(z);
  // Average slope of the sigmoid between baseline and this customer, so impacts sum exactly to p - baseline.
  const slope = Math.abs(z - BASE_LOGIT) < 1e-9 ? p * (1 - p) : (p - BASELINE_RISK) / (z - BASE_LOGIT);
  const factors = ts
    .filter((t) => t.provided)
    .map((t) => ({ id: t.id, text: t.text, impact: t.logit * slope }))
    .filter((f) => Math.abs(f.impact) >= MIN_IMPACT)
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, limit);
  return { estimate: round4(p), factors };
}

export const fmtImpact = (impact: number): string => {
  const pts = Math.round(Math.abs(impact) * 100);
  return `${impact >= 0 ? '+' : '\u2212'}${pts === 0 ? '<1 pt' : `${pts} pts`}`;
};

export const summarizeFactors = (factors: RiskFactor[]): string =>
  factors.map((f) => `${f.text} (${fmtImpact(f.impact)})`).join('; ');
