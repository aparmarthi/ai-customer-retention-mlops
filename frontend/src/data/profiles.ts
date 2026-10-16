import type { CustomerRecord, Sample } from '@/data/samples';
import { addDaysYmd } from '@/lib/dates';
import { demoEstimate } from '@/lib/demoScore';

const CUTOFF = 20170228;

type Rng = () => number;

// mulberry32: deterministic so the example batch is identical on every load.
function seeded(seed: number): Rng {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const between = (rng: Rng, lo: number, hi: number) => lo + rng() * (hi - lo);
const intBetween = (rng: Rng, lo: number, hi: number) => Math.round(between(rng, lo, hi));
const pick = <T,>(rng: Rng, xs: readonly T[]): T => xs[Math.floor(rng() * xs.length)];
const r2 = (x: number) => Math.round(x * 100) / 100;
const r3 = (x: number) => Math.round(x * 1000) / 1000;

interface Core {
  txnCnt: number;
  cancelCnt: number;
  autoRenewCnt: number;
  listPrice: number;
  paidPrice: number;
  tenureDays: number;
  expireInDays: number;
  activeDays: number;
  silentDays: number;
  secsPerDay: number;
  unqPerDay: number;
  fullShare: number;
}

function buildRecord(rng: Rng, c: Core): CustomerRecord {
  const firstTxn = addDaysYmd(CUTOFF, -c.tenureDays);
  const lastTxn = addDaysYmd(CUTOFF, -intBetween(rng, 1, 28));
  const logLast = addDaysYmd(CUTOFF, -c.silentDays);
  const logFirst = addDaysYmd(logLast, -Math.max(c.activeDays, Math.round(c.tenureDays * between(rng, 0.7, 1))));
  const regDate = addDaysYmd(firstTxn, -intBetween(rng, 0, 400));
  const unq = Math.round(c.activeDays * c.unqPerDay);
  const plays = Math.round(unq * between(rng, 1.15, 2.2));
  const full = Math.round(plays * c.fullShare);
  const rest = plays - full;
  const age = rng() < 0.3 ? 0 : intBetween(rng, 18, 52);
  return {
    city: pick(rng, [1, 1, 1, 4, 5, 6, 13, 13, 15, 22]),
    bd: age,
    gender: age === 0 ? 'unknown' : pick(rng, ['male', 'female'] as const),
    registered_via: pick(rng, [3, 4, 7, 9, 9]),
    registration_init_time: regDate,
    txn_cnt: c.txnCnt,
    cancel_cnt: c.cancelCnt,
    cancel_rate: r3(c.cancelCnt / c.txnCnt),
    auto_renew_cnt: c.autoRenewCnt,
    auto_renew_rate: r3(c.autoRenewCnt / c.txnCnt),
    plan_list_price_mean: c.listPrice,
    plan_list_price_max: c.listPrice,
    plan_list_price_min: c.paidPrice === 0 ? 0 : c.listPrice,
    actual_paid_mean: c.paidPrice,
    actual_paid_max: Math.max(c.paidPrice, c.paidPrice === 0 ? 0 : c.listPrice),
    actual_paid_min: c.paidPrice,
    payment_method_nunique: rng() < 0.85 ? 1 : 2,
    txn_first_date: firstTxn,
    txn_tenure_days_approx: lastTxn - firstTxn,
    membership_expire_date_max: addDaysYmd(CUTOFF, c.expireInDays),
    log_row_cnt: c.activeDays,
    log_first_date: logFirst,
    log_last_date: logLast,
    log_active_days: c.activeDays,
    log_span_days_approx: logLast - logFirst,
    total_secs_sum: Math.round(c.activeDays * c.secsPerDay),
    total_secs_mean_per_active_day: r2(c.secsPerDay),
    total_secs_max_per_day: r2(c.secsPerDay * between(rng, 2.5, 5)),
    num_unq_sum: unq,
    num_unq_mean_per_active_day: r2(c.unqPerDay),
    num_unq_max_per_day: Math.round(c.unqPerDay * between(rng, 3, 6)),
    num_25_sum: Math.round(rest * 0.55),
    num_50_sum: Math.round(rest * 0.2),
    num_75_sum: Math.round(rest * 0.12),
    num_985_sum: Math.round(rest * 0.13),
    num_100_sum: full,
    listen_full_share: r3(full / Math.max(plays, 1)),
    listens_per_unique_track: r3(plays / Math.max(unq, 1)),
  };
}

interface Archetype {
  segment: string;
  label: string;
  core: (rng: Rng) => Core;
}

const ARCHETYPES: Archetype[] = [
  {
    segment: 'loyal_heavy',
    label: 'Loyal auto-renewer, daily listener',
    core: (rng) => {
      const txn = intBetween(rng, 14, 26);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: txn, listPrice: 149, paidPrice: 149, tenureDays: txn * 30, expireInDays: intBetween(rng, 3, 30), activeDays: intBetween(rng, 300, 650), silentDays: intBetween(rng, 0, 2), secsPerDay: between(rng, 6000, 11000), unqPerDay: between(rng, 20, 35), fullShare: between(rng, 0.75, 0.88) };
    },
  },
  {
    segment: 'prepaid_annual',
    label: 'Prepaid annual plan, renews far ahead',
    core: (rng) => {
      const txn = intBetween(rng, 2, 4);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: 0, listPrice: 1788, paidPrice: 1788, tenureDays: intBetween(rng, 300, 700), expireInDays: intBetween(rng, 90, 300), activeDays: intBetween(rng, 120, 260), silentDays: intBetween(rng, 0, 6), secsPerDay: between(rng, 3500, 7000), unqPerDay: between(rng, 14, 25), fullShare: between(rng, 0.65, 0.8) };
    },
  },
  {
    segment: 'steady_auto',
    label: 'Steady auto-renewer, moderate use',
    core: (rng) => {
      const txn = intBetween(rng, 5, 11);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: txn, listPrice: 99, paidPrice: 99, tenureDays: txn * 30, expireInDays: intBetween(rng, 1, 25), activeDays: intBetween(rng, 60, 180), silentDays: intBetween(rng, 0, 10), secsPerDay: between(rng, 2500, 6000), unqPerDay: between(rng, 10, 22), fullShare: between(rng, 0.6, 0.78) };
    },
  },
  {
    segment: 'past_cancel_auto',
    label: 'Auto-renewer with a past cancellation',
    core: (rng) => {
      const txn = intBetween(rng, 8, 16);
      const cancels = intBetween(rng, 1, 2);
      return { txnCnt: txn, cancelCnt: cancels, autoRenewCnt: txn - cancels, listPrice: 149, paidPrice: 149, tenureDays: txn * 30, expireInDays: intBetween(rng, 2, 25), activeDays: intBetween(rng, 90, 260), silentDays: intBetween(rng, 0, 12), secsPerDay: between(rng, 3000, 7500), unqPerDay: between(rng, 12, 26), fullShare: between(rng, 0.6, 0.8) };
    },
  },
  {
    segment: 'lapsing_auto',
    label: 'Auto-renewer gone quiet for weeks',
    core: (rng) => {
      const txn = intBetween(rng, 4, 10);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: txn, listPrice: 149, paidPrice: 149, tenureDays: txn * 30, expireInDays: intBetween(rng, 1, 20), activeDays: intBetween(rng, 25, 90), silentDays: intBetween(rng, 35, 60), secsPerDay: between(rng, 1200, 3500), unqPerDay: between(rng, 6, 14), fullShare: between(rng, 0.45, 0.65) };
    },
  },
  {
    segment: 'promo_trial',
    label: 'Free-trial newcomer, half auto-renew',
    core: (rng) => ({ txnCnt: 2, cancelCnt: 0, autoRenewCnt: 1, listPrice: 149, paidPrice: 0, tenureDays: intBetween(rng, 30, 60), expireInDays: intBetween(rng, 2, 20), activeDays: intBetween(rng, 18, 45), silentDays: intBetween(rng, 0, 15), secsPerDay: between(rng, 2000, 5000), unqPerDay: between(rng, 12, 30), fullShare: between(rng, 0.5, 0.7) }),
  },
  {
    segment: 'manual_active',
    label: 'Manual renewer, still listening',
    core: (rng) => {
      const txn = intBetween(rng, 5, 12);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: 0, listPrice: 180, paidPrice: 180, tenureDays: txn * 30, expireInDays: intBetween(rng, 1, 25), activeDays: intBetween(rng, 80, 220), silentDays: intBetween(rng, 0, 6), secsPerDay: between(rng, 3000, 7000), unqPerDay: between(rng, 12, 25), fullShare: between(rng, 0.6, 0.78) };
    },
  },
  {
    segment: 'downgrader',
    label: 'Switched off auto-renew, discounted, listening less',
    core: (rng) => {
      const txn = intBetween(rng, 5, 9);
      const auto = intBetween(rng, 1, 3);
      return { txnCnt: txn, cancelCnt: 1, autoRenewCnt: auto, listPrice: 149, paidPrice: 99, tenureDays: txn * 30, expireInDays: intBetween(rng, 2, 18), activeDays: intBetween(rng, 40, 100), silentDays: intBetween(rng, 18, 38), secsPerDay: between(rng, 1800, 4000), unqPerDay: between(rng, 8, 18), fullShare: between(rng, 0.5, 0.68) };
    },
  },
  {
    segment: 'manual_expiring',
    label: 'Manual renewer, just expired, fading use',
    core: (rng) => {
      const txn = intBetween(rng, 3, 7);
      return { txnCnt: txn, cancelCnt: 0, autoRenewCnt: 0, listPrice: 180, paidPrice: 180, tenureDays: txn * 30, expireInDays: -intBetween(rng, 2, 20), activeDays: intBetween(rng, 25, 70), silentDays: intBetween(rng, 15, 35), secsPerDay: between(rng, 1500, 4000), unqPerDay: between(rng, 8, 18), fullShare: between(rng, 0.45, 0.65) };
    },
  },
  {
    segment: 'cancelled_new',
    label: 'Brand-new, already cancelled',
    core: (rng) => ({ txnCnt: 1, cancelCnt: 1, autoRenewCnt: 0, listPrice: 100, paidPrice: 100, tenureDays: intBetween(rng, 20, 45), expireInDays: -intBetween(rng, 5, 15), activeDays: intBetween(rng, 2, 8), silentDays: intBetween(rng, 30, 45), secsPerDay: between(rng, 800, 2500), unqPerDay: between(rng, 8, 20), fullShare: between(rng, 0.3, 0.55) }),
  },
  {
    segment: 'expired_single',
    label: 'One-off purchase, expired, inactive',
    core: (rng) => ({ txnCnt: 1, cancelCnt: 0, autoRenewCnt: 0, listPrice: 180, paidPrice: 180, tenureDays: intBetween(rng, 40, 60), expireInDays: -intBetween(rng, 15, 28), activeDays: intBetween(rng, 4, 12), silentDays: intBetween(rng, 40, 60), secsPerDay: between(rng, 3000, 9000), unqPerDay: between(rng, 25, 55), fullShare: between(rng, 0.3, 0.45) }),
  },
];

export const PROFILE_SAMPLES: Sample[] = ARCHETYPES.map((a, i) => {
  const rng = seeded(1000 + i);
  const record = buildRecord(rng, a.core(rng));
  return { label: a.label, expected: demoEstimate(record), record, estimated: true };
});

export const EXAMPLE_BATCH_SIZE = 60;

export const EXAMPLE_BATCH: CustomerRecord[] = (() => {
  const rng = seeded(20170228);
  const rows = Array.from({ length: EXAMPLE_BATCH_SIZE }, (_, i) => {
    const a = ARCHETYPES[i % ARCHETYPES.length];
    return { segment: a.segment, ...buildRecord(rng, a.core(rng)) };
  });
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows.map((r, i) => ({ customer_id: `demo-${String(i + 1).padStart(3, '0')}`, ...r }));
})();
