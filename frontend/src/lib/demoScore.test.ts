import { describe, expect, it } from 'vitest';
import { PROFILE_SAMPLES } from '@/data/profiles';
import { SAMPLES } from '@/data/samples';
import { BASELINE_RISK, demoEstimate, explainRecord, fmtImpact } from '@/lib/demoScore';

describe('explainRecord', () => {
  it('agrees with demoEstimate', () => {
    for (const s of [...PROFILE_SAMPLES, ...SAMPLES]) expect(explainRecord(s.record).estimate).toBe(demoEstimate(s.record));
  });

  it('returns at most the requested number of factors, largest first', () => {
    for (const s of PROFILE_SAMPLES) {
      const { factors } = explainRecord(s.record, 3);
      expect(factors.length).toBeLessThanOrEqual(3);
      for (let i = 1; i < factors.length; i++) expect(Math.abs(factors[i - 1].impact)).toBeGreaterThanOrEqual(Math.abs(factors[i].impact));
    }
  });

  it('blames missing auto-renew for a high-risk customer and credits it for a loyal one', () => {
    const risky = explainRecord({ auto_renew_rate: 0, cancel_rate: 0.5, txn_cnt: 1 });
    expect(risky.factors[0].id).toBe('auto_renew_rate');
    expect(risky.factors[0].impact).toBeGreaterThan(0);
    const loyal = explainRecord({ auto_renew_rate: 1, txn_cnt: 20, membership_expire_date_max: 20170601 });
    expect(loyal.factors.some((f) => f.impact < 0)).toBe(true);
    expect(loyal.factors.find((f) => f.id === 'auto_renew_rate')).toBeUndefined();
  });

  it('attributes the full gap from baseline when every input is provided', () => {
    for (const s of PROFILE_SAMPLES) {
      const { estimate, factors } = explainRecord(s.record, 99);
      const shown = factors.reduce((t, f) => t + f.impact, 0);
      expect(Math.abs(shown - (estimate - BASELINE_RISK))).toBeLessThan(0.02);
    }
  });

  it('never explains with inputs the record did not provide', () => {
    expect(explainRecord({}).factors).toEqual([]);
  });

  it('formats impact as signed percentage points', () => {
    expect(fmtImpact(0.214)).toBe('+21 pts');
    expect(fmtImpact(-0.05)).toBe('\u22125 pts');
  });
});
