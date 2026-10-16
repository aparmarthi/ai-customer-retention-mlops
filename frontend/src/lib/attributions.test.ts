import { describe, expect, it } from 'vitest';
import { modelBaseline, toModelFactors } from '@/lib/attributions';

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

describe('toModelFactors', () => {
  it('sums to score minus baseline when every contribution is present', () => {
    const base = -2.5;
    const contribs = [1.8, 0.9, -0.4];
    const p = sigmoid(base + contribs.reduce((a, b) => a + b, 0));
    const factors = toModelFactors(
      contribs.map((c, i) => ({ feature: `f${i}`, value: i, contribution: c })),
      base,
      p,
    );
    const total = factors.reduce((a, f) => a + f.impact, 0);
    expect(total).toBeCloseTo(p - modelBaseline(base), 10);
  });

  it('keeps the sign of each contribution', () => {
    const [up, down] = toModelFactors(
      [
        { feature: 'cancel_rate', value: 0.5, contribution: 1.2 },
        { feature: 'auto_renew_rate', value: 1, contribution: -0.7 },
      ],
      -3,
      sigmoid(-2.5),
    );
    expect(up.impact).toBeGreaterThan(0);
    expect(down.impact).toBeLessThan(0);
  });

  it('labels features and formats dates and missing values', () => {
    const [date, missing, unknown] = toModelFactors(
      [
        { feature: 'membership_expire_date_max', value: 20170315, contribution: 1 },
        { feature: 'cancel_rate', value: null, contribution: 0.5 },
        { feature: 'not_in_contract', value: 3, contribution: 0.1 },
      ],
      -3,
      0.4,
    );
    expect(date.text).toBe('Membership expires: 2017-03-15');
    expect(missing.text).toBe('Cancel rate: missing');
    expect(unknown.text).toBe('not_in_contract: 3');
  });
});
