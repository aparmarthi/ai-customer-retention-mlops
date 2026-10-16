import { describe, expect, it } from 'vitest';
import { runSanityChecks } from '@/lib/sanity';
import { logSpace, sensitivity, tpAtK } from '@/lib/economics';
import { CURVE } from '@/data/validation';

describe('documented sanity values', () => {
  for (const c of runSanityChecks()) {
    it(c.label, () => {
      expect(Math.abs(c.actual - c.expected)).toBeLessThanOrEqual(c.tolerance);
    });
  }
});

describe('tpAtK interpolation', () => {
  it('returns 0 for non-positive K', () => {
    expect(tpAtK(0)).toBe(0);
    expect(tpAtK(-5)).toBe(0);
  });

  it('hits every curve point exactly', () => {
    for (const p of CURVE) expect(tpAtK(p.k)).toBeCloseTo(p.tp, 9);
  });

  it('is monotonic non-decreasing', () => {
    let prev = 0;
    for (let k = 1; k <= CURVE[CURVE.length - 1].k + 1000; k += 97) {
      const tp = tpAtK(k);
      expect(tp).toBeGreaterThanOrEqual(prev);
      prev = tp;
    }
  });

  it('caps at the last curve point', () => {
    const last = CURVE[CURVE.length - 1];
    expect(tpAtK(last.k * 10)).toBe(last.tp);
  });
});

describe('sensitivity', () => {
  it('sorts bars by swing size, largest first', () => {
    const { bars } = sensitivity(1478, { valuePerSave: 120, costPerContact: 5, saveRate: 0.2 });
    const swings = bars.map((b) => Math.abs(b.high - b.low));
    expect([...swings].sort((a, b) => b - a)).toEqual(swings);
  });
});

describe('logSpace', () => {
  it('spans the range inclusively', () => {
    const xs = logSpace(100, 50000, 25);
    expect(xs).toHaveLength(25);
    expect(xs[0]).toBe(100);
    expect(xs[24]).toBe(50000);
  });
});
